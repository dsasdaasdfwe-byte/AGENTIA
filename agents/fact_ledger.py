#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import time

from ollama_client import chat
from grounding_prompts import fact_ledger_extract_section, fact_ledger_audit_section


FACT_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "subject", "predicate", "object",
        "source_lines"
    ],
    "properties": {
        "id": {"type": "string"},
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
        "source_lines": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1}
        },
    },
}

PROCEDURE_EDGE_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "actor", "action", "target",
        "source_lines"
    ],
    "properties": {
        "id": {"type": "string"},
        "actor": {"type": "string"},
        "action": {"type": "string"},
        "target": {"type": "string"},
        "source_lines": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1}
        },
    },
}

ISSUE_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "issue", "authority", "status",
        "source_lines"
    ],
    "properties": {
        "id": {"type": "string"},
        "issue": {"type": "string"},
        "authority": {"type": "string"},
        "status": {
            "type": "string",
            "enum": [
                "DECIDED", "NOT_EXAMINED",
                "SUBSIDIARY_REASONING", "PARTY_ARGUMENT",
                "UNRESOLVED"
            ],
        },
        "source_lines": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1}
        },
    },
}

SECTION_CONFIG = {
    "facts": {
        "prefix": "F",
        "min_items": 5,
        "max_items": 15,
        "compressed_max_items": 12,
        "item_schema": FACT_ITEM_SCHEMA,
        "draft_num_predict": 1200,
        "audit_num_predict": 1000,
        "compressed_draft_num_predict": 900,
        "compressed_audit_num_predict": 850,
        "seed": 21001,
    },
    "procedure_edges": {
        "prefix": "P",
        "min_items": 2,
        "max_items": 12,
        "compressed_max_items": 9,
        "item_schema": PROCEDURE_EDGE_ITEM_SCHEMA,
        "draft_num_predict": 900,
        "audit_num_predict": 800,
        "compressed_draft_num_predict": 700,
        "compressed_audit_num_predict": 650,
        "seed": 21101,
    },
    "issues": {
        "prefix": "I",
        "min_items": 1,
        "max_items": 8,
        "compressed_max_items": 6,
        "item_schema": ISSUE_ITEM_SCHEMA,
        "draft_num_predict": 650,
        "audit_num_predict": 600,
        "compressed_draft_num_predict": 500,
        "compressed_audit_num_predict": 500,
        "seed": 21201,
    },
}


def section_schema(section, compressed=False):
    cfg = SECTION_CONFIG[section]
    max_items = (
        cfg["compressed_max_items"] if compressed else cfg["max_items"]
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [section],
        "properties": {
            section: {
                "type": "array",
                "minItems": cfg["min_items"],
                "maxItems": max_items,
                "items": cfg["item_schema"],
            }
        },
    }


LEDGER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts", "procedure_edges", "issues"],
    "properties": {
        section: section_schema(section)["properties"][section]
        for section in ("facts", "procedure_edges", "issues")
    },
}


def parse_json(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("model did not return a JSON object")
    return json.loads(raw[start:end + 1])


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


DATE_PATTERNS = [
    re.compile(
        r"\\b(?:1er|[0-3]?\\d)\\s+"
        r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
        r"septembre|octobre|novembre|décembre|decembre)\\s+\\d{4}\\b",
        re.IGNORECASE,
    ),
    re.compile(r"\\b\\d{4}-\\d{2}-\\d{2}\\b"),
    re.compile(r"\\b\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}\\b"),
]


def exact_dates(text):
    matches = []
    for pattern in DATE_PATTERNS:
        matches.extend((m.start(), m.group(0)) for m in pattern.finditer(text))
    matches.sort(key=lambda item: item[0])
    seen = set()
    values = []
    for _, value in matches:
        key = norm(value)
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def evidence_text(source_lines, line_numbers):
    parts = []
    for n in line_numbers:
        if not isinstance(n, int) or n < 1 or n > len(source_lines):
            raise RuntimeError(f"invalid ledger source line: {n}")
        parts.append(source_lines[n - 1])
    return "\n".join(parts)


def anchor_dates(data, source_lines, section):
    """Derive dates only from cited source text; ambiguous evidence stays undated."""
    if section not in {"facts", "procedure_edges"}:
        return 0
    if not isinstance(data, dict) or set(data) != {section}:
        raise RuntimeError(f"{section} block must contain only the '{section}' key")
    entries = data.get(section)
    if not isinstance(entries, list):
        raise RuntimeError(f"ledger missing list: {section}")

    rewrites = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lines = entry.get("source_lines")
        if not isinstance(lines, list) or not lines:
            continue
        evidence = evidence_text(source_lines, lines)
        candidates = exact_dates(evidence)
        anchored = candidates[0] if len(candidates) == 1 else None
        previous = entry.get("date_text")
        entry["date_text"] = anchored
        if previous != anchored:
            rewrites += 1
    return rewrites


def anchor_quotes(data, source_lines, section):
    """Attach exact cited source text; the model never authors ledger quotes."""
    if not isinstance(data, dict) or set(data) != {section}:
        raise RuntimeError(f"{section} block must contain only the '{section}' key")
    entries = data.get(section)
    if not isinstance(entries, list):
        raise RuntimeError(f"ledger missing list: {section}")

    rewrites = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lines = entry.get("source_lines")
        if not isinstance(lines, list) or not lines:
            continue
        evidence = evidence_text(source_lines, lines).strip()
        if not evidence:
            continue
        previous = str(entry.get("quote") or "").strip()
        entry["quote"] = evidence
        if previous != evidence:
            rewrites += 1
    return rewrites


def validate_entry(entry, source_lines, section):
    required_by_section = {
        "facts": {
            "id", "subject", "predicate", "object",
            "date_text", "source_lines", "quote"
        },
        "procedure_edges": {
            "id", "actor", "action", "target",
            "date_text", "source_lines", "quote"
        },
        "issues": {
            "id", "issue", "authority", "status",
            "source_lines", "quote"
        },
    }
    text_fields = {
        "facts": ("subject", "predicate", "object"),
        "procedure_edges": ("actor", "action", "target"),
        "issues": ("issue", "authority"),
    }

    if not isinstance(entry, dict):
        raise RuntimeError(f"{section} entry must be an object")
    missing = required_by_section[section] - set(entry)
    if missing:
        raise RuntimeError(f"{section} entry missing fields: {sorted(missing)}")
    for field in text_fields[section]:
        if not str(entry.get(field) or "").strip():
            raise RuntimeError(
                f"{section} {entry.get('id')} has empty field: {field}"
            )

    lines = entry.get("source_lines")
    if not isinstance(lines, list) or not lines:
        raise RuntimeError(f"{section} {entry.get('id')} has no source_lines")
    evidence = evidence_text(source_lines, lines)
    quote = str(entry.get("quote") or "").strip()
    if len(quote) < 8:
        raise RuntimeError(f"{section} {entry.get('id')} quote too short")
    if norm(quote) not in norm(evidence):
        raise RuntimeError(
            f"{section} {entry.get('id')} quote is not contained in cited source lines"
        )
    date_text = entry.get("date_text")
    if date_text and norm(date_text) not in norm(evidence):
        raise RuntimeError(
            f"{section} {entry.get('id')} date_text absent from cited evidence"
        )


def validate_section(data, source_lines, section, *, compressed=False):
    if section not in SECTION_CONFIG:
        raise RuntimeError(f"unknown ledger section: {section}")
    if not isinstance(data, dict) or set(data) != {section}:
        raise RuntimeError(f"{section} block must contain only the '{section}' key")
    entries = data.get(section)
    if not isinstance(entries, list):
        raise RuntimeError(f"ledger missing list: {section}")

    cfg = SECTION_CONFIG[section]
    max_items = (
        cfg["compressed_max_items"] if compressed else cfg["max_items"]
    )
    if len(entries) < cfg["min_items"]:
        raise RuntimeError(f"{section} contains too few entries")
    if len(entries) > max_items:
        raise RuntimeError(
            f"{section} contains too many entries: {len(entries)} > {max_items}"
        )

    ids = set()
    prefix = cfg["prefix"]
    valid_status = {
        "DECIDED",
        "NOT_EXAMINED",
        "SUBSIDIARY_REASONING",
        "PARTY_ARGUMENT",
        "UNRESOLVED",
    }
    for entry in entries:
        validate_entry(entry, source_lines, section)
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id or entry_id in ids:
            raise RuntimeError(f"duplicate/empty ledger id: {entry_id}")
        if not entry_id.startswith(prefix):
            raise RuntimeError(
                f"{section} id has wrong prefix: {entry_id} (expected {prefix})"
            )
        ids.add(entry_id)
        if section == "issues" and entry.get("status") not in valid_status:
            raise RuntimeError(
                f"invalid issue status {entry.get('status')} for {entry_id}"
            )


def validate_ledger(data, source_lines):
    if not isinstance(data, dict) or set(data) != {
        "facts", "procedure_edges", "issues"
    }:
        raise RuntimeError("ledger root must contain facts, procedure_edges and issues")

    ids = set()
    for section in ("facts", "procedure_edges", "issues"):
        block = {section: data[section]}
        validate_section(block, source_lines, section)
        for entry in data[section]:
            entry_id = entry["id"]
            if entry_id in ids:
                raise RuntimeError(f"duplicate ledger id across sections: {entry_id}")
            ids.add(entry_id)


def _call_section_model(
    model,
    case_text,
    section,
    *,
    stage,
    draft=None,
    compressed=False,
):
    cfg = SECTION_CONFIG[section]
    if stage == "draft":
        system, user = fact_ledger_extract_section(
            case_text, section, compressed=compressed
        )
        num_predict = cfg[
            "compressed_draft_num_predict" if compressed else "draft_num_predict"
        ]
        seed = cfg["seed"]
    elif stage == "audit":
        if draft is None:
            raise RuntimeError(f"missing draft for {section} audit")
        model_draft = [
            {k: v for k, v in entry.items() if k not in {"quote", "date_text"}}
            for entry in draft
        ]
        system, user = fact_ledger_audit_section(
            case_text,
            section,
            json.dumps(
                {section: model_draft},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            compressed=compressed,
        )
        num_predict = cfg[
            "compressed_audit_num_predict" if compressed else "audit_num_predict"
        ]
        seed = cfg["seed"] + 1
    else:
        raise RuntimeError(f"unknown ledger stage: {stage}")

    if compressed:
        seed += 50

    return chat(
        model,
        system,
        user,
        num_predict=num_predict,
        num_ctx=32768,
        temperature=0.0 if stage == "audit" else 0.01,
        seed=seed,
        timeout=1800,
        keep_alive="10m",
        json_schema=section_schema(section, compressed=compressed),
    )


def generate_validated_block(model, case_text, source_lines, section, *, stage, draft=None):
    attempts = []
    for compressed in (False, True):
        mode = "compressed" if compressed else "normal"
        print(f"ledger: {section} {stage} {mode} start", flush=True)
        raw, meta = _call_section_model(
            model,
            case_text,
            section,
            stage=stage,
            draft=draft,
            compressed=compressed,
        )
        attempt = {"compressed": compressed, **meta}
        attempts.append(attempt)
        print(
            f"ledger: {section} {stage} {mode} "
            f"done_reason={meta.get('done_reason')} eval_count={meta.get('eval_count')}",
            flush=True,
        )
        if meta.get("done_reason") == "length":
            if compressed:
                raise RuntimeError(
                    f"{section} {stage} was truncated after compression retry"
                )
            continue

        try:
            data = parse_json(raw)
            date_rewrites = anchor_dates(data, source_lines, section)
            quote_rewrites = anchor_quotes(data, source_lines, section)
            attempt["date_rewrites"] = date_rewrites
            attempt["quote_rewrites"] = quote_rewrites
            validate_section(data, source_lines, section, compressed=compressed)
        except (RuntimeError, json.JSONDecodeError) as exc:
            attempt["validation_error"] = str(exc)
            print(
                f"ledger: {section} {stage} {mode} invalid: {exc}",
                flush=True,
            )
            if compressed:
                raise RuntimeError(
                    f"{section} {stage} invalid after compression retry: {exc}"
                ) from exc
            continue

        print(
            f"ledger: {section} {stage} {mode} validated "
            f"entries={len(data[section])} date_rewrites={date_rewrites} "
            f"quote_rewrites={quote_rewrites}",
            flush=True,
        )
        return data[section], attempts

    raise RuntimeError(f"{section} {stage} did not produce a complete block")


def render_markdown(data):
    lines = ["# Verified Fact Ledger", ""]
    lines.append("## Facts")
    for x in data["facts"]:
        date = f" — {x.get('date_text')}" if x.get("date_text") else ""
        refs = ",".join(f"L{n:04d}" for n in x["source_lines"])
        lines.append(
            f"- {x['id']}: {x.get('subject','')} | {x.get('predicate','')} | "
            f"{x.get('object','')}{date} [{refs}] — \"{x.get('quote','')}\""
        )
    lines.extend(["", "## Procedural graph"])
    for x in data["procedure_edges"]:
        date = f" — {x.get('date_text')}" if x.get("date_text") else ""
        refs = ",".join(f"L{n:04d}" for n in x["source_lines"])
        lines.append(
            f"- {x['id']}: {x.get('actor','')} -> {x.get('action','')} -> "
            f"{x.get('target','')}{date} [{refs}] — \"{x.get('quote','')}\""
        )
    lines.extend(["", "## Issue states"])
    for x in data["issues"]:
        refs = ",".join(f"L{n:04d}" for n in x["source_lines"])
        lines.append(
            f"- {x['id']} [{x.get('status')}]: {x.get('issue','')} | "
            f"authority={x.get('authority','')} [{refs}] — \"{x.get('quote','')}\""
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--case", required=True)
    p.add_argument("--out", default="work-ledger")
    args = p.parse_args()

    case_text = Path(args.case).read_text(encoding="utf-8")
    source_lines = case_text.splitlines()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    data = {}
    block_metrics = {}
    for section in ("facts", "procedure_edges", "issues"):
        draft, draft_attempts = generate_validated_block(
            args.model,
            case_text,
            source_lines,
            section,
            stage="draft",
        )
        audited, audit_attempts = generate_validated_block(
            args.model,
            case_text,
            source_lines,
            section,
            stage="audit",
            draft=draft,
        )
        data[section] = audited
        block_metrics[section] = {
            "draft_attempts": draft_attempts,
            "audit_attempts": audit_attempts,
            "entries": len(audited),
        }

    validate_ledger(data, source_lines)

    elapsed = round(time.monotonic() - started, 2)
    (out / "fact-ledger.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "fact-ledger.md").write_text(render_markdown(data), encoding="utf-8")
    metrics = {
        "model": args.model,
        "facts": len(data["facts"]),
        "procedure_edges": len(data["procedure_edges"]),
        "issues": len(data["issues"]),
        "source_lines": len(source_lines),
        "elapsed_seconds": elapsed,
        "blocks": block_metrics,
        "status": "ok",
    }
    (out / "fact-ledger-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"ledger: OK facts={metrics['facts']} edges={metrics['procedure_edges']} "
        f"issues={metrics['issues']} in {elapsed}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
