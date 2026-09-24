#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import time

from ollama_client import chat
from grounding_prompts import fact_ledger_extract, fact_ledger_audit


def parse_json(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("model did not return a JSON object")
    return json.loads(raw[start:end + 1])


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def evidence_text(source_lines, line_numbers):
    parts = []
    for n in line_numbers:
        if not isinstance(n, int) or n < 1 or n > len(source_lines):
            raise RuntimeError(f"invalid ledger source line: {n}")
        parts.append(source_lines[n - 1])
    return "\n".join(parts)


def validate_entry(entry, source_lines, section):
    required = {"id", "source_lines", "quote"}
    missing = required - set(entry)
    if missing:
        raise RuntimeError(f"{section} entry missing fields: {sorted(missing)}")
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


def validate_ledger(data, source_lines):
    if not isinstance(data, dict):
        raise RuntimeError("ledger root must be an object")
    for key in ("facts", "procedure_edges", "issues"):
        if not isinstance(data.get(key), list):
            raise RuntimeError(f"ledger missing list: {key}")

    if len(data["facts"]) < 5:
        raise RuntimeError("ledger contains too few facts")
    if len(data["procedure_edges"]) < 2:
        raise RuntimeError("ledger contains too few procedure edges")
    if len(data["issues"]) < 1:
        raise RuntimeError("ledger contains no legal issue states")

    valid_status = {
        "DECIDED",
        "NOT_EXAMINED",
        "SUBSIDIARY_REASONING",
        "PARTY_ARGUMENT",
        "UNRESOLVED",
    }
    ids = set()
    for section in ("facts", "procedure_edges", "issues"):
        for entry in data[section]:
            validate_entry(entry, source_lines, section)
            entry_id = str(entry.get("id") or "")
            if not entry_id or entry_id in ids:
                raise RuntimeError(f"duplicate/empty ledger id: {entry_id}")
            ids.add(entry_id)
            if section == "issues" and entry.get("status") not in valid_status:
                raise RuntimeError(
                    f"invalid issue status {entry.get('status')} for {entry_id}"
                )


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

    system, user = fact_ledger_extract(case_text)
    draft_raw, draft_meta = chat(
        args.model, system, user,
        num_predict=3800,
        num_ctx=32768,
        temperature=0.01,
        seed=21001,
        timeout=1800,
        keep_alive="10m",
    )
    draft = parse_json(draft_raw)

    audit_system, audit_user = fact_ledger_audit(
        case_text, json.dumps(draft, ensure_ascii=False, indent=2)
    )
    final_raw, audit_meta = chat(
        args.model, audit_system, audit_user,
        num_predict=4200,
        num_ctx=32768,
        temperature=0.0,
        seed=21002,
        timeout=1800,
        keep_alive="10m",
    )
    data = parse_json(final_raw)
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
        "draft": draft_meta,
        "audit": audit_meta,
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
