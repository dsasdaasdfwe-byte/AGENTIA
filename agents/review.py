#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import time

from ollama_client import chat
from prompts import build_review, build_review_audit, build_review_compress, build_review_semantic_check, build_review_semantic_repair
from semantic_gate import run_atomic_semantic_gate

CITE_RE = re.compile(r"\[L(\d{4})(?:-L(\d{4}))?\]")

def validate_citations(text, max_line):
    refs = set()
    invalid = []
    for m in CITE_RE.finditer(text):
        a = int(m.group(1))
        b = int(m.group(2) or m.group(1))
        if 1 <= a <= b <= max_line:
            refs.update(range(a, b + 1))
        else:
            invalid.append(m.group(0))
    return len(refs), invalid

MONTHS = (
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    "septembre|octobre|novembre|décembre|decembre"
)
DATE_WORD_RE = re.compile(rf"\\b(?:[0-3]?\\d)\\s+(?:{MONTHS})\\s+20\\d{{2}}\\b", re.I)
DATE_NUM_RE = re.compile(r"\\b(?:[0-3]?\\d)[./-](?:[01]?\\d)[./-](?:19|20)\\d{2}\\b")
YEAR_RE = re.compile(r"\\b(?:19|20)\\d{2}\\b")

def _citation_ranges(text):
    ranges = []
    for m in CITE_RE.finditer(text):
        a = int(m.group(1))
        b = int(m.group(2) or m.group(1))
        ranges.append((a, b))
    return ranges

def _evidence_for_claim(claim, source_lines):
    chunks = []
    seen = set()
    for a, b in _citation_ranges(claim):
        for n in range(a, b + 1):
            if 1 <= n <= len(source_lines) and n not in seen:
                seen.add(n)
                chunks.append(f"[L{n:04d}] {source_lines[n-1]}")
    return "\n".join(chunks)

def extract_cited_claims(report):
    claims = []
    for raw in report.splitlines():
        line = raw.strip()
        if not line or not CITE_RE.search(line):
            continue
        if line.startswith("#"):
            continue
        claims.append(line)
    return claims

def deterministic_date_violations(claims, source_lines):
    violations = []
    for idx, claim in enumerate(claims, start=1):
        evidence = _evidence_for_claim(claim, source_lines)
        claim_without_cites = CITE_RE.sub("", claim)
        dates = set(DATE_WORD_RE.findall(claim_without_cites))
        dates.update(DATE_NUM_RE.findall(claim_without_cites))
        # Years are checked only when no fuller date captured for that year.
        full_years = {m.group(0)[-4:] for m in DATE_WORD_RE.finditer(claim_without_cites)}
        full_years.update({m.group(0)[-4:] for m in DATE_NUM_RE.finditer(claim_without_cites)})
        years = {y for y in YEAR_RE.findall(claim_without_cites) if y not in full_years}
        for token in sorted(dates | years):
            if token.lower() not in evidence.lower():
                violations.append({
                    "id": idx,
                    "status": "UNSUPPORTED_DATE",
                    "reason": f"date/année '{token}' absente des lignes citées",
                })
    return violations

def build_claim_bundle(report, source_lines):
    claims = extract_cited_claims(report)
    blocks = []
    for idx, claim in enumerate(claims, start=1):
        evidence = _evidence_for_claim(claim, source_lines)
        blocks.append(
            f"CLAIM {idx}\n{claim}\nEVIDENCE\n{evidence}\n"
        )
    return claims, "\n".join(blocks)

def parse_semantic_violations(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("semantic checker did not return JSON")
    data = json.loads(raw[start:end + 1])
    violations = data.get("violations")
    if not isinstance(violations, list):
        raise RuntimeError("semantic checker JSON missing violations list")
    return violations

def run_semantic_gate(model, mission_text, case_text, report):
    source_lines = case_text.splitlines()
    history = []
    answer = report
    for round_no in range(1, 4):
        claims, bundle = build_claim_bundle(answer, source_lines)
        if not claims:
            raise RuntimeError("semantic gate found no cited claims")

        deterministic = deterministic_date_violations(claims, source_lines)
        check_system, check_user = build_review_semantic_check(case_text, bundle)
        check_raw, check_meta = chat(
            model,
            check_system,
            check_user,
            num_predict=2200,
            num_ctx=32768,
            temperature=0.0,
            seed=12000 + round_no,
            timeout=1800,
            keep_alive="10m",
        )
        model_violations = parse_semantic_violations(check_raw)
        violations = deterministic + model_violations
        history.append({
            "round": round_no,
            "claims_checked": len(claims),
            "deterministic_violations": deterministic,
            "model_violations": model_violations,
            "checker": check_meta,
        })
        if not violations:
            return answer, history

        if round_no == 3:
            raise RuntimeError(
                f"semantic gate still failing after repairs: {violations[:8]}"
            )

        repair_system, repair_user = build_review_semantic_repair(
            mission_text,
            case_text,
            answer,
            json.dumps(violations, ensure_ascii=False, indent=2),
        )
        answer, repair_meta = chat(
            model,
            repair_system,
            repair_user,
            num_predict=4200,
            num_ctx=32768,
            temperature=0.01,
            seed=13000 + round_no,
            timeout=1800,
            keep_alive="10m",
        )
        history[-1]["repair"] = repair_meta
        if repair_meta.get("done_reason") == "length":
            raise RuntimeError("semantic repair was truncated")
    raise RuntimeError("unreachable semantic gate state")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--case", required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--ledger")
    p.add_argument("--out", default="work-review")
    args = p.parse_args()

    reports_dir = Path(args.reports_dir)
    reports = sorted(reports_dir.glob("*.md"))
    if len(reports) != 4:
        raise SystemExit(f"expected 4 panel reports, found {len(reports)}")

    case_text = Path(args.case).read_text(encoding="utf-8")
    mission_text = Path(args.mission).read_text(encoding="utf-8")
    ledger_text = Path(args.ledger).read_text(encoding="utf-8") if args.ledger else ""
    max_line = max(1, len(case_text.splitlines()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()

    system_prompt, user_prompt = build_review(
        mission_text, case_text, reports_dir
    )
    if ledger_text:
        user_prompt += "\n\nLEDGER FACTUEL VÉRIFIÉ + GRAPHE PROCÉDURAL\n" + ledger_text
    draft, draft_meta = chat(
        args.model,
        system_prompt,
        user_prompt,
        num_predict=3500,
        num_ctx=32768,
        temperature=0.02,
        seed=4242,
        timeout=1800,
        keep_alive="10m",
    )

    audit_system, audit_user = build_review_audit(
        mission_text, case_text, draft
    )
    if ledger_text:
        audit_user += "\n\nLEDGER FACTUEL VÉRIFIÉ + GRAPHE PROCÉDURAL\n" + ledger_text
    answer, audit_meta = chat(
        args.model,
        audit_system,
        audit_user,
        num_predict=4000,
        num_ctx=32768,
        temperature=0.02,
        seed=4343,
        timeout=1800,
        keep_alive="10m",
    )

    refs, invalid = validate_citations(answer, max_line)
    compress_meta = None
    if audit_meta.get("done_reason") == "length":
        compress_system, compress_user = build_review_compress(case_text, answer)
        answer, compress_meta = chat(
            args.model,
            compress_system,
            compress_user,
            num_predict=3000,
            num_ctx=32768,
            temperature=0.01,
            seed=4444,
            timeout=1800,
            keep_alive="10m",
        )
        refs, invalid = validate_citations(answer, max_line)
        if compress_meta.get("done_reason") == "length":
            raise RuntimeError("compact final reviewer report was still truncated")
    if invalid:
        raise RuntimeError(f"reviewer produced invalid citations: {invalid[:10]}")
    if refs < 8:
        raise RuntimeError(
            f"reviewer insufficiently grounded: only {refs} unique source lines cited"
        )

    answer, semantic_gate = run_atomic_semantic_gate(
        args.model, mission_text, case_text, ledger_text, answer
    )
    refs, invalid = validate_citations(answer, max_line)
    if invalid:
        raise RuntimeError(f"semantic-repaired report has invalid citations: {invalid[:10]}")
    if refs < 8:
        raise RuntimeError(
            f"semantic-repaired report insufficiently grounded: only {refs} source lines"
        )

    elapsed = round(time.monotonic() - started, 2)
    (out / "final-report.md").write_text(answer.rstrip() + "\n", encoding="utf-8")

    metrics = {
        "model": args.model,
        "panel_reports_count": len(reports),
        "source_lines": max_line,
        "unique_source_lines_cited": refs,
        "elapsed_seconds": elapsed,
        "status": "ok",
        "draft": draft_meta,
        "audit": audit_meta,
        "compression": compress_meta,
        "semantic_gate": semantic_gate,
    }
    (out / "reviewer-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "fidelity-metrics.json").write_text(
        json.dumps(semantic_gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"reviewer: OK in {elapsed}s; source_lines_cited={refs}", flush=True)

if __name__ == "__main__":
    main()
