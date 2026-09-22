#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import time

from ollama_client import chat
from prompts import build_review, build_review_audit

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--case", required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--out", default="work-review")
    args = p.parse_args()

    reports_dir = Path(args.reports_dir)
    reports = sorted(reports_dir.glob("*.md"))
    if len(reports) != 4:
        raise SystemExit(f"expected 4 panel reports, found {len(reports)}")

    case_text = Path(args.case).read_text(encoding="utf-8")
    mission_text = Path(args.mission).read_text(encoding="utf-8")
    max_line = max(1, len(case_text.splitlines()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()

    system_prompt, user_prompt = build_review(
        mission_text, case_text, reports_dir
    )
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
    if invalid:
        raise RuntimeError(f"reviewer produced invalid citations: {invalid[:10]}")
    if refs < 8:
        raise RuntimeError(
            f"reviewer insufficiently grounded: only {refs} unique source lines cited"
        )
    if audit_meta.get("done_reason") == "length":
        raise RuntimeError("reviewer output was truncated; refusing partial final report")

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
    }
    (out / "reviewer-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"reviewer: OK in {elapsed}s; source_lines_cited={refs}", flush=True)

if __name__ == "__main__":
    main()
