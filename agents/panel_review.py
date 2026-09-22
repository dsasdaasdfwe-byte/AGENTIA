#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import time

from ollama_client import chat
from prompts import build_panel_review, build_panel_audit

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
    p.add_argument("--panel", required=True)
    p.add_argument("--roles", required=True)
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--case", required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--out", default="work-panel")
    args = p.parse_args()

    roles = [x.strip() for x in args.roles.split(",") if x.strip()]
    if len(roles) != 5:
        raise SystemExit(f"expected 5 roles, found {len(roles)}")

    case_text = Path(args.case).read_text(encoding="utf-8")
    mission_text = Path(args.mission).read_text(encoding="utf-8")
    max_line = max(1, len(case_text.splitlines()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()

    system_prompt, user_prompt = build_panel_review(
        args.panel, roles, mission_text, case_text, args.reports_dir
    )
    draft, draft_meta = chat(
        args.model,
        system_prompt,
        user_prompt,
        num_predict=2400,
        num_ctx=32768,
        temperature=0.04,
        seed=5000 + int(args.panel),
        timeout=1800,
        keep_alive="10m",
    )

    audit_system, audit_user = build_panel_audit(
        args.panel, mission_text, case_text, draft
    )
    answer, audit_meta = chat(
        args.model,
        audit_system,
        audit_user,
        num_predict=3200,
        num_ctx=32768,
        temperature=0.02,
        seed=6000 + int(args.panel),
        timeout=1800,
        keep_alive="10m",
    )

    refs, invalid = validate_citations(answer, max_line)
    if audit_meta.get("done_reason") == "length":
        raise RuntimeError("panel output was truncated; refusing partial synthesis")
    if invalid:
        raise RuntimeError(f"panel produced invalid citations: {invalid[:10]}")
    if refs < 8:
        raise RuntimeError(
            f"panel insufficiently grounded: only {refs} unique source lines cited"
        )

    elapsed = round(time.monotonic() - started, 2)
    report_path = out / f"panel-{args.panel}.md"
    metrics_path = out / f"panel-{args.panel}-metrics.json"
    report_path.write_text(answer.rstrip() + "\n", encoding="utf-8")

    metrics = {
        "panel": args.panel,
        "roles": roles,
        "model": args.model,
        "elapsed_seconds": elapsed,
        "source_lines": max_line,
        "unique_source_lines_cited": refs,
        "status": "ok",
        "draft": draft_meta,
        "audit": audit_meta,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"panel {args.panel}: OK in {elapsed}s; source_lines_cited={refs}", flush=True)

if __name__ == "__main__":
    main()
