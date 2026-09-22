#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import time

from ollama_client import chat
from prompts import build_agent, build_agent_audit, build_agent_repair

CITE_RE = re.compile(r"\[L(\d{4})(?:-L(\d{4}))?\]")

def citation_quality(text, max_line):
    refs = []
    invalid = []
    for m in CITE_RE.finditer(text):
        a = int(m.group(1))
        b = int(m.group(2) or m.group(1))
        if 1 <= a <= b <= max_line:
            refs.extend(range(a, b + 1))
        else:
            invalid.append(m.group(0))
    return len(set(refs)), invalid

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--case", required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--out", default="work-agent")
    args = p.parse_args()

    case_text = Path(args.case).read_text(encoding="utf-8")
    mission_text = Path(args.mission).read_text(encoding="utf-8")
    max_line = max(1, len(case_text.splitlines()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{args.role}.md"
    metrics_path = out / f"{args.role}-metrics.json"

    started = time.monotonic()
    system_prompt, user_prompt = build_agent(args.role, mission_text, case_text)
    draft, draft_meta = chat(
        args.model,
        system_prompt,
        user_prompt,
        num_predict=2200,
        num_ctx=32768,
        temperature=0.08,
        seed=1000 + sum(ord(c) for c in args.role),
        timeout=1200,
    )

    audit_system, audit_user = build_agent_audit(
        args.role, mission_text, case_text, draft
    )
    answer, audit_meta = chat(
        args.model,
        audit_system,
        audit_user,
        num_predict=3000,
        num_ctx=32768,
        temperature=0.02,
        seed=7000 + sum(ord(c) for c in args.role),
        timeout=1200,
    )

    unique_refs, invalid_refs = citation_quality(answer, max_line)
    needs_repair = (
        audit_meta.get("done_reason") == "length"
        or unique_refs < 5
        or bool(invalid_refs)
    )

    repair_meta = None
    if needs_repair:
        repair_system, repair_user = build_agent_repair(
            args.role, case_text, answer
        )
        answer, repair_meta = chat(
            args.model,
            repair_system,
            repair_user,
            num_predict=3000,
            num_ctx=32768,
            temperature=0.02,
            seed=9000 + sum(ord(c) for c in args.role),
            timeout=1200,
        )
        unique_refs, invalid_refs = citation_quality(answer, max_line)

    final_meta = repair_meta if repair_meta is not None else audit_meta
    if final_meta.get("done_reason") == "length":
        raise RuntimeError("final agent report was truncated; refusing partial output")
    if invalid_refs:
        raise RuntimeError(f"invalid source citations: {invalid_refs[:10]}")
    if unique_refs < 5:
        raise RuntimeError(
            f"insufficient source grounding: only {unique_refs} unique source lines cited"
        )

    elapsed = round(time.monotonic() - started, 2)
    report_path.write_text(answer.rstrip() + "\n", encoding="utf-8")

    metrics = {
        "role": args.role,
        "model": args.model,
        "elapsed_seconds": elapsed,
        "status": "ok",
        "source_lines": max_line,
        "unique_source_lines_cited": unique_refs,
        "draft": draft_meta,
        "audit": audit_meta,
        "repair": repair_meta,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.role}: OK in {elapsed}s; source_lines_cited={unique_refs}",
        flush=True,
    )

if __name__ == "__main__":
    main()
