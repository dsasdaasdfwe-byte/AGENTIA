#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time

from ollama_client import chat
from prompts import build_review

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
    if len(reports) != 20:
        raise SystemExit(f"expected 20 reports, found {len(reports)}")

    case_text = Path(args.case).read_text(encoding="utf-8")
    mission_text = Path(args.mission).read_text(encoding="utf-8")
    system_prompt, user_prompt = build_review(
        mission_text,
        case_text,
        reports_dir,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    answer, meta = chat(
        args.model,
        system_prompt,
        user_prompt,
        num_predict=1200,
        num_ctx=24576,
        temperature=0.10,
        seed=4242,
        timeout=1200,
        keep_alive="5m",
    )
    elapsed = round(time.monotonic() - started, 2)

    (out / "final-report.md").write_text(answer + "\n", encoding="utf-8")
    meta.update({
        "model": args.model,
        "reports_count": len(reports),
        "elapsed_seconds": elapsed,
        "status": "ok",
    })
    (out / "reviewer-metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"reviewer: OK in {elapsed}s")

if __name__ == "__main__":
    main()
