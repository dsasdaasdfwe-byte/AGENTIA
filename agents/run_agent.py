#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time

from ollama_client import chat
from prompts import build_agent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--case", required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--out", default="work-agent")
    args = p.parse_args()

    case_text = Path(args.case).read_text(encoding="utf-8")
    mission_text = Path(args.mission).read_text(encoding="utf-8")
    system_prompt, user_prompt = build_agent(args.role, mission_text, case_text)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{args.role}.md"
    metrics_path = out / f"{args.role}-metrics.json"

    started = time.monotonic()
    answer, meta = chat(
        args.model,
        system_prompt,
        user_prompt,
        num_predict=520,
        num_ctx=16384,
        temperature=0.22,
        seed=1000 + sum(ord(c) for c in args.role),
        timeout=900,
    )
    elapsed = round(time.monotonic() - started, 2)

    report_path.write_text(answer + "\n", encoding="utf-8")
    meta.update({
        "role": args.role,
        "model": args.model,
        "elapsed_seconds": elapsed,
        "status": "ok",
    })
    metrics_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{args.role}: OK in {elapsed}s")

if __name__ == "__main__":
    main()
