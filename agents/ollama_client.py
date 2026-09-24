#!/usr/bin/env python3
import json
import time
import urllib.request

def _post_json(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def chat(
    model,
    system_prompt,
    user_prompt,
    *,
    num_predict,
    num_ctx,
    temperature,
    seed,
    timeout,
    keep_alive="10m",
    json_mode=False,
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "seed": seed,
            "num_thread": 4,
        },
    }

    if json_mode:
        payload["format"] = "json"

    last_error = None
    for attempt in range(1, 3):
        try:
            data = _post_json(
                "http://127.0.0.1:11434/api/chat",
                payload,
                timeout,
            )
            message = data.get("message") or {}
            content = (message.get("content") or "").strip()
            if not content:
                raise RuntimeError("empty model response")
            meta = {
                "attempt": attempt,
                "transport": "local",
                "model": model,
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "done_reason": data.get("done_reason"),
            }
            return content, meta
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(3)

    raise RuntimeError(f"Ollama request failed after 2 attempts: {last_error}")
