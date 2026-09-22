#!/usr/bin/env python3
import json
import os
import time
import urllib.request

LOCAL_URL = "http://127.0.0.1:11434/api/chat"
CLOUD_URL = "https://ollama.com/api/chat"

def _post_json(url, payload, timeout, headers=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
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
):
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    force_cloud = os.environ.get("AGENTIA_OLLAMA_CLOUD", "").strip() == "1"
    use_cloud = force_cloud or model.endswith(":cloud") or model.endswith("-cloud")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "seed": seed,
        },
    }

    headers = {}
    url = LOCAL_URL
    if use_cloud:
        if not api_key:
            raise RuntimeError(
                "cloud model requested but OLLAMA_API_KEY is missing"
            )
        url = os.environ.get("OLLAMA_CLOUD_URL", CLOUD_URL).strip() or CLOUD_URL
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        payload["keep_alive"] = keep_alive
        payload["options"]["num_thread"] = 4

    last_error = None
    for attempt in range(1, 3):
        try:
            data = _post_json(url, payload, timeout, headers=headers)
            message = data.get("message") or {}
            content = (message.get("content") or "").strip()
            if not content:
                raise RuntimeError("empty model response")
            meta = {
                "attempt": attempt,
                "transport": "cloud" if use_cloud else "local",
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
