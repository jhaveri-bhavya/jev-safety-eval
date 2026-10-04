"""Smoke test: Hof University's LISA (Open WebUI) as an LLM-judge host.

LISA is at https://chat-1.ki-awz.iisys.de (found via go.hof-university.de/lisa). The API key is an
Open WebUI key (`sk-` + 32 hex), created in LISA under Settings -> Account -> API keys.
Open WebUI's OpenAI-compatible routes are /api/models and /api/chat/completions.

Uses the raw vLLM models, not the dated `lisa-*-2026` wrappers, which add tools and web search.
"""

import time

import httpx
from smoke_gemma import SCHEMA, parse_json, schema_valid
from smoke_hf import CASES, p_unsafe

from jev_safety.config import LISA_API_KEY

BASE_URL = "https://chat-1.ki-awz.iisys.de/api"
MODELS = ["lisa-flash", "lisa-pro"]


def main() -> None:
    headers = {"Authorization": f"Bearer {LISA_API_KEY}"}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60) as client:
        listed = {m["id"] for m in client.get("/models").json()["data"]}
        for model in MODELS:
            print(f"=== {model} (listed: {model in listed}) ===")
            for case, messages in CASES.items():
                body = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0,
                    "max_tokens": 64,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "verdict", "schema": SCHEMA, "strict": True},
                    },
                    "logprobs": True,
                    "top_logprobs": 5,
                    # Qwen 3.x thinks by default; vLLM reads this to switch it off
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                start = time.perf_counter()
                try:
                    resp = client.post("/chat/completions", json=body)
                except httpx.TimeoutException:
                    print(f"  {case}: timed out after {time.perf_counter() - start:.0f} s, skipping model")
                    break
                latency_s = time.perf_counter() - start
                if resp.is_error:
                    print(f"  {case}: HTTP {resp.status_code} in {latency_s:.2f} s: {resp.text[:300]}")
                    continue
                data = resp.json()
                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                logprobs = choice.get("logprobs")
                usage = data.get("usage") or {}
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                print(f"  {case}: {latency_s:.2f} s | {content!r}")
                print(
                    f"    schema-valid={schema_valid(parse_json(content))} "
                    f"logprobs={bool(logprobs and logprobs.get('content'))} "
                    f"P(unsafe)={p_unsafe(logprobs)} "
                    f"tokens={usage.get('prompt_tokens')}/{usage.get('completion_tokens')} "
                    f"reasoning_chars={len(reasoning)} finish={choice.get('finish_reason')}"
                )


if __name__ == "__main__":
    main()
