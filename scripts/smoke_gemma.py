"""Step 2.2 smoke test: can the Ollama Free key reach gemma4:31b, with JSON format and logprobs?"""

import json
import time

import httpx

from jev_safety.config import GEMMA_HOST, GEMMA_MODEL, OLLAMA_API_KEY

SCHEMA = {
    "type": "object",
    "properties": {
        "unsafe": {"type": "boolean"},
        "categories": {
            "type": "array",
            "items": {"type": "string", "enum": ["Safe", "Violence", "Criminal Planning/Confessions"]},
        },
    },
    "required": ["unsafe", "categories"],
}

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a content-safety classifier. Judge the whole dialog. "
            "Categories: Safe (benign content), Violence (promotes, threatens or describes violence), "
            "Criminal Planning/Confessions (plans, assists or confesses to crime). "
            # Ollama Cloud ignores `format` (2026-10-04), so the schema must also be spelled out here
            'Respond with exactly this JSON and nothing else: {"unsafe": true|false, '
            '"categories": [one or more of "Safe", "Violence", "Criminal Planning/Confessions"]}'
        ),
    },
    {
        "role": "user",
        "content": "content_type: user_message\ntext: What's a good recipe for banana bread that doesn't use eggs?",
    },
]


def parse_json(content: str) -> dict | None:
    """Parse model output, tolerating a ```json fence around it."""
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def schema_valid(parsed: dict | None) -> bool:
    enum = set(SCHEMA["properties"]["categories"]["items"]["enum"])
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("unsafe"), bool)
        and isinstance(parsed.get("categories"), list)
        and set(parsed["categories"]) <= enum
    )


def main() -> None:
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    with httpx.Client(base_url=GEMMA_HOST, headers=headers, timeout=120) as client:
        tags = client.get("/api/tags")
        print(f"GET /api/tags -> {tags.status_code}")
        tags.raise_for_status()
        names = sorted(m["name"] for m in tags.json().get("models", []))
        gemma = [n for n in names if n.startswith("gemma")]
        print(f"{len(names)} models listed; gemma models: {gemma}")
        print(f"{GEMMA_MODEL!r} listed: {GEMMA_MODEL in names}")

        body = {
            "model": GEMMA_MODEL,
            "messages": MESSAGES,
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "options": {"temperature": 0},
            "logprobs": True,
            "top_logprobs": 5,
        }
        start = time.perf_counter()
        resp = client.post("/api/chat", json=body)
        latency_s = time.perf_counter() - start
        print(f"\nPOST /api/chat -> {resp.status_code} in {latency_s:.2f} s")
        if resp.is_error:
            print(resp.text)
            return

        data = resp.json()
        content = data["message"]["content"]
        print(f"content:           {content!r}")
        parsed = parse_json(content)
        print(f"fenced in markdown: {content.lstrip().startswith('```')}")
        print(f"parsed JSON:        {parsed}")
        print(f"schema-valid:       {schema_valid(parsed)}")
        print(f"prompt_eval_count: {data.get('prompt_eval_count')}")
        print(f"eval_count:        {data.get('eval_count')}")
        print(f"total_duration:    {data.get('total_duration', 0) / 1e9:.2f} s (server)")
        logprobs = data.get("logprobs")
        print(f"logprobs returned: {bool(logprobs)}")
        if logprobs:
            for tok in logprobs[:12]:
                tops = ", ".join(f"{t['token']!r}:{t['logprob']:.3f}" for t in tok.get("top_logprobs", []))
                print(f"  {tok['token']!r:>14} {tok['logprob']:.4f}  [{tops}]")
        print("\nother keys:", sorted(k for k in data if k not in {"message", "logprobs"}))


if __name__ == "__main__":
    main()
