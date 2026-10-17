"""Probe how DeepSeek's thinking mode affects output tokens, before committing to a big run.

The measured pilot showed 913-1,924 completion tokens per bug against an assumed 150, and the
thinking mode is on by default, so output is the dominant cost. The official docs list two
controls that llm_client.py never sends:

    "thinking": {"type": "enabled"}      and      "reasoning_effort": "high"

This script sends a handful of tiny requests with those controls varied and reports the usage the
API returns, so the choice is made on measured numbers rather than on the assumption that thinking
is needed for a classification answer. It costs a fraction of a cent.

Usage: python probe_thinking.py
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
KEY_ENV = "DEEPSEEK_API_KEY"

PROMPT = ('A bug report about a WebGL canvas crash was filed under Core::Graphics. '
          'Reply with JSON only, no prose: {"team": "...", "confidence": 0.0, "abstain": false}')

VARIANTS = [
    ("thinking disabled", {"thinking": {"type": "disabled"}}),
    ("thinking enabled, effort low", {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}),
    ("default (nothing sent)", {}),
]


def call(key, model, extra, max_tokens=800):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, **extra}).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req) as fh:
        data = json.load(fh)
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "seconds": round(time.time() - t0, 1),
        "content": (data["choices"][0]["message"].get("content") or "").strip()[:90],
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def main():
    key = os.environ.get(KEY_ENV)
    if not key:
        print(f"ABORT: {KEY_ENV} is not set in this process.")
        return 2
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-flash"
    rows = []
    for label, extra in VARIANTS:
        try:
            result = call(key, model, extra)
            result["variant"] = label
            result["sent"] = extra
        except Exception as exc:                       # noqa: BLE001 - report, never crash the probe
            body = ""
            if hasattr(exc, "read"):
                body = exc.read().decode("utf-8", "replace")[:300]
            result = {"variant": label, "sent": extra, "error": type(exc).__name__,
                      "detail": (body or str(exc))[:300]}
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "probe_thinking.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": model, "prompt": PROMPT, "variants": rows}, fh, indent=1)
    print("wrote results/probe_thinking.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
