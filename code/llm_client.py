"""Minimal DeepSeek chat-completions client with an on-disk response cache.

Design notes
------------
* Every request is keyed by the SHA-256 of its exact JSON payload, so an interrupted run
  replays cached responses instead of paying for them a second time (CLAUDE.md hard rule 6).
  The cache is content-addressed and lives in cache/, which .gitignore excludes.
* The API key is read from an environment variable only. It is never written to disk, never
  logged, and never committed (CLAUDE.md hard rule 8).
* Every genuinely billed call appends its token usage to results/usage_<model>.jsonl, so the
  spend can be audited against the quoted estimate.
* MockClient is a deterministic offline responder used by the --mock drivers to exercise the
  whole pipeline (including the tool loop) without any network access or any cost.

Usage:
    from llm_client import make_client
    client = make_client("deepseek-flash")
    reply = client.chat([{"role": "user", "content": "hi"}])
    print(reply["content"], reply["usage"])
"""
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(ROOT, "cache")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
KEY_ENV = os.environ.get("DEEPSEEK_KEY_ENV", "DEEPSEEK_API_KEY")

# USD per 1e6 tokens, off-peak, copied from https://api-docs.deepseek.com/quick_start/pricing
# on 2026-10-12. Peak hours are charged double.
PRICES = {
    "deepseek-flash": {"hit": 0.003, "miss": 0.15, "out": 0.60},
    "deepseek-v4-pro": {"hit": 0.022, "miss": 0.66, "out": 1.98},
}
PEAK = 2.0

# The usage log is appended by every call and read by the spend guard, so with --workers > 1 both
# happen from several threads at once. Two writers must not interleave into one file, and a reader
# must not catch a half-written line.
_USAGE_LOCK = threading.Lock()


class LLMError(RuntimeError):
    pass


def _hash(payload):
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _Base:
    model = "?"
    mock = False

    def chat(self, messages, tools=None, temperature=None, max_tokens=2048,
             response_format=None, seed=None):
        raise NotImplementedError

    def spend_usd(self):
        return 0.0


class MockClient(_Base):
    """Deterministic offline responder. Never touches the network, never costs anything.

    On the first turn of a tool-enabled conversation it asks for one tool call, so the agent
    loop in run_llm.py is exercised end to end; afterwards it returns a strict-JSON answer.
    """

    mock = True

    def __init__(self, model="mock"):
        self.model = model
        self.calls = 0

    def chat(self, messages, tools=None, temperature=None, max_tokens=2048,
             response_format=None, seed=None):
        self.calls += 1
        seeded = seed if seed is not None else self.calls
        digest = int(hashlib.sha256(f"{seeded}|{len(messages)}".encode()).hexdigest()[:8], 16)
        # Pull real class names out of the "[Class]" sections of the prompt so the mock exercises
        # the same label-validating parse path that a live answer goes through.
        text = "\n".join(m.get("content") or "" for m in messages)
        candidates, seen = [], set()
        for line in text.splitlines():
            line = line.strip()
            if len(line) > 2 and line.startswith("[") and line.endswith("]"):
                label = line[1:-1].strip()
                if label and label not in seen:
                    seen.add(label)
                    candidates.append(label)
        if not candidates:
            candidates = ["UNKNOWN"]
        picks = [candidates[(digest + off) % len(candidates)] for off in range(3)]
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tools and not tool_results:
            name = tools[0]["function"]["name"]
            args = {"query": "mock"} if name == "search_similar_bugs" else {"name": "Layout"}
            return {"content": "", "reasoning": "", "cached": False, "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "call_mock_1", "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        answer = {"top3": picks, "abstain": digest % 7 == 0,
                  "confidence": round((digest % 100) / 100.0, 2), "evidence": "mock evidence"}
        return {"content": json.dumps(answer), "reasoning": "", "cached": False,
                "finish_reason": "stop", "tool_calls": [],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


class DeepSeek(_Base):
    """OpenAI-compatible DeepSeek chat client with a content-addressed disk cache."""

    def __init__(self, model="deepseek-flash", use_cache=True, max_retries=5, timeout=180,
                 cache_dir=CACHE_DIR, thinking="disabled", reasoning_effort=None):
        self.model = model
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_dir = cache_dir
        # Left unset, the API turns thinking on by itself. The measured probe
        # (results/probe_thinking.json) showed 800 reasoning tokens and an empty, truncated answer
        # with thinking on, against 22 tokens and a correct JSON answer with it off. It is sent
        # explicitly so every record states which mode produced it, and because it is part of the
        # request payload it is also part of the cache key, so the two modes never mix in the cache.
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.usage_path = os.path.join(ROOT, "results", f"usage_{model}.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)

    # -- cache ---------------------------------------------------------------
    def _cache_path(self, key):
        return os.path.join(self.cache_dir, f"{self.model}__{key}.json")

    def _load(self, key):
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            reply = rec["response"]
            reply["cached"] = True
            return reply
        except Exception:  # a corrupt cache entry must never break a run
            return None

    def _store(self, key, payload, response):
        rec = {"model": self.model, "request": payload, "response": response,
               "stored": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = self._cache_path(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False)
        os.replace(tmp, self._cache_path(key))

    # -- accounting ----------------------------------------------------------
    def _account(self, response):
        usage = response.get("usage") or {}
        if not usage:
            return
        os.makedirs(os.path.dirname(self.usage_path), exist_ok=True)
        # `ts` is local time, which is how the earliest runs recorded it. `ts_utc` is what a
        # peak-hour calculation needs, because DeepSeek prices on UTC and this machine is UTC+8:
        # reading the local stamp as if it were UTC made 1,413 peak-billed calls look off-peak.
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "model": self.model, "usage": usage}) + "\n"
        with _USAGE_LOCK:
            with open(self.usage_path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    def spend_usd(self, peak=False):
        """Spend implied by results/usage_<model>.jsonl, at the copied list prices."""
        if not os.path.exists(self.usage_path):
            return 0.0
        p = PRICES.get(self.model)
        if not p:
            return float("nan")
        hit = miss = out = 0
        with _USAGE_LOCK:
            with open(self.usage_path, encoding="utf-8") as fh:
                lines = fh.readlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                u = json.loads(line)["usage"]
            except (ValueError, KeyError):
                continue          # a torn line from a concurrent writer is not fatal
            hit += u.get("prompt_cache_hit_tokens", 0)
            miss += u.get("prompt_cache_miss_tokens",
                          u.get("prompt_tokens", 0) - u.get("prompt_cache_hit_tokens", 0))
            out += u.get("completion_tokens", 0)
        cost = miss / 1e6 * p["miss"] + hit / 1e6 * p["hit"] + out / 1e6 * p["out"]
        return cost * (PEAK if peak else 1.0)

    # -- HTTP ----------------------------------------------------------------
    def _payload(self, messages, tools, temperature, max_tokens, response_format, seed):
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed
        return payload

    def _request(self, payload):
        key = os.environ.get(KEY_ENV)
        if not key:
            raise LLMError(f"environment variable {KEY_ENV} is not set; refusing to call the API")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            BASE_URL + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        last = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return self._normalise(json.load(resp))
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf-8", "replace")
                last = f"HTTP {exc.code}: {detail}"
                if exc.code in (400, 401, 403, 404, 422):
                    raise LLMError(last)          # a bad key or bad request will not fix itself
                if exc.code == 429:
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
            except Exception as exc:  # noqa: BLE001 - network hiccups are retried
                last = repr(exc)
            time.sleep(min(60, 2 ** attempt))
        raise LLMError(f"giving up after {self.max_retries} attempts; last error: {last}")

    @staticmethod
    def _normalise(data):
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return {"content": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or "",
                "tool_calls": message.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage") or {},
                "cached": False}

    def chat(self, messages, tools=None, temperature=None, max_tokens=2048,
             response_format=None, seed=None):
        payload = self._payload(messages, tools, temperature, max_tokens, response_format, seed)
        key = _hash(payload)
        hit = self._load(key)
        if hit is not None:
            return hit
        response = self._request(payload)
        self._store(key, payload, response)
        self._account(response)
        return response


def make_client(model, mock=False, use_cache=True, thinking="disabled", reasoning_effort=None):
    if mock:
        return MockClient(model)
    return DeepSeek(model, use_cache=use_cache, thinking=thinking,
                    reasoning_effort=reasoning_effort)
