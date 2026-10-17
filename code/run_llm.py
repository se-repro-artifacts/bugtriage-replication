"""Run one LLM condition over a bug sample, with disk caching, resumability and a spend guard.

Conditions
  zeroshot  one call: team list + bug report.
  rag       one call: team list + k retrieved already-routed bugs + bug report.
  agent     tool-using loop (search_similar_bugs / get_component_info / recent_bugs_for_team),
            then a final JSON answer.

Every API response is cached by llm_client keyed on the exact request, so re-running after an
interruption replays cached responses instead of paying again. Records are appended one line
per bug and flushed as they are produced; a re-run skips bug ids already present.

Usage:
  python run_llm.py --condition agent --model deepseek-flash --n 50 --seed 0 --mock
  python run_llm.py --condition agent --model deepseek-flash --n 1000 --seed 0 --max-spend 150
"""
import argparse
import json
import os
import random
import sys
import time

import prompts
import tools as tools_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import LLMError, make_client
from offpeak import wait_for_offpeak
from retrieval import Retriever, text_of

ROOT = os.path.join(os.path.dirname(__file__), "..")


def stratified_sample(rows, classes, label_key, split, n, seed, max_created=None):
    """Proportional stratified sample over the true class, deterministic for a given seed.

    The sample is NESTED in n: the bugs chosen for a small n are a prefix of those chosen for a
    larger n. That matters because a long sweep is split across days under a token budget, and each
    day has to continue where the previous one stopped rather than draw a different subset.
    `random.sample(items, take)` is NOT nested -- a 1000-item draw is not a prefix of a 6347-item
    draw -- so each class's items are sorted, shuffled once, and then taken as a prefix.
    """
    pool = [r for r in rows if r["split"] == split and r[label_key] in set(classes)
            and (max_created is None or r["created"][:10] <= max_created)]
    if n is None or n >= len(pool):
        # A whole-pool run is also split across days under a token budget, so it needs a fixed
        # order too: the raw dataset order may track filing time, which would make any prefix a
        # biased slice. A seeded shuffle fixes the order and makes every prefix an unbiased sample.
        pool = sorted(pool, key=lambda r: str(r["id"]))
        random.Random(seed).shuffle(pool)
        return pool
    rng = random.Random(seed)
    by_class = {}
    for row in pool:
        by_class.setdefault(row[label_key], []).append(row)
    total = len(pool)
    picked = []
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda r: str(r["id"]))   # deterministic, then shuffled
        rng.shuffle(items)
        take = max(1, round(n * len(items) / total))
        picked.extend(items[:min(take, len(items))])                  # a prefix, so nesting holds
    rng.shuffle(picked)
    return picked[:n]


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return ids


def run_agent(client, messages, retriever, bug, classes, max_rounds, temperature, max_tokens):
    """Tool loop: keep answering until the model stops asking for tools or rounds run out."""
    msgs = list(messages)
    calls, usage_in, usage_out, cached_rounds, rounds = [], 0, 0, 0, 0
    for _ in range(max_rounds):
        reply = client.chat(msgs, tools=tools_mod.TOOLS, temperature=temperature,
                            max_tokens=max_tokens)
        rounds += 1
        usage = reply.get("usage") or {}
        usage_in += usage.get("prompt_tokens", 0)
        usage_out += usage.get("completion_tokens", 0)
        cached_rounds += 1 if reply.get("cached") else 0
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return reply, calls, usage_in, usage_out, cached_rounds, rounds
        msgs.append({"role": "assistant", "content": reply.get("content") or "",
                     "tool_calls": tool_calls})
        for tc in tool_calls:
            function = tc.get("function") or {}
            name = function.get("name")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except Exception:
                args = {}
            result = tools_mod.dispatch(name, args, retriever, bug["_text"], set(classes))
            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or name, "content": result})
            calls.append({"name": name, "args": args})
    msgs.append({"role": "user", "content": "Stop using tools and reply with the JSON object now."})
    reply = client.chat(msgs, tools=None, temperature=temperature, max_tokens=max_tokens)
    rounds += 1
    usage = reply.get("usage") or {}
    usage_in += usage.get("prompt_tokens", 0)
    usage_out += usage.get("completion_tokens", 0)
    cached_rounds += 1 if reply.get("cached") else 0
    return reply, calls, usage_in, usage_out, cached_rounds, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["zeroshot", "rag", "agent"])
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset.jsonl"))
    ap.add_argument("--components", default=os.path.join(ROOT, "data", "components.json"))
    ap.add_argument("--label", default="team", choices=["fine", "team", "coarse"])
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-created", default=None,
                    help="keep only bugs created on or before this date (YYYY-MM-DD)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="retrieved examples for RAG")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled"],
                    help="Sent explicitly. The API defaults to enabled, which in the probe spent "
                         "800 reasoning tokens and returned an empty answer at the cap, against 22 "
                         "tokens and a correct answer when disabled (results/probe_thinking.json).")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="Only meaningful with --thinking enabled.")
    ap.add_argument("--peak-policy", default="wait", choices=["wait", "allow"],
                    help="DeepSeek charges double during peak hours (01:00-04:00 and 06:00-10:00 "
                         "UTC, Mon-Fri). 'wait' sleeps until the cheap window before spending "
                         "anything; 'allow' runs now and pays double. See code/offpeak.py.")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many NEW bugs")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="Stop once this many input+output tokens have been used by THIS run. A "
                         "long sweep is split across days under a budget, so bugs are processed in "
                         "waves and a wave starts only when every bug in it still fits under the "
                         "budget, which keeps the total at or below it.")
    ap.add_argument("--workers", type=int, default=1,
                    help="bugs processed concurrently. Sequential runs of the agent take about 20 s "
                         "per bug, so a 1000-bug sweep is hours without this. The retriever only "
                         "reads after it is built and the client serialises its usage log, so "
                         "workers do not share mutable state.")
    ap.add_argument("--max-spend", type=float, default=None, help="abort once spend exceeds USD")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    label_key = prompts.LABEL_KEYS[a.label]
    rows = prompts.load_rows(a.dataset)
    classes = prompts.label_space(rows, a.label, a.topn)
    block = prompts.label_block(a.components, classes, a.label)
    sampled = stratified_sample(rows, classes, label_key, a.split, a.n, a.seed, a.max_created)
    for row in sampled:
        row["_text"] = text_of(row)

    tag = f"{a.condition}_{a.model}_{a.split}"
    # thinking mode and the output cap change what the records mean, so they belong in the file
    # name: without them the two modes collide and the resume logic silently skips one of them.
    tag += f"_{a.thinking}"
    if a.reasoning_effort:
        tag += f"-{a.reasoning_effort}"
    if a.max_tokens != 2048:
        tag += f"_mt{a.max_tokens}"
    if a.max_created:
        tag += f"_settled{a.max_created}"
    if a.n:
        tag += f"_n{a.n}"
    tag += f"_seed{a.seed}"
    if a.mock:
        tag += "_mock"
    out_path = os.path.join(ROOT, "results", f"llm_{tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    already = done_ids(out_path)

    retriever = None
    if a.condition in ("rag", "agent"):
        retriever = Retriever(rows, label_key=label_key, text_mode="sd")

    client = make_client(a.model, mock=a.mock, use_cache=not a.no_cache,
                         thinking=a.thinking, reasoning_effort=a.reasoning_effort)
    todo = [r for r in sampled if r["id"] not in already]
    print(f"{tag}: {len(sampled)} sampled, {len(already)} already done, {len(todo)} to run; "
          f"{len(classes)} classes, label block {len(block)} chars", flush=True)
    if todo and a.peak_policy == "wait":
        wait_for_offpeak()
    # The usage log is cumulative across every run ever made, so the cap has to be measured from
    # where this run started. Comparing the running total against --max-spend meant a fresh run
    # stopped after its first record as soon as earlier sessions had pushed the log past the cap.
    spend0 = client.spend_usd()

    def process_one(bug):
        """One bug through one condition.

        Runs in a worker thread, so it must touch no shared mutable state. It does not: the
        retriever is read-only after construction, the client serialises its own usage log, and the
        disk cache writes each key by atomic replace.
        """
        t0 = time.time()
        calls, usage_in, usage_out, cached_rounds, rounds = [], 0, 0, 0, 0
        error = None
        try:
            if a.condition == "zeroshot":
                reply = client.chat(prompts.messages_zeroshot(block, bug["_text"]),
                                    temperature=a.temperature, max_tokens=a.max_tokens)
                u = reply.get("usage") or {}
                usage_in, usage_out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                cached_rounds, rounds = (1 if reply.get("cached") else 0), 1
            elif a.condition == "rag":
                examples = retriever.examples_for(bug["_text"], a.k)
                reply = client.chat(prompts.messages_rag(block, bug["_text"], examples, label_key),
                                    temperature=a.temperature, max_tokens=a.max_tokens)
                u = reply.get("usage") or {}
                usage_in, usage_out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                cached_rounds, rounds = (1 if reply.get("cached") else 0), 1
            else:
                reply, calls, usage_in, usage_out, cached_rounds, rounds = run_agent(
                    client, prompts.messages_agent(block, bug["_text"]), retriever, bug,
                    classes, a.max_rounds, a.temperature, a.max_tokens)
        except LLMError as exc:
            error = str(exc)
            reply = {"content": "", "cached": False}

        parsed = prompts.parse_answer(reply.get("content", ""), classes)
        return {"id": bug["id"], "split": bug["split"], "created": bug["created"],
                "true_team": bug[label_key], "true_fine": bug["label_fine"],
                "comp_changed": bug["comp_changed"], "initial_label": bug["initial_label"],
                "condition": a.condition, "model": a.model, "seed": a.seed,
                "thinking": a.thinking, "reasoning_effort": a.reasoning_effort,
                "top3": parsed["top3"], "abstain": parsed["abstain"],
                "confidence": parsed["confidence"], "evidence": parsed["evidence"],
                "parse_ok": parsed["parse_ok"], "unknown_labels": parsed["unknown_labels"],
                "n_tool_calls": len(calls), "tool_calls": calls[:20], "rounds": rounds,
                "prompt_tokens": usage_in, "completion_tokens": usage_out,
                "cached_rounds": cached_rounds,
                "latency_s": round(time.time() - t0, 2), "error": error}

    workers = max(1, a.workers)
    started = time.time()
    written = 0
    used_tokens = 0
    fh = open(out_path, "a", encoding="utf-8")
    pool = ThreadPoolExecutor(max_workers=workers)
    # Worst case for one bug, deliberately above the measured means (100-bug A/B, thinking off:
    # zeroshot 9.3k, rag 11.5k, agent 29.3k tokens). A wave starts only when every bug in it still
    # fits under the budget, so the run total stays at or below it even with workers in flight.
    per_bug_worst = {"zeroshot": 25000, "rag": 25000, "agent": 90000}.get(a.condition, 40000)
    try:
        n_total = len(todo)
        while todo:
            if a.token_budget is not None:
                remaining = a.token_budget - used_tokens
                if remaining < per_bug_worst:
                    print(f"STOP: {used_tokens:,} tokens used and the next bug could need up to "
                          f"{per_bug_worst:,}; staying under --token-budget "
                          f"{a.token_budget:,}", flush=True)
                    break
                wave_n = min(len(todo), workers, max(1, remaining // per_bug_worst))
            else:
                wave_n = min(len(todo), workers)
            wave, todo = todo[:wave_n], todo[wave_n:]

            for record in pool.map(process_one, wave):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                used_tokens += record.get("prompt_tokens", 0) + record.get("completion_tokens", 0)

            rate = written / max(1e-9, time.time() - started)
            print(f"  {written}/{n_total} done | this run ${client.spend_usd() - spend0:.3f} | "
                  f"{used_tokens:,} tokens | {rate:.2f} bug/s", flush=True)
            if a.max_spend is not None and (client.spend_usd() - spend0) > a.max_spend:
                print(f"STOP: this run spent ${client.spend_usd() - spend0:.3f}, over --max-spend "
                      f"{a.max_spend}", flush=True)
                break
            if a.limit is not None and written >= a.limit:
                print(f"STOP: reached --limit {a.limit}", flush=True)
                break
    finally:
        fh.close()
        pool.shutdown(wait=True)

    print(f"wrote {written} records to {out_path}")
    print(f"this run: {used_tokens:,} tokens, ${client.spend_usd() - spend0:.4f} | "
          f"cumulative log: ${client.spend_usd():.4f} (list price, off-peak)")
    if a.mock:
        print("MOCK MODE: no network call was made and no money was spent.")


if __name__ == "__main__":
    main()
