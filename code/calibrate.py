"""Calibrate the real token cost on a small pilot before any batch is allowed to run.

CLAUDE.md hard rule 5 requires a quote built from real token counts, confirmed by the user,
before a paid batch runs. This script produces that quote:

  1. verifies the API key with one minimal call (16 output tokens);
  2. measures the zero-shot and RAG prompt sizes LOCALLY with tiktoken on the real sample, so
     those two conditions need no API call to cost;
  3. runs the agent on a small stratified pilot (default 50 bugs x 3 seeds) because the agent's
     step count and output length cannot be known without asking the model;
  4. reports measured tokens and USD per bug at the list prices copied on 2026-10-12, and
     extrapolates to the full planned sweep.

It refuses to do anything billable unless --yes-spend is passed.

Usage:
  python calibrate.py --model deepseek-flash --n 50 --seeds 0,1,2            # quote only, no spend
  python calibrate.py --model deepseek-flash --n 50 --seeds 0,1,2 --yes-spend
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import prompts                                    # noqa: E402
from llm_client import PRICES, DeepSeek, KEY_ENV  # noqa: E402
from run_llm import stratified_sample             # noqa: E402

import tiktoken                                   # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")


def ntok(text):
    return len(ENC.encode(text, disallowed_special=()))


def local_prompt_costs(dataset, components, label, topn, split, n, seed, k, label_key):
    """Exact input-prompt sizes for zero-shot and RAG, measured locally. No API call."""
    rows = prompts.load_rows(dataset)
    classes = prompts.label_space(rows, label, topn)
    block = prompts.label_block(components, classes, label)
    sample = stratified_sample(rows, classes, label_key, split, n, seed)
    # retrieval is needed for the RAG prompt size; reuse the same index the run uses
    from retrieval import Retriever, text_of
    retriever = Retriever(rows, label_key=label_key, text_mode="sd")
    zs, rag = [], []
    for row in sample:
        text = text_of(row)
        zs.append(sum(ntok(m["content"]) for m in prompts.messages_zeroshot(block, text)))
        examples = retriever.examples_for(text, k)
        rag.append(sum(ntok(m["content"])
                       for m in prompts.messages_rag(block, text, examples, label_key)))
    return {"n_sampled": len(sample), "label_block_tokens": ntok(block),
            "zeroshot_input_mean": float(np.mean(zs)) if zs else 0.0,
            "rag_input_mean": float(np.mean(rag)) if rag else 0.0,
            "assumed_output_tokens": 150}


def read_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset.jsonl"))
    ap.add_argument("--components", default=os.path.join(ROOT, "data", "components.json"))
    ap.add_argument("--label", default="team")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--full-n", type=int, default=1000, help="bugs in the full planned sweep")
    ap.add_argument("--yes-spend", action="store_true")
    ap.add_argument("--max-spend", type=float, default=5.0, help="hard stop for the pilot itself")
    a = ap.parse_args()

    label_key = prompts.LABEL_KEYS[a.label]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    print("=== step 1: local prompt-size measurement (no API call) ===", flush=True)
    local = local_prompt_costs(a.dataset, a.components, a.label, a.topn, a.split, a.n, seeds[0],
                               a.k, label_key)
    print(json.dumps(local, indent=1), flush=True)

    report = {"model": a.model, "prices_copied": "2026-10-12",
              "price_source": "https://api-docs.deepseek.com/quick_start/pricing",
              "local_prompt_measurement": local, "pilot": {}, "full_sweep_estimate": {}}

    if not os.environ.get(KEY_ENV):
        print(f"\nABORT: environment variable {KEY_ENV} is not set, so the agent pilot cannot run "
              f"and the full sweep cannot be quoted from measured cost. The prompt sizes printed "
              f"above are real local measurements and need no key. Nothing was called and nothing "
              f"was spent.")
        with open(os.path.join(ROOT, "results", "cost_calibrated.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        return 2

    if not a.yes_spend:
        print("\nQUOTE ONLY: --yes-spend was not passed, so no API call was made.")
        print("Re-run with --yes-spend once the budget is confirmed.")
        with open(os.path.join(ROOT, "results", "cost_calibrated.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        return 0

    print("\n=== step 2: verify the key with one minimal call ===", flush=True)
    client = DeepSeek(a.model)
    reply = client.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                        max_tokens=16)
    print("key OK, reply:", repr(reply["content"])[:80], "usage:", reply["usage"], flush=True)

    print("\n=== step 3: agent pilot ===", flush=True)
    pilot_records = []
    for seed in seeds:
        out = os.path.join(ROOT, "results", f"llm_agent_{a.model}_{a.split}_n{a.n}_seed{seed}.jsonl")
        cmd = [sys.executable, os.path.join(HERE, "run_llm.py"), "--condition", "agent",
               "--model", a.model, "--dataset", a.dataset, "--components", a.components,
               "--label", a.label, "--topn", str(a.topn), "--split", a.split, "--n", str(a.n),
               "--seed", str(seed), "--max-spend", str(a.max_spend)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=HERE)
        pilot_records += read_records(out)

    if not pilot_records:
        print("ABORT: the pilot produced no records; cannot calibrate.")
        return 1

    pt = np.array([r.get("prompt_tokens", 0) for r in pilot_records], dtype=float)
    ct = np.array([r.get("completion_tokens", 0) for r in pilot_records], dtype=float)
    rounds = np.array([r.get("rounds", 1) for r in pilot_records], dtype=float)
    tools = np.array([r.get("n_tool_calls", 0) for r in pilot_records], dtype=float)
    price = PRICES[a.model]

    def cost(n_in, n_out):
        return n_in / 1e6 * price["miss"] + n_out / 1e6 * price["out"]

    agent_in, agent_out = float(pt.mean()), float(ct.mean())
    per_bug_agent = cost(agent_in, agent_out)
    per_bug_zs = cost(local["zeroshot_input_mean"], local["assumed_output_tokens"])
    per_bug_rag = cost(local["rag_input_mean"], local["assumed_output_tokens"])
    # full sweep: zero-shot + RAG + agent x len(seeds) + self-consistency k=5 on RAG
    per_bug_full = per_bug_zs + per_bug_rag + len(seeds) * per_bug_agent + 5 * per_bug_rag

    report["pilot"] = {
        "n_records": len(pilot_records), "seeds": seeds,
        "rounds_mean": float(rounds.mean()), "tool_calls_mean": float(tools.mean()),
        "prompt_tokens_mean": agent_in, "completion_tokens_mean": agent_out,
        "parse_ok_rate": float(np.mean([r.get("parse_ok", False) for r in pilot_records])),
        "acc_top1_pilot": float(np.mean([bool(r.get("top3")) and r["top3"][0] == r["true_team"]
                                         for r in pilot_records])),
        "usd_per_bug_agent": per_bug_agent,
        "usd_pilot_total": cost(pt.sum(), ct.sum()),
    }
    report["full_sweep_estimate"] = {
        "n_bugs": a.full_n,
        "usd_per_bug_zeroshot": per_bug_zs,
        "usd_per_bug_rag": per_bug_rag,
        "usd_per_bug_agent": per_bug_agent,
        "usd_per_bug_full_sweep": per_bug_full,
        "usd_full_sweep": per_bug_full * a.full_n,
        "usd_full_sweep_both_models_rough": per_bug_full * a.full_n * 3.0,
        "note": ("Output tokens are the least certain input and dominate the agent cost; the pilot "
                 "measured them instead of assuming them. Peak hours are charged double."),
    }
    with open(os.path.join(ROOT, "results", "cost_calibrated.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["pilot"], indent=1))
    print(json.dumps(report["full_sweep_estimate"], indent=1))
    print("wrote results/cost_calibrated.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
