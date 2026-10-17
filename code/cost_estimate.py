"""Token-count based cost model for the (not yet run) single-shot RAG and agent experiments.

No LLM API is called. Token counts use tiktoken's o200k_base encoding as a PROXY tokenizer;
other vendors' tokenizers (e.g. Claude, Qwen) will give different counts (typically within a
few tens of percent for English text, but this has not been measured here).

Measured from real data:
  Q  = tokens of the query bug (summary + description truncated to QUERY_CHARS)
  E  = tokens of one retrieved example (summary + description truncated to EX_CHARS + its label)
  L  = tokens of the label-space block (one line per class: name + BMO description)
Assumed (parameters, change freely):
  S  = fixed instruction tokens, K = retrieved examples, O = output tokens per call,
  T  = agent steps, R = tokens returned per tool call

Single-shot RAG per bug: in = S + L + Q + K*E ; out = O
Agent per bug (each step re-sends the growing context, no prompt caching):
  in = sum_{t=1..T} (S + L + Q + (t-1)*(R + O_step)) ; out = T * O_step
Cost = in/1e6 * p_in + out/1e6 * p_out  (prices per million tokens, from the vendor's page)
"""
import json, os, sys

import numpy as np
import tiktoken

ROOT = os.path.join(os.path.dirname(__file__), "..")
QUERY_CHARS, EX_CHARS = 4000, 1000
enc = tiktoken.get_encoding("o200k_base")


def ntok(s):
    return len(enc.encode(s, disallowed_special=()))


def main(label="team", topn=20):
    key = {"fine": "label_fine", "team": "label_team", "coarse": "label_coarse"}[label]
    res = json.load(open(os.path.join(ROOT, "results", f"baselines_{label}_top{topn}_sd.json")))
    classes = set(res["classes"])
    comps = json.load(open(os.path.join(ROOT, "data", "components.json")))
    # label block: for teams, list each team with its component names + short descriptions
    lines = {}
    for p in comps["products"]:
        for c in p["components"]:
            fine = f"{p['name']}::{c['name']}"
            lab = {"fine": fine, "team": c.get("team_name") or "?", "coarse": p["name"]}[label]
            if lab in classes:
                lines.setdefault(lab, []).append(f"{fine}: {(c.get('description') or '')[:200]}")
    L = ntok("\n".join(f"[{k}]\n" + "\n".join(v) for k, v in lines.items()))
    q, e, n_test = [], [], 0
    for l in open(os.path.join(ROOT, "data", "dataset.jsonl")):
        r = json.loads(l)
        if r[key] not in classes:
            continue
        d = r.get("description") or ""
        if r["split"] == "test":
            n_test += 1
            q.append(ntok(r["summary"] + "\n" + d[:QUERY_CHARS]))
        elif len(e) < 3000 and r["id"] % 3 == 0:  # deterministic sample of train examples
            e.append(ntok(r["summary"] + "\n" + d[:EX_CHARS] + "\nLabel: " + r[key]))
    Q, E = float(np.mean(q)), float(np.mean(e))
    S, K, O = 400, 10, 150          # assumptions
    T, R, O_step = 6, 1500, 200     # assumptions
    rag_in = S + L + Q + K * E
    rag_out = O
    agent_in = sum(S + L + Q + (t - 1) * (R + O_step) for t in range(1, T + 1))
    agent_out = T * O_step
    out = {"label": label, "topn": topn, "n_test": n_test, "tokenizer": "tiktoken o200k_base (proxy)",
           "measured": {"Q_mean": Q, "Q_median": float(np.median(q)), "Q_p95": float(np.percentile(q, 95)),
                        "E_mean": E, "L_label_block": L, "n_example_sample": len(e)},
           "assumed": {"S": S, "K": K, "O": O, "T": T, "R": R, "O_step": O_step},
           "per_bug": {"rag_in": rag_in, "rag_out": rag_out, "agent_in": agent_in, "agent_out": agent_out},
           "total_test_set_Mtok": {"rag_in": rag_in * n_test / 1e6, "rag_out": rag_out * n_test / 1e6,
                                   "agent_in": agent_in * n_test / 1e6, "agent_out": agent_out * n_test / 1e6}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", f"cost_{label}_top{topn}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "team", int(sys.argv[2]) if len(sys.argv) > 2 else 20)
