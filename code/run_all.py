"""Run the whole step-2 pipeline in order, so the paper's reproducibility claim is one command.

Every step writes into results/ or data/, and each one is idempotent: re-running replays the
existing outputs rather than duplicating them. The LLM steps are the only ones that cost money,
so they are skipped unless --with-llm is passed AND the budget guard is satisfied.

This script never invents numbers: it only calls the other scripts in this directory and reports
what they wrote.

Usage:
  python run_all.py                 # everything that is free
  python run_all.py --with-llm      # also run the LLM conditions (requires DEEPSEEK_API_KEY)
  python run_all.py --with-llm --max-spend 150
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PY = sys.executable


def run(label, args, skip=False):
    if skip:
        print(f"\n=== SKIP {label} ===", flush=True)
        return True
    print(f"\n=== {label} ===\n$ {' '.join(args)}", flush=True)
    t0 = time.time()
    result = subprocess.run(args, cwd=HERE)
    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"--- {label}: {status} in {time.time() - t0:.0f}s", flush=True)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset_all.jsonl"))
    ap.add_argument("--split", default="test2026")
    ap.add_argument("--settled", default="2026-07-14",
                    help="creation cutoff for the settled-subset robustness check")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-spend", type=float, default=150.0)
    a = ap.parse_args()

    python = lambda script, *rest: [PY, os.path.join(HERE, script), *rest]  # noqa: E731
    ok = True

    # --- data -----------------------------------------------------------------
    if not os.path.exists(a.dataset):
        run("build 2026 dataset", python("build_2026.py"))
        run("merge datasets", python("merge_datasets.py"))
    run("clean ambiguity labels", python("label_cleaning.py", "--dataset", a.dataset, "--n", "150"))
    run("label stability", python("label_stability.py"))

    # --- baselines ------------------------------------------------------------
    for split, extra in ((a.split, []), (a.split, ["--max-created", a.settled])):
        run(f"baselines {split} {'settled' if extra else 'full'}",
            python("baselines_ext.py", "--dataset", a.dataset, "--test-split", split,
                   "--skip-optional", *extra))
    run("embedding kNN baseline",
        python("baseline_embed.py", "--dataset", a.dataset, "--test-split", a.split))
    run("score every prediction file", python("score_all.py"))
    run("significance tests", python("significance.py", "--split", a.split))
    run("figures", python("make_figures.py", "--split", a.split))
    run("tables", python("make_tables.py", "--split", a.split))

    # --- LLM conditions (the only paid part) ----------------------------------
    if not a.with_llm:
        print("\nLLM conditions skipped (pass --with-llm to include them).", flush=True)
    elif not os.environ.get("DEEPSEEK_API_KEY"):
        print("\nLLM conditions skipped: DEEPSEEK_API_KEY is not set.", flush=True)
        ok = False
    else:
        for seed in [s.strip() for s in a.seeds.split(",") if s.strip()]:
            for condition in ("zeroshot", "rag", "agent"):
                run(f"{condition} seed {seed}",
                    python("run_llm.py", "--condition", condition, "--model", a.model,
                           "--dataset", a.dataset, "--split", a.split, "--n", str(a.n),
                           "--seed", seed, "--max-spend", str(a.max_spend)))
        run("score LLM records", python("score_all.py"))
        run("tables again", python("make_tables.py", "--split", a.split))

    print("\nrun_all finished.", "Some steps failed." if not ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
