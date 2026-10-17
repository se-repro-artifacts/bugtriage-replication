"""Score every prediction file in results/ with the single evaluator in evaluate.py.

There is exactly one scorer on purpose: a classical baseline and an LLM condition must go through
the same code, or their risk-coverage curves are not comparable. Mock runs are skipped, because
they only exercise the pipeline and must never reach a paper table.

Usage: python score_all.py
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+",
                    default=[os.path.join(ROOT, "results", "pred_*.jsonl"),
                             os.path.join(ROOT, "results", "llm_*.jsonl")])
    a = ap.parse_args()
    files = set()
    for pattern in a.patterns:
        files.update(glob.glob(pattern))
    files = sorted(f for f in files if "mock" not in os.path.basename(f))
    if not files:
        print("no prediction files found")
        return 1
    failed = []
    for path in files:
        result = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"),
                                 "--records", path], cwd=HERE)
        if result.returncode != 0:
            failed.append(path)
    print(f"scored {len(files)} prediction files, {len(failed)} failed")
    for path in failed:
        print("  FAILED:", path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
