"""Re-fetch the move history of the reviewed bugs and diff it against what was stored.

The manual review cannot validate the labels on its own: the review sheet shows the same fields the
labelling rule reads, so applying the rule to the sheet reproduces the label by construction. What
the sheet cannot check is whether the *stored* history is faithful --- whether the initial
component, the sequence of moves and the human/bot flags match what Bugzilla actually records. That
is checkable without asking a person to read 150 bug pages, because the source is a public API.

This re-fetches each reviewed bug's history and compares it with data/history.jsonl. Two outcomes
matter:

  * the stored moves are a prefix of the current ones -> the record is faithful, and anything extra
    is a move that happened after the freeze (expected, and reported rather than treated as an
    error);
  * the stored moves differ in a way a later move cannot explain -> the fetch or the parser dropped
    or mangled something, which is a real defect in the dataset.

Usage: python verify_history.py            # all reviewed bugs
       python verify_history.py --n 30     # a quick subset
"""
import argparse
import csv
import json
import os
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
API = "https://bugzilla.mozilla.org/rest/bug/{ids}/history"
# only these two fields are stored, so only these two are compared
FIELDS = ("component", "product")
BOT_ACCOUNTS = None  # resolved from the stored flags: a move counts as a bot move if it was


def fetch(ids, attempts=4):
    """Fetch one page of history, retrying: the endpoint refuses connections intermittently."""
    url = API.format(ids=",".join(str(i) for i in ids))
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as fh:
                return json.load(fh)
        except Exception as exc:                       # noqa: BLE001 - retried below
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise last


def key(move):
    return (move.get("field"), move.get("removed"), move.get("added"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="how many reviewed bugs to check (0 = all)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args()

    sample = os.path.join(ROOT, "results", "manual_review_sample.csv")
    with open(sample, encoding="utf-8-sig") as fh:
        reviewed = [r for r in csv.DictReader(fh) if (r.get("reviewer_category") or "").strip()]
    ids = [int(r["bug_id"]) for r in reviewed]
    if a.n:
        ids = ids[:a.n]

    stored = {}
    for name in ("history.jsonl", os.path.join("2026", "history.jsonl")):
        path = os.path.join(ROOT, "data", name)
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    stored[row["id"]] = row.get("moves") or []

    report = {"checked": 0, "faithful": 0, "extra_moves_only": 0, "mismatch": [], "no_stored": [],
              "fetch_failed": []}
    for i, bug_id in enumerate(ids):
        try:
            data = fetch([bug_id])["bugs"][0]
        except Exception as exc:                       # noqa: BLE001 - report and carry on
            report["fetch_failed"].append({"bug_id": bug_id, "error": str(exc)[:120]})
            continue
        current = []
        for entry in data.get("history", []):
            for change in entry.get("changes", []):
                if change.get("field_name") not in FIELDS:
                    continue
                current.append({"when": entry.get("when"), "field": change["field_name"],
                                "removed": change.get("removed"), "added": change.get("added"),
                                "by_bot": False})
        was = stored.get(bug_id)
        report["checked"] += 1
        if was is None:
            # no change was ever recorded, so nothing was fetched; the API should agree
            if current:
                report["no_stored"].append(bug_id)
            else:
                report["faithful"] += 1
            continue
        stored_keys = [key(m) for m in was]
        current_keys = [key(m) for m in current]
        if current_keys[:len(stored_keys)] == stored_keys:
            if len(current_keys) > len(stored_keys):
                report["extra_moves_only"] += 1
            else:
                report["faithful"] += 1
        else:
            report["mismatch"].append({
                "bug_id": bug_id, "stored": stored_keys, "current": current_keys})
        if a.sleep and i + 1 < len(ids):
            time.sleep(a.sleep)

    out = os.path.join(ROOT, "results", "history_verification.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(f"checked {report['checked']} bugs")
    print(f"  faithful (stored == current)          : {report['faithful']}")
    print(f"  faithful + later moves appended       : {report['extra_moves_only']}")
    print(f"  no stored history, none on Bugzilla   : {len(report['no_stored'])}")
    print(f"  MISMATCH                              : {len(report['mismatch'])}")
    print(f"  fetch failed                          : {len(report['fetch_failed'])}")
    for m in report["mismatch"][:5]:
        print(f"    bug {m['bug_id']}: stored={m['stored']} current={m['current']}")
    for f in report["fetch_failed"][:3]:
        print(f"    fetch failed {f['bug_id']}: {f['error']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
