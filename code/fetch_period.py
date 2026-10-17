"""Fetch resolved (FIXED) defect bugs from Mozilla Bugzilla for an arbitrary creation window.

This is the PARAMETERIZED sibling of fetch.py. fetch.py stays untouched as the record of the
2022-2024 run; this script takes the window and output directory from the command line so a
fresh, contamination-free test period can be pulled without touching the original data.
Same REST API, same four steps, same JSONL schema, same 1 s politeness delay.

Steps (each resumable within itself, by skipping ids already on disk):
  1. bugs      -> <out>/bugs_meta.jsonl
  2. changed   -> <out>/changed_ids.json
  3. comments  -> <out>/first_comment.jsonl
  4. history   -> <out>/history.jsonl

Usage:
  python fetch_period.py --start 2026-01-01 --end 2027-01-01 --out data/2026
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://bugzilla.mozilla.org/rest"
PRODUCTS = ["Firefox", "Core", "Toolkit", "DevTools"]
SLEEP = 1.0
UA = "bugtriage-research-script/0.2 (academic study)"

ARGS = None  # set in main()


def get(path, params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            time.sleep(SLEEP)
            return out
        except Exception as e:  # noqa: BLE001
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt} after error {e!r}; sleep {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("failed: " + url)


def base_query(product):
    return {
        "product": product, "resolution": "FIXED", "bug_type": "defect",
        "f1": "creation_ts", "o1": "greaterthaneq", "v1": ARGS.start,
        "f2": "creation_ts", "o2": "lessthan", "v2": ARGS.end,
        "f3": "keywords", "o3": "nowords", "v3": "intermittent-failure",
    }


def read_jsonl_ids(path, key="id"):
    s = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    s.add(json.loads(line)[key])
    return s


def step_bugs():
    out = os.path.join(ARGS.out, "bugs_meta.jsonl")
    done = read_jsonl_ids(out)
    print(f"bugs: {len(done)} already on disk", flush=True)
    fields = "id,summary,product,component,creation_time,cf_last_resolved,keywords,severity,priority"
    written = 0
    with open(out, "a") as f:
        for p in PRODUCTS:
            off = 0
            while True:
                q = base_query(p) | {"include_fields": fields, "limit": 500, "offset": off,
                                     "order": "bug_id"}
                bugs = get("bug", q)["bugs"]
                for b in bugs:
                    if b["id"] not in done:
                        f.write(json.dumps(b) + "\n")
                        done.add(b["id"])
                        written += 1
                f.flush()
                print(f"  {p} offset {off}: {len(bugs)} fetched, {written} new written", flush=True)
                if len(bugs) < 500:
                    break
                off += 500
    print(f"bugs: wrote {written} new records", flush=True)


def step_changed():
    out = os.path.join(ARGS.out, "changed_ids.json")
    res = {"component": [], "product": []}
    if os.path.exists(out):
        res = json.load(open(out))
    for field in res:
        have = set(res[field])
        for p in PRODUCTS:
            off = 0
            while True:
                q = base_query(p) | {"f4": field, "o4": "changedafter", "v4": "1998-01-01",
                                     "include_fields": "id", "limit": 1000, "offset": off,
                                     "order": "bug_id"}
                bugs = get("bug", q)["bugs"]
                for b in bugs:
                    if b["id"] not in have:
                        res[field].append(b["id"])
                        have.add(b["id"])
                if len(bugs) < 1000:
                    break
                off += 1000
        print(f"  {field}: {len(res[field])} ids", flush=True)
        json.dump(res, open(out, "w"))
    print("changed: wrote", {k: len(v) for k, v in res.items()}, flush=True)


def step_comments():
    out = os.path.join(ARGS.out, "first_comment.jsonl")
    done = read_jsonl_ids(out)
    ids = [json.loads(l)["id"] for l in open(os.path.join(ARGS.out, "bugs_meta.jsonl"))]
    todo = [i for i in ids if i not in done]
    print(f"comments: {len(done)} done, {len(todo)} todo", flush=True)
    with open(out, "a") as f:
        for k in range(0, len(todo), 50):
            chunk = todo[k:k + 50]
            d = get(f"bug/{chunk[0]}/comment", {"ids": chunk[1:], "include_fields": "count,text"})
            for bid, v in d["bugs"].items():
                first = [c for c in v["comments"] if c.get("count") == 0]
                text = first[0]["text"] if first else ""
                f.write(json.dumps({"id": int(bid), "description": text[:6000],
                                    "n_comments": len(v["comments"])}) + "\n")
            f.flush()
            if (k // 50) % 20 == 0:
                print(f"  comments {k + len(chunk)}/{len(todo)}", flush=True)


def step_history():
    out = os.path.join(ARGS.out, "history.jsonl")
    done = read_jsonl_ids(out)
    ch = json.load(open(os.path.join(ARGS.out, "changed_ids.json")))
    todo = sorted((set(ch["component"]) | set(ch["product"])) - done)
    print(f"history: {len(done)} done, {len(todo)} todo", flush=True)
    with open(out, "a") as f:
        for k in range(0, len(todo), 50):
            chunk = todo[k:k + 50]
            d = get(f"bug/{chunk[0]}/history", {"ids": chunk[1:]})
            for b in d["bugs"]:
                moves = []
                for h in b["history"]:
                    for c in h["changes"]:
                        if c["field_name"] in ("component", "product"):
                            # store only whether automation made the change; no e-mail addresses
                            moves.append({"when": h["when"], "field": c["field_name"],
                                          "removed": c["removed"], "added": c["added"],
                                          "by_bot": "bot" in (h.get("who") or "").lower()})
                f.write(json.dumps({"id": b["id"], "moves": moves}) + "\n")
            f.flush()
            if (k // 50) % 10 == 0:
                print(f"  history {k + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("steps", nargs="*", default=None)
    ARGS = ap.parse_args()
    os.makedirs(ARGS.out, exist_ok=True)
    for s in (ARGS.steps or ["bugs", "changed", "comments", "history"]):
        globals()["step_" + s]()
