"""Fetch resolved (FIXED) defect bugs from Mozilla Bugzilla (BMO) REST API.

Streams everything to JSONL under data/. Low memory, polite (sleep between requests).

Steps:
  1. bugs:       metadata for products P, created in [START, END), resolution FIXED, type defect,
                 excluding intermittent-failure bot bugs  -> data/bugs_meta.jsonl
  2. changed:    ids of bugs whose component / product ever changed (server-side search)
                                                            -> data/changed_ids.json
  3. comments:   first comment (description, comment count==0), batched 50 ids/request
                                                            -> data/first_comment.jsonl
  4. history:    full field history only for changed bugs (to recover the initially filed
                 product/component and number of moves)     -> data/history.jsonl
Each step is resumable (skips ids already on disk).
"""
import json, os, sys, time, urllib.parse, urllib.request

BASE = "https://bugzilla.mozilla.org/rest"
PRODUCTS = ["Firefox", "Core", "Toolkit", "DevTools"]
START, END = "2022-01-01", "2025-01-01"
SLEEP = 1.0
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
UA = "bugtriage-research-script/0.1 (academic feasibility study)"


def get(path, params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            time.sleep(SLEEP)
            return out
        except Exception as e:  # noqa
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt} after error {e!r}; sleep {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("failed: " + url)


def base_query(product):
    return {
        "product": product, "resolution": "FIXED", "bug_type": "defect",
        "f1": "creation_ts", "o1": "greaterthaneq", "v1": START,
        "f2": "creation_ts", "o2": "lessthan", "v2": END,
        "f3": "keywords", "o3": "nowords", "v3": "intermittent-failure",
    }


def step_bugs():
    out = os.path.join(DATA, "bugs_meta.jsonl")
    if os.path.exists(out):
        print("bugs_meta exists, skip"); return
    fields = "id,summary,product,component,creation_time,cf_last_resolved,keywords,severity,priority"
    with open(out + ".tmp", "w") as f:
        for p in PRODUCTS:
            off = 0
            while True:
                q = base_query(p) | {"include_fields": fields, "limit": 500, "offset": off, "order": "bug_id"}
                bugs = get("bug", q)["bugs"]
                for b in bugs:
                    f.write(json.dumps(b) + "\n")
                print(f"{p} offset {off}: {len(bugs)}", flush=True)
                if len(bugs) < 500:
                    break
                off += 500
    os.rename(out + ".tmp", out)


def step_changed():
    out = os.path.join(DATA, "changed_ids.json")
    if os.path.exists(out):
        print("changed_ids exists, skip"); return
    res = {"component": [], "product": []}
    for field in res:
        for p in PRODUCTS:
            off = 0
            while True:
                q = base_query(p) | {"f4": field, "o4": "changedafter", "v4": "1998-01-01",
                                     "include_fields": "id", "limit": 1000, "offset": off, "order": "bug_id"}
                bugs = get("bug", q)["bugs"]
                res[field] += [b["id"] for b in bugs]
                if len(bugs) < 1000:
                    break
                off += 1000
        print(field, len(res[field]), flush=True)
    json.dump(res, open(out, "w"))


def load_ids(path, key):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            s.add(json.loads(line)[key])
    return s


def step_comments():
    out = os.path.join(DATA, "first_comment.jsonl")
    done = load_ids(out, "id")
    ids = [json.loads(l)["id"] for l in open(os.path.join(DATA, "bugs_meta.jsonl"))]
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
    out = os.path.join(DATA, "history.jsonl")
    done = load_ids(out, "id")
    ch = json.load(open(os.path.join(DATA, "changed_ids.json")))
    todo = sorted(set(ch["component"]) | set(ch["product"]))
    todo = [i for i in todo if i not in done]
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
                            # store only whether the change was made by an automation account
                            # (no personal e-mail addresses are written to disk)
                            moves.append({"when": h["when"], "field": c["field_name"],
                                          "removed": c["removed"], "added": c["added"],
                                          "by_bot": "bot" in (h.get("who") or "").lower()})
                f.write(json.dumps({"id": b["id"], "moves": moves}) + "\n")
            f.flush()
            if (k // 50) % 10 == 0:
                print(f"  history {k + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    steps = sys.argv[1:] or ["bugs", "changed", "comments", "history"]
    for s in steps:
        globals()["step_" + s]()
