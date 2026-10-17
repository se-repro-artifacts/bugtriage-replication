"""Audit every paid call this project has made, from results/usage_<model>.jsonl.

The paper reports cost, so the cost has to be reconstructible from what was recorded rather than
from memory. Two things make that less trivial than summing a column:

* The vendor charges **double during its peak windows** (01:00-04:00 and 06:00-10:00 UTC on
  weekdays), so the multiplier is applied per call. `llm_client.spend_usd()` deliberately returns
  the off-peak figure, because it is used as a guard *before* a run starts; this script returns what
  was actually billed.
* The `ts_utc` column was added after the first day of calls. The earliest records carry local time
  only, so those are shifted by this machine's UTC offset. That fallback is stated here rather than
  hidden, because it is the one figure in the report that is not read directly off a timestamp.

Usage: python spend_report.py
"""
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import PRICES, PEAK            # noqa: E402
from offpeak import is_peak                    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOCAL_UTC_OFFSET_HOURS = 8                     # this machine runs UTC+8


def call_time(record):
    """(datetime in UTC, inferred?) for one usage record."""
    stamp = record.get("ts_utc")
    if stamp:
        return datetime.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S"), False
    local = (record.get("ts") or "")[:19]
    if not local:
        return None, True
    naive = datetime.datetime.strptime(local, "%Y-%m-%dT%H:%M:%S")
    return naive - datetime.timedelta(hours=LOCAL_UTC_OFFSET_HOURS), True


def main():
    rows, total = {}, {"calls": 0, "hit": 0, "miss": 0, "out": 0,
                       "offpeak_usd": 0.0, "billed_usd": 0.0, "inferred": 0, "peak_calls": 0}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "usage_*.jsonl"))):
        model = os.path.basename(path)[len("usage_"):-len(".jsonl")]
        prices = PRICES.get(model)
        if not prices:
            print(f"WARNING: no recorded prices for {model}; its calls are skipped")
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            usage = record.get("usage") or {}
            hit = usage.get("prompt_cache_hit_tokens") or 0
            miss = usage.get("prompt_cache_miss_tokens") or 0
            out = usage.get("completion_tokens") or 0
            when, inferred = call_time(record)
            peak = bool(when and is_peak(when))
            offpeak_usd = (hit * prices["hit"] + miss * prices["miss"]
                           + out * prices["out"]) / 1e6
            day = when.strftime("%Y-%m-%d") if when else "unknown"
            row = rows.setdefault(day, {"calls": 0, "hit": 0, "miss": 0, "out": 0,
                                        "offpeak_usd": 0.0, "billed_usd": 0.0,
                                        "inferred": 0, "peak_calls": 0})
            for target in (row, total):
                target["calls"] += 1
                target["hit"] += hit
                target["miss"] += miss
                target["out"] += out
                target["offpeak_usd"] += offpeak_usd
                target["billed_usd"] += offpeak_usd * (PEAK if peak else 1.0)
                target["inferred"] += 1 if inferred else 0
                target["peak_calls"] += 1 if peak else 0

    print(f"{'UTC day':12} {'calls':>7} {'peak':>6} {'cache hit':>14} {'cache miss':>12} "
          f"{'output':>11} {'off-peak $':>11} {'billed $':>10}")
    for day in sorted(rows):
        r = rows[day]
        print(f"{day:12} {r['calls']:>7} {r['peak_calls']:>6} {r['hit']:>14,} {r['miss']:>12,} "
              f"{r['out']:>11,} {r['offpeak_usd']:>11.4f} {r['billed_usd']:>10.4f}")
    print("-" * 96)
    print(f"{'TOTAL':12} {total['calls']:>7} {total['peak_calls']:>6} {total['hit']:>14,} "
          f"{total['miss']:>12,} {total['out']:>11,} {total['offpeak_usd']:>11.4f} "
          f"{total['billed_usd']:>10.4f}")
    tokens = total["hit"] + total["miss"] + total["out"]
    print()
    print(f"tokens processed      : {tokens:,}")
    print(f"cache hit rate        : {100 * total['hit'] / max(1, total['hit'] + total['miss']):.1f}%")
    print(f"calls at peak price   : {total['peak_calls']} of {total['calls']}")
    print(f"calls with an inferred timestamp: {total['inferred']} "
          f"(they predate the ts_utc column and were shifted by UTC+{LOCAL_UTC_OFFSET_HOURS})")
    print(f"billed (as charged)   : ${total['billed_usd']:.4f}")
    print(f"same calls off-peak   : ${total['offpeak_usd']:.4f}")

    out_path = os.path.join(ROOT, "results", "spend_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"by_day": rows, "total": total, "peak_multiplier": PEAK,
                   "prices_offpeak": PRICES, "local_utc_offset_hours": LOCAL_UTC_OFFSET_HOURS},
                  fh, indent=1, default=str)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
