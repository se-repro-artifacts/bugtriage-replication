"""DeepSeek charges double during peak hours, so a paid run should wait for the cheap window.

Copied from https://api-docs.deepseek.com/quick_start/pricing on 2026-10-12: peak is
01:00-04:00 and 06:00-10:00 UTC, Monday to Friday, excluding Chinese public holidays. Every other
hour, and the whole weekend, is off-peak at half price.

Chinese public holidays cannot be derived here, so they are treated as peak. That is the safe
direction: the guard may wait longer than strictly needed, but it never starts a run at double
price by mistake. Update HOLIDAYS by hand if a long run lands on one.

Usage:
  python offpeak.py             report the current state and the next cheap window
  python offpeak.py --wait      sleep until the next cheap window opens
"""
import argparse
import datetime as dt
import sys
import time

# UTC, half-open [start, end), Monday to Friday
PEAK_HOURS = ((1, 4), (6, 10))

# Chinese public holidays, if a run has to be scheduled across one. Left empty on purpose rather
# than guessed; an empty set only ever makes the guard more conservative.
HOLIDAYS = frozenset()


def is_peak(now):
    """True when DeepSeek is charging the peak rate at `now` (a timezone-aware UTC datetime)."""
    if now.weekday() >= 5:                      # Saturday or Sunday: the whole day is cheap
        return False
    if now.date().isoformat() in HOLIDAYS:
        return False
    return any(lo <= now.hour < hi for lo, hi in PEAK_HOURS)


def seconds_until_offpeak(now):
    """Seconds to wait from `now` before the cheap window opens; 0 when it is already open."""
    if not is_peak(now):
        return 0
    for lo, hi in PEAK_HOURS:
        if lo <= now.hour < hi:
            end = now.replace(hour=hi, minute=0, second=0, microsecond=0)
            return max(0.0, (end - now).total_seconds())
    return 0.0


def describe(now):
    wait = seconds_until_offpeak(now)
    if wait <= 0:
        return f"{now:%Y-%m-%d %H:%M} UTC: OFF-PEAK (half price)"
    return (f"{now:%Y-%m-%d %H:%M} UTC: PEAK (double price); cheap window opens in "
            f"{wait / 60:.0f} min, at {now + dt.timedelta(seconds=wait):%H:%M} UTC")


def wait_for_offpeak(now=None, sleep=time.sleep, verbose=True):
    """Block until the cheap window is open. Returns the seconds actually slept."""
    now = now or dt.datetime.now(dt.timezone.utc)
    wait = seconds_until_offpeak(now)
    if wait <= 0:
        if verbose:
            print(f"[offpeak] {describe(now)} -- starting now", flush=True)
        return 0.0
    if verbose:
        print(f"[offpeak] {describe(now)}", flush=True)
        print(f"[offpeak] sleeping {wait / 60:.1f} min so this run pays half price", flush=True)
    slept, target = 0.0, now.timestamp() + wait
    while True:
        remaining = target - dt.datetime.now(dt.timezone.utc).timestamp()
        if remaining <= 0:
            break
        step = min(remaining, 60.0)             # wake every minute so the log shows progress
        sleep(step)
        slept += step
        if verbose:
            print(f"[offpeak] {remaining / 60:.0f} min left", flush=True)
    return slept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    print(describe(now))
    if a.wait:
        wait_for_offpeak(now)
        print(describe(dt.datetime.now(dt.timezone.utc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
