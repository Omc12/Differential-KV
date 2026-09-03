#!/usr/bin/env python3
"""Emit one line per data-health problem. Silence means healthy.

Watches for the four ways this campaign has actually gone wrong, rather than
for problems in the abstract:

  1. AUDIT FAILURE   -- validate_results.py reports a FAIL.
  2. CONCURRENT WRITERS -- two stores locked by two LIVE pids. Cost a set of
     peak-memory and latency numbers once already: both processes were resident
     on one 12 GB card at 96% and neither row records it.
  3. STALL           -- the active store has not grown in --stall-min minutes
     while a worker is up.
  4. LATENCY CLIFF   -- the marginal seconds-per-item, backed out of the
     harness's CUMULATIVE average, exceeds --cliff x the run's median. This is
     the one that matters: a cumulative average CANNOT show a cliff. Dense at
     64k displayed "14.7s/item, ~62 min left" while each item was costing
     532-850s and the true remainder was 37 hours.

Liveness is checked with tasklist. wmic is ABSENT on Windows 11, and a detector
built on it silently reported every live worker as gone -- which is how a
second worker got put on the card beside a running one.
"""
from __future__ import annotations
import argparse, glob, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def live_pids() -> set[int]:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                              "/FO", "CSV"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return set()
    pids = set()
    for line in out.splitlines()[1:]:
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def held_locks(alive: set[int]):
    out = []
    for p in glob.glob(os.path.join(REPO, "paper", "results", "*", "*.lock")):
        try:
            pid = json.load(open(p, encoding="utf-8")).get("pid")
        except Exception:
            pid = None
        if pid in alive:
            out.append((os.path.basename(p)[:-5], pid))
    return out


def marginal_costs(log: str, tail: int = 4000):
    """(n, seconds) for recent items, from the cumulative average."""
    try:
        lines = open(log, encoding="utf-8", errors="ignore").read().splitlines()[-tail:]
    except Exception:
        return []
    pts = []
    for ln in lines:
        m = re.search(r"\[(\d+)/\d+\].*?\(([\d.]+)s/item", ln)
        if m:
            pts.append((int(m.group(1)), float(m.group(2))))
    out = []
    for i in range(1, len(pts)):
        (n0, a0), (n1, a1) = pts[i - 1], pts[i]
        if n1 > n0:
            out.append((n1, (a1 * n1 - a0 * n0) / (n1 - n0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="paper/results/all.log")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--stall-min", type=int, default=45)
    ap.add_argument("--cliff", type=float, default=6.0)
    a = ap.parse_args()

    sizes: dict[str, tuple[int, float]] = {}
    while True:
        alive = live_pids()

        py = os.path.join(REPO, "benchmarks", "validate_results.py")
        try:
            r = subprocess.run([sys.executable, py], capture_output=True,
                               text=True, cwd=REPO, timeout=600)
            # Only real failure ROWS. The validator also prints a TALLY
            # ("24 ok, 3 warn, 0 FAIL") and an explanatory sentence, both of
            # which contain the word -- matching those made the watchdog fire
            # every interval on a clean run.
            bad = [l for l in (r.stdout or "").splitlines()
                   if l.lstrip().startswith("[ FAIL ]")]
            if bad:
                print(f"AUDIT: {len(bad)} FAIL line(s) -- "
                      f"{bad[0].strip()[:110]}", flush=True)
        except Exception as e:
            print(f"AUDIT: validator did not run: {type(e).__name__}", flush=True)

        locks = held_locks(alive)
        if len(locks) > 1:
            print("CONTENTION: " + ", ".join(f"{n}(pid {p})" for n, p in locks)
                  + " -- two live writers share one card; peak-memory and "
                    "latency rows written now are contaminated", flush=True)

        now = time.time()
        for f in glob.glob(os.path.join(REPO, "paper", "results", "*", "*.jsonl")):
            try:
                sz = os.path.getsize(f)
            except OSError:
                continue
            key = os.path.basename(f)
            prev = sizes.get(key)
            if prev and sz == prev[0] and locks and \
                    (now - prev[1]) > a.stall_min * 60:
                print(f"STALL: {key} unchanged for "
                      f"{int((now - prev[1]) / 60)} min while a writer holds a "
                      f"lock", flush=True)
                sizes[key] = (sz, now)
            elif not prev or sz != prev[0]:
                sizes[key] = (sz, now)

        mc = marginal_costs(os.path.join(REPO, a.log))
        if len(mc) >= 8:
            med = statistics.median(s for _, s in mc)
            recent = mc[-3:]
            if med > 0 and all(s > a.cliff * med for _, s in recent):
                print(f"CLIFF: last 3 items cost "
                      f"{', '.join(f'{s:.0f}s' for _, s in recent)} vs median "
                      f"{med:.0f}s -- spill signature; the displayed average "
                      f"will NOT show this", flush=True)

        time.sleep(a.interval)


if __name__ == "__main__":
    main()
