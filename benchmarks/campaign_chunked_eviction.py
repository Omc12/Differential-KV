#!/usr/bin/env python3
"""The chunk-wise eviction campaign, end to end, resumable at every level.

WHAT IT RUNS
------------
Gates first -- the store tests and the per-model self-checks. If any fails,
nothing long starts. Then the context ladder, LongBench and RULER for
streamingllm_chunked and h2o_chunked, each configured exactly like the arm it
is compared with (see STEPS).

RESUMING
--------
After a power cut, a crash, or a Ctrl-C, run the same command again:

    python benchmarks/campaign_chunked_eviction.py

* A step that finished is skipped (state file, written atomically).
* A step that was in flight resumes inside itself: every harness here appends
  one fsync'd record per item or ladder point and skips what is on disk, so
  the cost of an interruption is the one item that was running.
* Items that ERRORED are not counted as done by the stores. After each
  harness step the campaign reads the store back and, if any item's latest
  record is an error, runs the step again (up to --retries times) so
  transient failures are retried rather than left as holes.

WATCHING
--------
    python benchmarks/campaign_chunked_eviction.py --status

prints each step's state, and the tail of the step in flight. Everything a
step prints goes to paper/results/campaign/logs/<step>.log (appended, so a
resumed step's log shows both attempts).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL, _boot_time, _pid_alive   # noqa: E402

PY = sys.executable
CAMP = os.path.join(REPO, "paper", "results", "campaign")
STATE = os.path.join(CAMP, "chunked_eviction.state.json")
LOCK = os.path.join(CAMP, "chunked_eviction.lock")
LOGS = os.path.join(CAMP, "logs")
RES = os.path.join(REPO, "paper", "results")

QWEN, GRANITE = "Qwen/Qwen3.5-4B", "ibm-granite/granite-4.2-8b"
SL_P = '{"n_sink": 4, "recency_window": 2044}'
H2_P = '{"budget": 2048, "recency_window": 512, "prefill_chunk": 256}'
SK_P = '{"budget": 2016, "window": 32}'          # as the committed RULER arm


def _b(script, *a):
    return [PY, os.path.join(HERE, script), *a]


# name, command, gate (stop the campaign if it fails), store to audit (or None)
STEPS = [
    ("selfcheck_store", _b("selfcheck_chunked_eviction.py", "--store"), True, None),
    ("selfcheck_qwen4b", _b("selfcheck_chunked_eviction.py", "--model", QWEN), True, None),
    ("selfcheck_granite", _b("selfcheck_chunked_eviction.py", "--model", GRANITE), True, None),

    # Ladders: the ceiling question. Same file layout, gen and chunk as the
    # existing ladders; a separate file so the committed ones are untouched.
    ("ladder_qwen4b", _b("context_ladder.py", "--model", QWEN,
                         "--arms", "streamingllm_chunked", "h2o_chunked",
                         "--contexts", "8192", "16384", "32768", "49152",
                         "65536", "98304", "131072",
                         "--out", os.path.join(RES, "ladder",
                                               "Qwen3.5-4B_mid_nf4_chunked.jsonl")),
     False, "ladder/Qwen3.5-4B_mid_nf4_chunked.jsonl"),
    ("ladder_granite", _b("context_ladder.py", "--model", GRANITE,
                          "--arms", "streamingllm_chunked", "h2o_chunked",
                          "--contexts", "4096", "8192", "16384", "24576", "32768",
                          "49152", "65536", "98304", "131072",
                          "--out", os.path.join(RES, "ladder",
                                                "granite-4.2-8b_mid_nf4_chunked.jsonl")),
     False, "ladder/granite-4.2-8b_mid_nf4_chunked.jsonl"),

    # SnapKV on Qwen3.5-4B, re-measured with the attention-alignment fix: the
    # old arm evicted 2 of 8 KV layers. Same rungs as the committed ladder.
    ("ladder_qwen4b_snapkv_aligned",
     _b("context_ladder.py", "--model", QWEN, "--arms", "snapkv",
        "--contexts", "32768", "49152", "65536",
        "--baseline-params", SK_P,
        "--out", os.path.join(RES, "ladder", "Qwen3.5-4B_mid_nf4_snapkv_aligned.jsonl")),
     False, "ladder/Qwen3.5-4B_mid_nf4_snapkv_aligned.jsonl"),

    # LongBench: granite @ 12k, 6 tasks x 20, as every other arm in Table 1.
    ("longbench_streamingllm_chunked",
     _b("run_longbench_cuda.py", "--model", GRANITE, "--arm", "streamingllm_chunked",
        "--max-length", "12000", "--num-samples", "20", "--baseline-params", SL_P,
        "--out", os.path.join(RES, "longbench",
                              "granite-4.2-8b_streamingllm_chunked_len12000.jsonl")),
     False, "longbench/granite-4.2-8b_streamingllm_chunked_len12000.jsonl"),
    ("longbench_h2o_chunked",
     _b("run_longbench_cuda.py", "--model", GRANITE, "--arm", "h2o_chunked",
        "--max-length", "12000", "--num-samples", "20", "--baseline-params", H2_P,
        "--out", os.path.join(RES, "longbench",
                              "granite-4.2-8b_h2o_chunked_len12000.jsonl")),
     False, "longbench/granite-4.2-8b_h2o_chunked_len12000.jsonl"),

    # RULER: Qwen3.5-4B. SnapKV again first (its committed numbers are from
    # the misaligned arm), then the chunked arms at 4k-32k, then 64k like DKV.
    ("ruler32k_snapkv_aligned",
     _b("run_ruler_cuda.py", "--model", QWEN, "--arm", "snapkv",
        "--max-length", "32768", "--baseline-params", SK_P,
        "--out", os.path.join(RES, "ruler", "Qwen3.5-4B_snapkv_aligned_max32768.jsonl")),
     False, "ruler/Qwen3.5-4B_snapkv_aligned_max32768.jsonl"),
    ("ruler32k_streamingllm_chunked",
     _b("run_ruler_cuda.py", "--model", QWEN, "--arm", "streamingllm_chunked",
        "--max-length", "32768", "--baseline-params", SL_P,
        "--out", os.path.join(RES, "ruler",
                              "Qwen3.5-4B_streamingllm_chunked_max32768.jsonl")),
     False, "ruler/Qwen3.5-4B_streamingllm_chunked_max32768.jsonl"),
    ("ruler32k_h2o_chunked",
     _b("run_ruler_cuda.py", "--model", QWEN, "--arm", "h2o_chunked",
        "--max-length", "32768", "--baseline-params", H2_P,
        "--out", os.path.join(RES, "ruler",
                              "Qwen3.5-4B_h2o_chunked_max32768.jsonl")),
     False, "ruler/Qwen3.5-4B_h2o_chunked_max32768.jsonl"),
    ("ruler64k_streamingllm_chunked",
     _b("run_ruler_cuda.py", "--model", QWEN, "--arm", "streamingllm_chunked",
        "--min-length", "65536", "--max-length", "65536", "--baseline-params", SL_P,
        "--out", os.path.join(RES, "ruler",
                              "Qwen3.5-4B_streamingllm_chunked_max65536.jsonl")),
     False, "ruler/Qwen3.5-4B_streamingllm_chunked_max65536.jsonl"),
    ("ruler64k_h2o_chunked",
     _b("run_ruler_cuda.py", "--model", QWEN, "--arm", "h2o_chunked",
        "--min-length", "65536", "--max-length", "65536", "--baseline-params", H2_P,
        "--out", os.path.join(RES, "ruler",
                              "Qwen3.5-4B_h2o_chunked_max65536.jsonl")),
     False, "ruler/Qwen3.5-4B_h2o_chunked_max65536.jsonl"),

    # What granite's peak is made of: allocator trace at the peak, bucketed by
    # the code site that allocated it, at two lengths, with Qwen as control.
    ("diag_granite_snapshot", _b("diag_granite_snapshot.py"),
     False, "diag/granite_peak_snapshot.jsonl"),

    # Users per GPU: sessions of 16k held resident at once until the card
    # spills, one process per arm.
    ("concurrency_qwen4b_16k", _b("bench_concurrency_cuda.py"),
     False, "concurrency/Qwen3.5-4B_ctx16384.jsonl"),
]


# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not os.path.exists(STATE):
        return {}
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


def save_state(st: dict) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE)


def store_audit(rel: str) -> dict:
    """Latest-record counts for a store: how many items, how many errored."""
    p = os.path.join(RES, rel)
    if not os.path.exists(p):
        return {"records": 0, "errors": 0}
    s = ResumableJSONL(p, read_only=True)
    latest = s.load_latest()
    # Ladder points carry a `status`, and an OOM/died/timeout there is a
    # MEASURED CEILING with an `error` message attached, not a failure; only
    # status "error" is a crash. Item stores have no status: any `error` is one.
    errs = [k for k, r in latest.items()
            if (r.get("status") == "error" if "status" in r else r.get("error"))]
    by_status = {}
    for r in latest.values():
        if "status" in r:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    out = {"records": len(latest), "errors": len(errs), "error_keys": errs[:10]}
    if by_status:
        out["status"] = by_status
    return out


def acquire_lock() -> None:
    if os.path.exists(LOCK):
        with open(LOCK, encoding="utf-8") as f:
            prev = json.load(f)
        same_boot = abs(prev.get("boot", 0) - _boot_time()) <= 120
        if same_boot and _pid_alive(int(prev.get("pid", 0))):
            raise SystemExit(f"campaign already running (pid {prev['pid']}, "
                             f"started {prev.get('started')})")
    with open(LOCK, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "boot": _boot_time(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(CAMP, "chunked_eviction.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, cmd: list) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with open(os.path.join(LOGS, f"{name}.log"), "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                 f"{' '.join(cmd)}\n")
        lf.flush()
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    return p.returncode


def status() -> None:
    st = load_state()
    for name, _cmd, gate, store in STEPS:
        s = st.get(name, {})
        extra = ""
        if store:
            a = store_audit(store)
            extra = f"  records={a['records']} errors={a['errors']}"
            if a.get("status"):
                extra += f" {a['status']}"
        print(f"{s.get('status', 'pending'):>18}  {name}{extra}")
    running = [n for n, *_ in STEPS if st.get(n, {}).get("status") == "running"]
    for n in running:
        p = os.path.join(LOGS, f"{n}.log")
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-6:]
            print(f"\n--- {n} (tail) ---\n" + "".join(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these step names (gates still run first)")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if args.status:
        return status()

    acquire_lock()
    if os.name == "nt":
        # Hold off idle sleep for as long as THIS process lives. A sleep in the
        # middle of a step stalls a GPU run and poisons its timings; this is
        # released automatically on exit and changes no power settings.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    try:
        st = load_state()
        log(f"campaign start (pid {os.getpid()})")
        for name, cmd, gate, store in STEPS:
            if args.only is not None and not gate and name not in args.only:
                continue
            if st.get(name, {}).get("status") in ("done", "done_with_errors"):
                log(f"skip {name} ({st[name]['status']})")
                continue
            for attempt in range(1 + (args.retries if store else 0)):
                st[name] = {"status": "running", "attempt": attempt + 1,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_state(st)
                log(f"run  {name} (attempt {attempt + 1})")
                t0 = time.time()
                rc = run_step(name, cmd)
                audit = store_audit(store) if store else None
                st[name].update({"rc": rc, "hours": round((time.time() - t0) / 3600, 2),
                                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
                if audit is not None:
                    st[name]["audit"] = audit
                if rc == 0 and (audit is None or audit["errors"] == 0):
                    st[name]["status"] = "done"
                    break
                st[name]["status"] = "failed"
                save_state(st)
                log(f"     {name}: rc={rc} audit={audit}")
                if gate:
                    break
            if st[name]["status"] == "failed" and store and rc == 0:
                # Ran to completion but some items still error after the
                # retries. Recorded, not hidden, and the campaign moves on.
                st[name]["status"] = "done_with_errors"
            save_state(st)
            log(f"end  {name}: {st[name]['status']} {st[name].get('audit', '')}")
            if st[name]["status"] == "failed" and gate:
                log(f"GATE FAILED at {name}; stopping. See logs/{name}.log")
                sys.exit(2)
        log("campaign complete")
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
