"""V3 paper campaign runner: parts A (DKV streaming, new defaults), B (tiled
KIVI-4), C (hybrid mode), in that order, each into its own log under
paper/results/campaign/. Holds off idle sleep for the whole run (sleep kills the
CUDA context). Every harness is resumable, so after a power cut just run this
again with the same arguments.

    python benchmarks/run_v3_campaign.py            # A, B, C1, C2
    python benchmarks/run_v3_campaign.py a b c1     # any subset, in order
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = {
    "a": (r"benchmarks\run_v3_a_dkv_stream.cmd", []),
    "b": (r"benchmarks\run_v3_b_kivi_tiled.cmd", []),
    "c1": (r"benchmarks\run_v3_c_hybrid.cmd", ["c1"]),
    "c2": (r"benchmarks\run_v3_c_hybrid.cmd", ["c2"]),
}


def main():
    parts = [p.lower() for p in sys.argv[1:]] or ["a", "b", "c1", "c2"]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise SystemExit("unknown part(s): %s (use a, b, c1, c2)" % bad)
    if os.name == "nt":                     # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    logdir = os.path.join(ROOT, "paper", "results", "campaign")
    os.makedirs(logdir, exist_ok=True)
    for p in parts:
        cmd, args = PARTS[p]
        log = os.path.join(logdir, "v3_%s.log" % p)
        t0 = time.time()
        print("[v3] part %s -> %s" % (p, log), flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n===== start %s =====\n" % time.ctime())
            f.flush()
            rc = subprocess.call(["cmd", "/c", os.path.join(ROOT, cmd)] + args,
                                 cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print("[v3] part %s done rc=%d in %.1f h" % (p, rc, (time.time() - t0) / 3600),
              flush=True)
    print("[v3] CAMPAIGN DONE", flush=True)


if __name__ == "__main__":
    main()
