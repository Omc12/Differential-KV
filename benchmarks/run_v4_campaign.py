"""V4 paper campaign runner. Re-runs every RULER result the paper uses on the
fixed harness (RULER's answer prefix attached) and adds the comparisons review
asked for: eviction at DKV's own bytes, tiled KIVI-2, DKV attending every block,
exact KV offloaded to host memory, SnapKV with the allocator fix.

Parts, in priority order (each logs to paper/results/campaign/v4_<part>.log):
    z  smoke test of every new arm (minutes)
    a  Qwen3.5-4B 4k-32k              b  Qwen3.5-4B 131k
    c  granite-4.2-8b 4k-32k          f  reach and timing for the new arms
    g  LongBench, tiled KIVI-2        e  Qwen3.5-4B 64k

Holds off idle sleep (sleep kills the CUDA context). Every harness is
resumable, so after a power cut run this again with the same arguments; a
part that fails its smoke test stops the campaign.

    python benchmarks/run_v4_campaign.py            # z a b c f g e
    python benchmarks/run_v4_campaign.py b c        # any subset, in order
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = {
    "z": r"benchmarks\run_v4_z_smoke.cmd",
    "a": r"benchmarks\run_v4_a_qwen.cmd",
    "b": r"benchmarks\run_v4_b_qwen131k.cmd",
    "c": r"benchmarks\run_v4_c_granite.cmd",
    "f": r"benchmarks\run_v4_f_reach.cmd",
    "g": r"benchmarks\run_v4_g_longbench.cmd",
    "e": r"benchmarks\run_v4_e_qwen64k.cmd",
}
ORDER = ["z", "a", "b", "c", "f", "g", "e"]


def smoke_ok(log):
    """Every smoke store must exist and hold rows without an error field."""
    import glob
    import json
    stores = glob.glob(os.path.join(ROOT, "paper", "results", "smoke", "V4Z_*.jsonl"))
    if len(stores) < 6:
        log.write("[v4] smoke: only %d stores\n" % len(stores))
        return False
    for s in stores:
        rows = [json.loads(l) for l in open(s, encoding="utf-8") if l.strip()]
        if not rows or any(r.get("error") for r in rows):
            log.write("[v4] smoke FAILED: %s\n" % s)
            return False
    return True


def main():
    parts = [p.lower() for p in sys.argv[1:]] or ORDER
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise SystemExit("unknown part(s): %s (use %s)" % (bad, " ".join(ORDER)))
    if os.name == "nt":                     # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    logdir = os.path.join(ROOT, "paper", "results", "campaign")
    os.makedirs(logdir, exist_ok=True)
    for p in parts:
        log = os.path.join(logdir, "v4_%s.log" % p)
        t0 = time.time()
        print("[v4] part %s -> %s" % (p, log), flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n===== start %s =====\n" % time.ctime())
            f.flush()
            rc = subprocess.call(["cmd", "/c", os.path.join(ROOT, PARTS[p])],
                                 cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if p == "z" and not smoke_ok(f):
                print("[v4] SMOKE FAILED -- see %s" % log, flush=True)
                raise SystemExit(1)
        print("[v4] part %s done rc=%d in %.1f h" % (p, rc, (time.time() - t0) / 3600),
              flush=True)
    print("[v4] CAMPAIGN DONE", flush=True)


if __name__ == "__main__":
    main()
