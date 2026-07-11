"""Phase-tagged phys_footprint profiler for the native binary.

Drives one NIAH prompt through the stdin protocol, samples the child's
phys_footprint at 20 Hz, and timestamps stderr phase lines. Reports footprint
at each phase marker + the largest jumps with their nearest stderr context.

This is what decomposed the native lego stage-1 "peak didn't move" mystery
(see docs/NATIVE_LEGO_PORT_PLAN.md) — GPU/Metal buffer bytes don't show up in
phys_footprint, only host RAM does, which is why ringing persistent_k_cache/
persistent_v_cache (device tensors) was invisible while the host mirrors
(k_activations/v_activations, plain std::vector<ggml_fp16_t>) are the real
target for stage 2. Re-run this after any native memory change to confirm the
peak actually moves before declaring a win.

Usage:
    PROF_CTX=16384 python benchmarks/native_mem_profile.py
    PROF_CTX=16384 DIFFKV_LEGO_PREFILL=1 python benchmarks/native_mem_profile.py
"""
import os, sys, time, threading, subprocess, ctypes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "diffkv_native/build/diffkv_native")
MODEL = os.path.join(REPO, "diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf")
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from bench_common import build_niah_prompt  # noqa

_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
class _RU(ctypes.Structure):
    _fields_ = [("u", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "a", "b", "c", "d", "e", "f", "g", "phys", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q")]
def footprint(pid):
    buf = _RU()
    if _libc.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
        return 0
    return int(buf.phys)

ctx = int(os.environ.get("PROF_CTX", "16384"))
prompt_text = build_niah_prompt(ctx)
if isinstance(prompt_text, tuple):
    prompt_text = prompt_text[0]

env = dict(os.environ)
env.setdefault("DIFFKV_ENGAGE_THRESHOLD", "4096")
env["DIFFKV_MAX_TOKENS"] = "8"

proc = subprocess.Popen([BIN, MODEL, "-"], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        bufsize=0, env=env)
t0 = time.time()
events = []   # (t, tag)
samples = []  # (t, bytes)
stop = threading.Event()

def rd_stderr():
    for raw in proc.stderr:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if any(s in line for s in ("Prefill Progress", "LEGO_PREFILL", "DEBUG_CAP",
                                   "engage_threshold", "Prefill Phase", "auto engage",
                                   "[DiffKV", "Loading", "loaded", "Step 1 ")):
            events.append((time.time() - t0, line[:110]))

def rd_stdout():
    buf = b""
    for raw in iter(lambda: proc.stdout.read(4096), b""):
        buf += raw
        if b"__READY__" in buf and not getattr(rd_stdout, "sent", False):
            rd_stdout.sent = True
            events.append((time.time() - t0, ">>> READY, sending prompt"))
            single = prompt_text.replace("\\", "\\\\").replace("\n", "\\n")
            proc.stdin.write((single + "\n").encode()); proc.stdin.flush()
        if b"__FINISH__" in buf:
            events.append((time.time() - t0, ">>> FINISH"))
            stop.set()
            try:
                proc.stdin.write(b"exit\n"); proc.stdin.flush()
            except Exception:
                pass
            return

threading.Thread(target=rd_stderr, daemon=True).start()
threading.Thread(target=rd_stdout, daemon=True).start()

while not stop.is_set() and proc.poll() is None and time.time() - t0 < 300:
    samples.append((time.time() - t0, footprint(proc.pid)))
    time.sleep(0.05)
try:
    proc.kill()
except Exception:
    pass

peak_t, peak_v = max(samples, key=lambda s: s[1])
print(f"PEAK {peak_v/1e9:.3f} GB at t={peak_t:.1f}s (run len {samples[-1][0]:.1f}s)")

print("\n--- phase timeline (footprint at event time) ---")
for et, tag in events:
    fp = max((v for t, v in samples if t <= et + 0.06), default=0)
    print(f"t={et:6.1f}s {fp/1e9:6.3f} GB | {tag}")

print("\n--- largest footprint jumps (>80MB between samples) ---")
jumps = []
for i in range(1, len(samples)):
    dv = samples[i][1] - samples[i - 1][1]
    if dv > 80e6:
        jumps.append((samples[i][0], dv))
for jt, dv in sorted(jumps, key=lambda x: -x[1])[:12]:
    near = min(events, key=lambda e: abs(e[0] - jt), default=(0, "?"))
    print(f"t={jt:6.1f}s +{dv/1e6:6.0f} MB | nearest event: {near[1][:80]}")
