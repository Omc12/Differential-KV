#!/usr/bin/env python3
"""
profile_run.py — Resource profiler for diffkv_native
Runs the model for 20 seconds, captures CPU/memory/threads every 250ms,
takes a macOS `sample` CPU call-stack trace, then prints a full diagnosis.
"""

import subprocess, sys, os, time, signal, json, threading, collections
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
BINARY     = str(ROOT / "build" / "diffkv_native")
MODEL      = str(ROOT / "qwen2.5-1.5b-instruct-q8_0.gguf")
PYTHON     = str(ROOT.parent / "diffkv_venv" / "bin" / "python3")
CLI        = str(ROOT / "serving" / "cli.py")
INPUT_FILE = str(ROOT.parent / "scratch" / "pride_and_prejudice.txt")
SAMPLE_SECS = 20          # how long to monitor
POLL_HZ     = 4           # measurements per second
SAMPLE_OUT  = "/tmp/diffkv_sample.txt"

# ── Try to import psutil (best effort) ──────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not found — install with: pip install psutil")
    print("       Falling back to ps(1) for metrics\n")

# ── Helpers ──────────────────────────────────────────────────────────────────
def ps_snapshot(pid):
    """Return (cpu_pct, rss_mb, n_threads) via ps(1) — no psutil needed."""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "pcpu=,rss=,nlwp="],
            stderr=subprocess.DEVNULL, text=True
        ).strip().split()
        return float(out[0]), int(out[1]) / 1024, int(out[2])
    except Exception:
        return 0.0, 0.0, 0

def vm_stat_pressure():
    """Return system free memory (MB) from vm_stat."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page = 16384  # default macOS page size bytes
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split()[-2])
                break
        free_pages = 0
        for line in out.splitlines():
            if "Pages free:" in line:
                free_pages = int(line.split()[-1].strip("."))
                break
        return free_pages * page / (1024 * 1024)
    except Exception:
        return -1

def read_input():
    """Read first 20KB of Pride and Prejudice."""
    try:
        with open(INPUT_FILE, "rb") as f:
            return f.read(20000).decode("utf-8", errors="replace")
    except FileNotFoundError:
        # Create a minimal test input if scratch file missing
        return "Summarize the following text:\n" + ("The quick brown fox. " * 500)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    text = read_input()

    env = os.environ.copy()
    env["DIFFKV_NATIVE_ATTN"] = "1"
    env["DIFFKV_MAX_TOKENS"]  = "60"

    print(f"{'='*60}")
    print(f"  diffkv_native Resource Profiler  ({SAMPLE_SECS}s window)")
    print(f"{'='*60}")
    print(f"  Binary : {BINARY}")
    print(f"  Input  : {len(text)} bytes")
    print(f"  Poll   : {POLL_HZ} Hz\n")

    # ── Launch diffkv_native ─────────────────────────────────────────────────
    proc = subprocess.Popen(
        [PYTHON, CLI,
         "--model", MODEL,
         "--binary-path", BINARY,
         "--preset", "low",
         "--max-tokens", "60"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    proc.stdin.write(text.encode())
    proc.stdin.close()

    t_start = time.time()
    pid = proc.pid
    print(f"[Profile] CLI wrapper launched PID={pid}")

    # ── Find the actual C++ diffkv_native child process ───────────────────────
    # The Python CLI just spawns diffkv_native as a subprocess and proxies I/O.
    # The real CPU/RAM is in the C++ child process, not the Python wrapper.
    target_pid = pid
    if HAS_PSUTIL:
        deadline_find = time.time() + 10
        print("[Profile] Waiting for C++ diffkv_native child process...")
        while time.time() < deadline_find:
            try:
                children = psutil.Process(pid).children(recursive=True)
                for ch in children:
                    if "diffkv_native" in ch.name() or "diffkv_native" in " ".join(ch.cmdline()):
                        target_pid = ch.pid
                        print(f"[Profile] Found C++ child PID={target_pid} name={ch.name()}")
                        break
            except Exception:
                pass
            if target_pid != pid:
                break
            time.sleep(0.3)
        if target_pid == pid:
            print("[Profile] WARNING: C++ child not found, profiling Python wrapper")
    print(f"[Profile] Profiling PID={target_pid} at t=0\n")

    # ── Metrics arrays ───────────────────────────────────────────────────────
    ts_list   = []
    cpu_list  = []
    rss_list  = []
    thr_list  = []
    sys_free  = []

    stop_flag = threading.Event()
    output_lines = []
    stderr_lines = []

    def drain(stream, dest):
        for line in stream:
            dest.append(line.decode("utf-8", errors="replace").rstrip())

    t_out = threading.Thread(target=drain, args=(proc.stdout, output_lines), daemon=True)
    t_err = threading.Thread(target=drain, args=(proc.stderr, stderr_lines), daemon=True)
    t_out.start(); t_err.start()

    # ── macOS `sample` for CPU call-stack profiling ──────────────────────────
    sample_proc = None
    try:
        sample_proc = subprocess.Popen(
            ["sample", str(target_pid), str(SAMPLE_SECS), "-wait", "-f", SAMPLE_OUT],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[Profile] macOS `sample` on PID={target_pid} → {SAMPLE_OUT}")
    except FileNotFoundError:
        print("[Profile] macOS `sample` not found — skipping call-stack capture")

    # ── Poll loop ────────────────────────────────────────────────────────────
    interval = 1.0 / POLL_HZ
    deadline = t_start + SAMPLE_SECS

    if HAS_PSUTIL:
        try:
            ps_obj = psutil.Process(target_pid)
            ps_obj.cpu_percent(interval=None)  # prime it
        except Exception:
            ps_obj = None
    else:
        ps_obj = None

    while time.time() < deadline:
        elapsed = time.time() - t_start
        if proc.poll() is not None and (not HAS_PSUTIL or not psutil.pid_exists(target_pid)):
            print(f"\n[Profile] Process exited at t={elapsed:.1f}s")
            break

        if HAS_PSUTIL and ps_obj:
            try:
                cpu  = ps_obj.cpu_percent(interval=None)
                rss  = ps_obj.memory_info().rss / (1024 * 1024)
                nthr = ps_obj.num_threads()
            except Exception:
                cpu, rss, nthr = 0, 0, 0
        else:
            cpu, rss, nthr = ps_snapshot(target_pid)

        free = vm_stat_pressure()

        ts_list.append(elapsed)
        cpu_list.append(cpu)
        rss_list.append(rss)
        thr_list.append(nthr)
        sys_free.append(free)

        bar = "█" * int(cpu / 5) + "░" * (20 - int(min(cpu, 100) / 5))
        print(f"  t={elapsed:5.1f}s  CPU={cpu:5.1f}%  [{bar}]  "
              f"RSS={rss:6.0f}MB  Threads={nthr}  SysFree={free:.0f}MB")

        time.sleep(interval)

    # ── Kill process ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    t_out.join(timeout=2); t_err.join(timeout=2)

    # ── Wait for sample ───────────────────────────────────────────────────────
    if sample_proc and sample_proc.poll() is None:
        print("\n[Profile] Waiting for `sample` to finish...")
        try:
            sample_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sample_proc.terminate()

    # ── Analysis ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ANALYSIS")
    print(f"{'='*60}\n")

    if cpu_list:
        avg_cpu  = sum(cpu_list) / len(cpu_list)
        max_cpu  = max(cpu_list)
        min_cpu  = min(cpu_list)
        avg_rss  = sum(rss_list) / len(rss_list)
        max_rss  = max(rss_list)
        min_rss  = min(rss_list)
        rss_growth = max_rss - min_rss
        avg_thr  = sum(thr_list) / len(thr_list)

        print(f"  CPU:     avg={avg_cpu:.1f}%  max={max_cpu:.1f}%  min={min_cpu:.1f}%")
        print(f"  RAM:     avg={avg_rss:.0f}MB  max={max_rss:.0f}MB  growth={rss_growth:.0f}MB")
        print(f"  Threads: avg={avg_thr:.1f}")
        print()

        # ── Spike detection ──────────────────────────────────────────────────
        print("  ── CPU Spike Timeline ──")
        for i, (t, c) in enumerate(zip(ts_list, cpu_list)):
            if c > avg_cpu * 1.5 or c > 80:
                rss_delta = rss_list[i] - rss_list[i-1] if i > 0 else 0
                print(f"    t={t:.1f}s  CPU={c:.1f}%  RSS={rss_list[i]:.0f}MB  (ΔRSS={rss_delta:+.0f}MB)")
        print()

        # ── Memory growth detection ──────────────────────────────────────────
        if rss_growth > 50:
            print(f"  ⚠  RSS grew {rss_growth:.0f}MB during the run — likely repeated heap allocation")
            # Find when growth happened
            for i in range(1, len(rss_list)):
                delta = rss_list[i] - rss_list[i-1]
                if delta > 20:
                    print(f"     t={ts_list[i]:.1f}s: +{delta:.0f}MB spike (CPU was {cpu_list[i]:.0f}%)")
        print()

        # ── Pattern diagnosis ─────────────────────────────────────────────────
        print("  ── Pattern Diagnosis ──")
        if avg_cpu < 15 and avg_rss > 500:
            print("  → LOW CPU + HIGH RAM = Memory pressure / page fault stalls")
            print("    Likely cause: large vector reallocations or Metal buffer fragmentation")
        elif avg_cpu < 30 and max_cpu > 100:
            print("  → BURSTY CPU (low avg, high peaks) = periodic heavy computation")
            print("    Likely cause: chunk_graph rebuild or IDF recomputation every N tokens")
        elif avg_cpu > 80:
            print("  → HIGH SUSTAINED CPU = computation-bound bottleneck")
        if rss_growth > 100:
            print("  → GROWING RSS = active memory allocation during decode")
            print("    Likely cause: std::vector push_back in decode loop (recent_decode_keys,")
            print("    generated_tokens, inverted_index occurrences, etc.)")

    # ── Stderr from process ───────────────────────────────────────────────────
    print(f"\n  ── Key stderr lines (first 40) ──")
    for line in stderr_lines[:40]:
        print(f"    {line}")

    print(f"\n  ── Output lines ──")
    for line in output_lines[:20]:
        print(f"    {line}")

    # ── Sample output ─────────────────────────────────────────────────────────
    if Path(SAMPLE_OUT).exists():
        print(f"\n  ── Top CPU consumers from `sample` ──")
        sample_text = Path(SAMPLE_OUT).read_text()
        # Extract the top-level call tree (first 60 lines after "Call graph:")
        lines = sample_text.splitlines()
        in_graph = False
        count = 0
        for line in lines:
            if "Call graph:" in line:
                in_graph = True
            if in_graph:
                print(f"    {line}")
                count += 1
                if count > 80:
                    break
        if not in_graph:
            # Just print first 80 lines of sample
            for line in lines[:80]:
                print(f"    {line}")

    print(f"\n{'='*60}")
    print(f"  Total profiling time: {elapsed_total:.1f}s")
    print(f"  Full sample output:   {SAMPLE_OUT}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
