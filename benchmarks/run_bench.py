#!/usr/bin/env python3
"""
Orchestrator for the DKV multi-engine context-length benchmark.

Sweeps {native, active, dense, ollama} x {4k..128k} on Qwen2.5-1.5B-Instruct (Q4).
Each (engine, context) measurement runs in an isolated worker subprocess
(bench_worker.py). The orchestrator:

  * builds one NIAH prompt per context length (same text for every engine),
  * samples the worker's process-tree RSS (peak) at 20 Hz — plus the ollama
    server's RSS for ollama runs,
  * enforces a per-test wall timeout and a RAM safety cap (kills the worker if
    the box is about to thrash — this is an 8 GB Mac),
  * on ANY failure (OOM / timeout / crash / error) marks that engine "dead" and
    SKIPS all larger context lengths for it (per the requirement: if it OOMs,
    don't carry it into the next segment),
  * writes results incrementally to JSON and regenerates a Markdown summary after
    every test, so a mid-sweep crash never loses collected data.

Nothing is fabricated: every number is measured. Failed cells are recorded with
their failure reason, not invented values.
"""

import os
import sys
import json
import time
import signal
import argparse
import threading
import subprocess
from datetime import datetime

import ctypes
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VENV_PY = os.path.join(REPO, "dkv_venv/bin/python3")
WORKER = os.path.join(HERE, "bench_worker.py")
RESULTS_DIR = os.path.join(HERE, "results")

DEFAULT_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_ENGINES = ["native", "active", "dense", "ollama"]
ENGINE_LABEL = {
    "native": "DKV native (C++, GGUF Q4_K_M)",
    "active": "DKV active runtime (MLX int4)",
    "dense": "Dense (mlx_lm int4, full KV)",
    "ollama": "Ollama / llama.cpp (GGUF Q4_K_M)",
}


# ───────────────────────────── memory sampling ──────────────────────────────
# On Apple Silicon (unified memory), psutil RSS does NOT fully count Metal/GPU
# buffer allocations — where the model weights and KV cache live. The accurate
# figure (what Activity Monitor shows as "Memory") is the kernel's per-process
# `phys_footprint`, read cheaply via libproc proc_pid_rusage(RUSAGE_INFO_V2).
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)


class _RUsageV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64) for n in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
            "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime",
            "ri_diskio_bytesread", "ri_diskio_byteswritten")]


def phys_footprint_bytes(pid):
    buf = _RUsageV2()
    if _libc.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
        return 0
    return int(buf.ri_phys_footprint)


def _proc_mem(pr):
    """(combined, phys, rss) bytes for one process. `combined` = max(phys, rss):
    MLX engines hide KV/weights in Metal buffers (in phys, not rss); llama.cpp /
    ggml mmap their weights (in rss, not phys). Taking the larger avoids
    undercounting either family."""
    rss = pr.memory_info().rss
    phys = phys_footprint_bytes(pr.pid)
    return max(phys, rss), phys, rss


def tree_mem_gb(pid):
    """(combined, phys, rss) in GB summed over a process tree."""
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0.0, 0.0, 0.0
    comb = phys = rss = 0
    for pr in procs:
        try:
            c, p_, r = _proc_mem(pr)
            comb += c
            phys += p_
            rss += r
        except psutil.Error:
            pass
    return comb / 1e9, phys / 1e9, rss / 1e9


def ollama_mem_gb():
    # The model runs in a `llama-server` child of `ollama serve` (named
    # "llama-server", NOT "ollama"); the app bundle path contains "Ollama.app".
    # Match those, but NOT our own benchmark worker (whose argv mentions ollama).
    comb = phys = rss = 0
    for pr in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (pr.info["name"] or "").lower()
            cmd = " ".join(pr.info["cmdline"] or []).lower()
            if name in ("ollama", "llama-server", "ollama-runner") or "ollama.app" in cmd:
                c, p_, r = _proc_mem(pr)
                comb += c
                phys += p_
                rss += r
        except psutil.Error:
            pass
    return comb / 1e9, phys / 1e9, rss / 1e9


class Sampler(threading.Thread):
    """Polls process-tree physical footprint; records peak; kills the worker if
    the true memory footprint blows the cap (protects this 8 GB box)."""

    def __init__(self, proc, include_ollama, cap_gb, interval=0.05):
        super().__init__(daemon=True)
        self.proc = proc
        self.include_ollama = include_ollama
        self.cap_gb = cap_gb
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_mem_gb = 0.0   # combined max(phys,rss) per proc (headline)
        self.peak_phys_gb = 0.0  # physical footprint only
        self.peak_rss_gb = 0.0   # resident set size only
        self.peak_ollama_gb = 0.0
        self.killed_oom = False
        self._breaches = 0

    def run(self):
        while not self.stop_event.is_set():
            comb, phys, rss = tree_mem_gb(self.proc.pid)
            if self.include_ollama:
                o_comb, o_phys, o_rss = ollama_mem_gb()
                comb += o_comb
                phys += o_phys
                rss += o_rss
                if o_comb > self.peak_ollama_gb:
                    self.peak_ollama_gb = o_comb
            if comb > self.peak_mem_gb:
                self.peak_mem_gb = comb
            if phys > self.peak_phys_gb:
                self.peak_phys_gb = phys
            if rss > self.peak_rss_gb:
                self.peak_rss_gb = rss
            if self.cap_gb and comb > self.cap_gb:
                self._breaches += 1
                if self._breaches >= 3:  # sustained, not a transient spike
                    self.killed_oom = True
                    _kill_proc(self.proc)
                    break
            else:
                self._breaches = 0
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


def _kill_proc(proc):
    try:
        for ch in psutil.Process(proc.pid).children(recursive=True):
            try:
                ch.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    try:
        proc.kill()
    except Exception:
        pass


# ───────────────────────────── ollama lifecycle ─────────────────────────────
def ollama_unload(model, url="http://localhost:11434"):
    import urllib.request
    try:
        data = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass
    time.sleep(1.5)


# ───────────────────────────── single test run ──────────────────────────────
def run_one(engine, ctx, gen, prompt_file, args):
    result_file = os.path.join(RESULTS_DIR, f".result_{engine}_{ctx}.json")
    pid_file = os.path.join(RESULTS_DIR, f".pid_{engine}_{ctx}.txt")
    for f in (result_file, pid_file):
        if os.path.exists(f):
            os.remove(f)

    if engine == "ollama":
        ollama_unload(args.ollama_model)

    cmd = [
        VENV_PY, WORKER,
        "--engine", engine, "--ctx", str(ctx), "--gen", str(gen),
        "--prompt-file", prompt_file, "--result-file", result_file,
        "--pid-file", pid_file,
        "--native-binary", args.native_binary,
        "--native-model", args.native_model,
        "--dense-model-id", args.dense_model_id,
        "--ollama-model", args.ollama_model,
        "--http-timeout", str(args.timeout + 60),
    ]
    log_file = os.path.join(RESULTS_DIR, f"log_{engine}_{ctx}.txt")
    logf = open(log_file, "w")
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            cwd=REPO)
    sampler = Sampler(proc, include_ollama=(engine == "ollama"),
                      cap_gb=args.ram_cap_gb)
    sampler.start()

    timed_out = False
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_proc(proc)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    finally:
        sampler.stop()
        sampler.join(timeout=2)
    wall = time.perf_counter() - t0
    logf.close()

    rec = {
        "engine": engine, "engine_label": ENGINE_LABEL[engine],
        "ctx_target": ctx, "gen_target": gen,
        "peak_mem_gb": round(sampler.peak_mem_gb, 3),    # max(phys,rss) (headline)
        "peak_phys_gb": round(sampler.peak_phys_gb, 3),  # phys footprint only
        "peak_rss_gb": round(sampler.peak_rss_gb, 3),    # resident set only
        "wall_s": round(wall, 2),
        "returncode": proc.returncode,
    }
    if engine == "ollama":
        rec["peak_ollama_mem_gb"] = round(sampler.peak_ollama_gb, 3)

    payload = None
    if os.path.exists(result_file):
        try:
            with open(result_file) as f:
                payload = json.load(f)
        except Exception:
            payload = None

    if sampler.killed_oom:
        rec["status"] = "oom"
        rec["error"] = (f"process-tree memory (max phys/rss) exceeded cap "
                        f"{args.ram_cap_gb} GB (killed)")
    elif timed_out:
        rec["status"] = "timeout"
        rec["error"] = f"exceeded per-test timeout {args.timeout}s"
    elif payload is None:
        # no result file: worker died without writing it
        rec["status"] = "oom" if proc.returncode in (-9, 137) else "crash"
        rec["error"] = (f"worker exited rc={proc.returncode} with no result "
                        f"(see {os.path.relpath(log_file, REPO)})")
    else:
        rec.update({k: v for k, v in payload.items()
                    if k not in ("engine", "ctx_target", "gen_target")})
        rec.setdefault("status", "ok")

    # Headline memory: process-tree max(phys,rss); for ollama prefer its own
    # size_vram (authoritative model+KV; the process metric undercounts mmap'd,
    # lazily-faulted weights).
    mh = rec.get("peak_mem_gb")
    if engine == "ollama" and rec.get("ollama_size_vram_gb"):
        mh = round(max(rec.get("peak_mem_gb") or 0, rec["ollama_size_vram_gb"]), 3)
    rec["mem_headline_gb"] = mh

    rec["log"] = os.path.relpath(log_file, REPO)
    return rec


# ──────────────────────────────── reporting ─────────────────────────────────
def fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def write_outputs(meta, records, json_path, md_path, latest_path):
    blob = {"meta": meta, "results": records}
    for p in (json_path, latest_path):
        with open(p, "w") as f:
            json.dump(blob, f, indent=2)

    lines = []
    lines.append("# DKV multi-engine benchmark — Qwen2.5-1.5B-Instruct (Q4)\n")
    lines.append(f"- Host: {meta['host']} · {meta['ram_gb']:.1f} GB RAM · {meta['chip']}")
    lines.append(f"- Started: {meta['started']}  |  Decode tokens/test: {meta['gen']}")
    lines.append(f"- Per-test timeout: {meta['timeout']}s · RAM kill-cap: {meta['ram_cap_gb']} GB")
    lines.append("- All numbers measured; failed cells show the failure reason "
                 "(OOM/timeout/crash), never fabricated.")
    lines.append("- **Memory** = peak of per-process `max(phys_footprint, RSS)` "
                 "summed over the engine's process tree, sampled at 20 Hz. Rationale: "
                 "MLX (active/dense) keeps weights/KV in Metal buffers counted by "
                 "`phys_footprint` but not RSS; llama.cpp/ggml (ollama, native) mmap "
                 "their GGUF weights, counted by RSS but not `phys_footprint`. Taking "
                 "the larger per process avoids undercounting either family. The "
                 "`(RSS GB)` column is plain resident set, for reference. ollama is "
                 "measured on its `llama-server` process tree.\n")
    lines.append("**Engines**")
    for e in meta["engines"]:
        lines.append(f"- `{e}` — {ENGINE_LABEL[e]}")
    lines.append("")

    by = {}
    for r in records:
        by[(r["engine"], r["ctx_target"])] = r

    for metric, label, spec in (
        ("prefill_s", "Prefill time (s)", ".2f"),
        ("decode_tps", "Decode throughput (tok/s)", ".1f"),
        ("mem_headline_gb", "Peak memory (GB)", ".2f"),
    ):
        lines.append(f"## {label}\n")
        hdr = "| Engine | " + " | ".join(f"{c // 1024}k" for c in meta["contexts"]) + " |"
        sep = "|" + "---|" * (len(meta["contexts"]) + 1)
        lines.append(hdr)
        lines.append(sep)
        for e in meta["engines"]:
            cells = []
            for c in meta["contexts"]:
                r = by.get((e, c))
                if r is None:
                    cells.append("·")
                elif r["status"] != "ok":
                    cells.append(f"**{r['status'].upper()}**")
                else:
                    cells.append(fmt(r.get(metric), spec))
            lines.append(f"| {e} | " + " | ".join(cells) + " |")
        lines.append("")

    # detail table
    lines.append("## Per-run detail\n")
    lines.append("| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | "
                 "Decode tok/s | Peak mem GB | (RSS GB) | MLX peak GB | Needle | Wall s |")
    lines.append("|" + "---|" * 12)
    for c in meta["contexts"]:
        for e in meta["engines"]:
            r = by.get((e, c))
            if r is None:
                continue
            lines.append(
                f"| {e} | {c // 1024}k | {r['status']} | "
                f"{fmt(r.get('prompt_tokens'), 'd')} | {fmt(r.get('gen_tokens'), 'd')} | "
                f"{fmt(r.get('prefill_s'), '.2f')} | {fmt(r.get('decode_tps'), '.1f')} | "
                f"{fmt(r.get('mem_headline_gb'), '.2f')} | {fmt(r.get('peak_rss_gb'), '.2f')} | "
                f"{fmt(r.get('mx_peak_gb'), '.2f')} | "
                f"{'Y' if r.get('needle_found') else ('-' if r.get('status') == 'ok' else '')} | "
                f"{fmt(r.get('wall_s'), '.1f')} |")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# ──────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES)
    ap.add_argument("--contexts", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-test wall timeout (s); exceeding it = failure (skip larger)")
    ap.add_argument("--ram-cap-gb", type=float, default=7.5,
                    help="kill a worker if process-tree RSS exceeds this (8 GB box guard)")
    ap.add_argument("--native-binary",
                    default=os.path.join(REPO, "dkv_native/build/dkv_native"))
    ap.add_argument("--native-model",
                    default=os.path.join(REPO, "dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
    ap.add_argument("--dense-model-id", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--ollama-model", default="qwen2.5:1.5b-instruct")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"results_{stamp}.json")
    latest_path = os.path.join(RESULTS_DIR, "results_latest.json")
    md_path = os.path.join(RESULTS_DIR, "summary.md")

    # Build all prompts up front (cached); same text for every engine.
    print("Building NIAH prompts ...", flush=True)
    from bench_common import build_niah_prompt, _load_ref_tokenizer
    tok = _load_ref_tokenizer()
    prompt_files = {}
    prompt_actual = {}
    for c in args.contexts:
        text, n = build_niah_prompt(c, tok)
        pf = os.path.join(RESULTS_DIR, f"prompt_{c}.txt")
        with open(pf, "w", encoding="utf-8") as f:
            f.write(text)
        prompt_files[c] = pf
        prompt_actual[c] = n
        print(f"  ctx {c // 1024:>3}k -> {n} ref-tokens ({len(text)} chars)", flush=True)

    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "chip": "Apple M3",
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "engines": args.engines,
        "contexts": args.contexts,
        "gen": args.gen,
        "timeout": args.timeout,
        "ram_cap_gb": args.ram_cap_gb,
        "model": "Qwen2.5-1.5B-Instruct (Q4)",
        "prompt_ref_tokens": prompt_actual,
    }

    records = []
    dead = set()  # engines that failed at a smaller context -> skip larger

    print("\n" + "=" * 100)
    print(f"{'ctx':>5} {'engine':>8} {'status':>8} {'prefill_s':>10} "
          f"{'tok/s':>8} {'peakMem':>8} {'mlxPk':>7} {'pTok':>7} {'gTok':>6} {'needle':>6}")
    print("=" * 100, flush=True)

    for ctx in args.contexts:
        for engine in args.engines:
            if engine in dead:
                rec = {"engine": engine, "engine_label": ENGINE_LABEL[engine],
                       "ctx_target": ctx, "gen_target": args.gen,
                       "status": "skipped",
                       "error": "engine failed at a smaller context"}
                records.append(rec)
                write_outputs(meta, records, json_path, md_path, latest_path)
                print(f"{ctx // 1024:>4}k {engine:>8} {'skip':>8} "
                      f"{'(failed earlier)':>40}", flush=True)
                continue

            rec = run_one(engine, ctx, args.gen, prompt_files[ctx], args)
            records.append(rec)
            write_outputs(meta, records, json_path, md_path, latest_path)

            st = rec["status"]
            print(f"{ctx // 1024:>4}k {engine:>8} {st:>8} "
                  f"{fmt(rec.get('prefill_s'), '10.2f')} "
                  f"{fmt(rec.get('decode_tps'), '8.1f')} "
                  f"{fmt(rec.get('mem_headline_gb'), '8.2f')} "
                  f"{fmt(rec.get('mx_peak_gb'), '7.2f')} "
                  f"{fmt(rec.get('prompt_tokens'), '7d')} "
                  f"{fmt(rec.get('gen_tokens'), '6d')} "
                  f"{('Y' if rec.get('needle_found') else '-'):>6}", flush=True)
            if st != "ok":
                dead.add(engine)
                print(f"      -> {engine} marked DEAD ({st}: {rec.get('error', '')[:80]}); "
                      f"skipping larger contexts.", flush=True)

    if "ollama" in args.engines:
        ollama_unload(args.ollama_model)

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    write_outputs(meta, records, json_path, md_path, latest_path)
    print("\nDone.")
    print(f"  JSON:    {os.path.relpath(json_path, REPO)}")
    print(f"  Summary: {os.path.relpath(md_path, REPO)}")


if __name__ == "__main__":
    main()
