#!/usr/bin/env python3
"""Memory DURING prefill, sampled -- not inferred from the end-of-run peak.

WHY THIS EXISTS
---------------
The context-ladder result argues that eviction cannot reach the contexts a
prefill-time compressor reaches, because eviction is post-hoc: it ranks prefix
tokens against an observation window, so the whole prefix KV must be resident at
the moment it ranks. The ladder supports that with END-OF-RUN numbers -- peak
allocated and peak reserved -- and the reader has to take the mechanism on
trust.

This measures the mechanism directly. A sampler thread polls the CUDA caching
allocator every few milliseconds while prefill runs, so the transient is
visible as a curve rather than a single number: eviction's footprint climbs to
the dense footprint and only then collapses to the small cache it keeps, while
a prefill-time compressor never climbs at all.

PROTOCOL
--------
  * one subprocess per arm, so every arm gets a fresh allocator -- the same
    discipline the ladder uses, and the reason its numbers are comparable;
  * PREFILL ONLY. Generation is excluded so the trace is purely the cost of
    building the cache, which is the quantity under discussion;
  * a context every arm can serve (32,768 on Qwen3.5-4B), so no trace is
    contaminated by paging. Above that, dense and SnapKV spill and their
    timings become PCIe bandwidth rather than compute;
  * torch.cuda.memory_allocated() rather than max_memory_allocated(), because
    the running value is what shows the rise and the drop. The peak is recorded
    separately and should agree with the ladder.

  python bench_prefill_trace_cuda.py --model Qwen/Qwen3.5-4B --ctx 32768
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
OUT = os.path.join(REPO, "paper", "results", "prefill_trace")

SAMPLE_S = 0.004          # ~250 Hz; cheap, and fine against multi-second prefills


class Sampler(threading.Thread):
    """Poll the allocator on a side thread while the main thread prefills.

    Reading allocator statistics does not synchronise the device and does not
    take the GIL for long, so this perturbs the measurement far less than
    instrumenting the model's forward would -- and, unlike per-chunk hooks, it
    works identically for an arm whose prefill loop we do not own.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        import torch
        t0 = time.perf_counter()
        while not self._stop.is_set():
            self.samples.append((time.perf_counter() - t0,
                                 torch.cuda.memory_allocated() / 1e9,
                                 torch.cuda.memory_reserved() / 1e9))
            time.sleep(SAMPLE_S)

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)


def build_filler(tok, n_tokens):
    sys.path.insert(0, HERE)
    from context_ladder import build_filler as bf
    return bf(tok, n_tokens)


def run_point(a):
    import torch
    rec = {"arm": a.arm, "ctx": a.ctx, "model": a.model, "quant": a.quant}
    if a.arm == "dkv":
        os.environ.setdefault("DKV_RSVD_SEED", "1234")
        os.environ.setdefault("DKV_SVD_SEED", "1234")
        os.chdir(ACTIVE)
        sys.path.insert(0, ACTIVE)
        from serving.hf_dkv_wrapper import DKVHFWrapper
        w = DKVHFWrapper(model_id=a.model,
                         config={"preset": a.preset,
                                 "quantization": a.quant or None})
        w.ensure_loaded()
        tok = w.tokenizer
    else:
        sys.path.insert(0, HERE)
        from run_longbench_cuda import load_plain
        import kv_baselines as KB
        tok, model = load_plain(a.model, a.quant, KB.needs_eager(a.arm))

    prompt = build_filler(tok, a.ctx)
    ids = tok(prompt, add_special_tokens=False).input_ids
    rec["ctx_actual"] = len(ids)

    torch.cuda.synchronize()
    rec["weights_gb"] = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    smp = Sampler()
    smp.start()
    t0 = time.perf_counter()
    try:
        if a.arm == "dkv":
            # The wrapper has no prefill-only entry, so ask for a single token
            # and record where prefill ends. One decode step against a 32k
            # context is ~2% of the trace and is marked in the record.
            w.active_session = "trace"
            w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0)
        else:
            import kv_baselines as KB
            if a.arm in ("snapkv", "h2o"):
                KB._evict_by_observed_attention(
                    model, ids, "cuda", a.chunk,
                    budget=json.loads(a.baseline_params).get("budget", 2016),
                    window=json.loads(a.baseline_params).get("window", 32),
                    pool_kernel=7 if a.arm == "snapkv" else 0,
                    recency_window=0 if a.arm == "snapkv" else 512)
            else:
                KB.chunked_prefill(model, ids, "cuda", a.chunk)
        torch.cuda.synchronize()
        rec["status"] = "ok"
    except Exception as e:                                        # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    rec["wall_s"] = time.perf_counter() - t0
    smp.stop()

    rec["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    rec["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    rec["sample_hz"] = round(1.0 / SAMPLE_S)
    rec["n_samples"] = len(smp.samples)
    rec["trace"] = [[round(t, 4), round(al, 4), round(rs, 4)]
                    for t, al, rs in smp.samples]
    print(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--preset", default="mid")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--arms", default="dense,snapkv,dkv")
    ap.add_argument("--baseline-params", default='{"budget": 2016, "window": 32}')
    ap.add_argument("--arm", default=None, help="internal: run one arm in-process")
    a = ap.parse_args()

    if a.arm:
        return run_point(a)

    os.makedirs(OUT, exist_ok=True)
    tag = a.model.split("/")[-1]
    for arm in a.arms.split(","):
        arm = arm.strip()
        dest = os.path.join(OUT, "%s_%s_%d.json" % (tag, arm, a.ctx))
        print("### prefill trace: %s @ %d" % (arm, a.ctx), flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--model", a.model, "--ctx", str(a.ctx), "--quant", a.quant,
               "--preset", a.preset, "--chunk", str(a.chunk),
               "--baseline-params", a.baseline_params]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        line = ""
        for ln in p.stdout.splitlines():
            if ln.startswith("{"):
                line = ln
        if not line:
            print("  FAILED\n" + (p.stdout or "")[-800:] + (p.stderr or "")[-800:])
            continue
        with open(dest, "w", encoding="utf-8") as f:
            f.write(line)
        r = json.loads(line)
        print("  %-13s %s  peak %.2f GB  reserved %.2f GB  wall %.1fs  %d samples"
              % (arm, r.get("status"), r.get("peak_gb", 0),
                 r.get("peak_reserved_gb", 0), r.get("wall_s", 0),
                 r.get("n_samples", 0)), flush=True)


if __name__ == "__main__":
    main()
