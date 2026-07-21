#!/usr/bin/env python3
import sys
import os
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

import mlx.core as mx
from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

def measure_prefill_mem(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", ctx_len=32000, lego=0):
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
    os.environ["DIFFKV_LEGO_PREFILL"] = str(lego)

    try:
        wrapper = MLXDiffKVWrapper(
            model_id=model_id,
            config={"rank": 32, "block_size": 256},
        )

        toks = wrapper.tokenizer.encode(FILLER, add_special_tokens=False)
        reps = (ctx_len // len(toks)) + 1
        prompt = wrapper.tokenizer.decode((toks * reps)[:ctx_len])

        sid = f"lego_mem_{ctx_len}_{lego}"
        wrapper.manager.clear_session(sid)
        if hasattr(wrapper, "_session_token_ids"):
            wrapper._session_token_ids[sid] = []
        wrapper.active_session = sid

        reset_fn = getattr(mx, "reset_peak_memory", getattr(mx.metal, "reset_peak_memory", lambda: None))
        get_fn = getattr(mx, "get_peak_memory", getattr(mx.metal, "get_peak_memory", lambda: 0))

        reset_fn()
        t0 = time.perf_counter()
        _ = wrapper.generate(prompt=prompt, max_new_tokens=1, temperature=0.0)
        dt = time.perf_counter() - t0
        peak_vram_bytes = get_fn()
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)

        return {
            "context_tokens": ctx_len,
            "lego_enabled": bool(lego),
            "prefill_sec": round(dt, 2),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "status": "SUCCESS",
        }
    except Exception as e:
        return {
            "context_tokens": ctx_len,
            "lego_enabled": bool(lego),
            "error": str(e),
            "status": "OOM",
        }

def run_lego_eval():
    print("--- Running Lego Prefill Peak Memory Benchmark ---", flush=True)
    contexts = [16000, 32000, 48000]
    results = []

    for ctx in contexts:
        res_off = measure_prefill_mem(ctx_len=ctx, lego=0)
        res_on = measure_prefill_mem(ctx_len=ctx, lego=1)

        vram_off = res_off.get("peak_vram_gb", 0.0)
        vram_on = res_on.get("peak_vram_gb", 0.0)

        saved_gb = vram_off - vram_on if (vram_off > 0 and vram_on > 0) else 0.0
        reduction_pct = (saved_gb / vram_off) * 100.0 if vram_off > 0 else 0.0

        entry = {
            "context_tokens": ctx,
            "standard_prefill_peak_vram_gb": vram_off if res_off["status"] == "SUCCESS" else "OOM",
            "lego_prefill_peak_vram_gb": vram_on if res_on["status"] == "SUCCESS" else "OOM",
            "vram_saved_gb": round(saved_gb, 3),
            "vram_reduction_pct": round(reduction_pct, 1),
            "standard_time_sec": res_off.get("prefill_sec", None),
            "lego_time_sec": res_on.get("prefill_sec", None),
        }
        results.append(entry)

        print(f"Context {ctx:>5} tokens | Standard Peak VRAM: {res_off.get('peak_vram_gb', 'OOM')} GB | Lego Peak VRAM: {res_on.get('peak_vram_gb', 'OOM')} GB | Saved: {saved_gb:.3f} GB ({reduction_pct:.1f}%)", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test7_lego_prefill_mem.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Lego prefill memory results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_lego_eval()
