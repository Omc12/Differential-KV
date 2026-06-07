import os
import sys
import argparse
import subprocess
import json
import time
import gc
import torch
import threading

# Ensure parent directory is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Subprocess Worker Logic ───────────────────────────────────────────────────

class MemoryTracker(threading.Thread):
    def __init__(self, interval=0.005):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0.0
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0

    def run(self):
        from native_core.mac_utils import get_true_diffkv_memory_mb
        while not self.stop_event.is_set():
            try:
                mem = get_true_diffkv_memory_mb()
                if mem['rss_mb'] > self.peak_rss:
                    self.peak_rss = mem['rss_mb']
                if mem['allocated_mb'] > self.peak_allocated:
                    self.peak_allocated = mem['allocated_mb']
                if mem['reserved_mb'] > self.peak_reserved:
                    self.peak_reserved = mem['reserved_mb']
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


def run_single_benchmark(mode, context_len, rank=16, micro_block_size=16):
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        prompt = "word " * context_len
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_ids = input_ids[:, :context_len]
        
        # Load weights baseline memory
        from native_core.mac_utils import get_true_diffkv_memory_mb
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        mem_before = get_true_diffkv_memory_mb()
        
        if mode == "standard":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            ).to(device)
            model.eval()
            
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            
            # Measure prefill
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
            t_prefill = time.perf_counter() - t0
            
            # Measure decode
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
            t1 = time.perf_counter()
            current_past = past_key_values
            current_input = next_token_id
            
            with torch.no_grad():
                for _ in range(64):
                    outputs = model(
                        input_ids=current_input,
                        past_key_values=current_past,
                        use_cache=True
                    )
                    current_past = outputs.past_key_values
                    current_input = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                    
            t_decode = time.perf_counter() - t1
            tps = 64 / max(t_decode, 0.001)
            
            tracker.stop()
            tracker.join()
            
            res = {
                "prefill_s": t_prefill,
                "decode_tps": tps,
                "peak_rss_mb": tracker.peak_rss,
                "peak_allocated_mb": tracker.peak_allocated,
                "peak_reserved_mb": tracker.peak_reserved,
                "avg_cos_sim": 1.0,
            }
            print(json.dumps(res))
            
        elif mode == "diffkv":
            from serving.hf_diffkv_wrapper import DiffKVHFWrapper
            config = {
                "rank": rank,
                "micro_block_size": micro_block_size,
                "serving_mode": "balanced",
            }
            
            wrapper = DiffKVHFWrapper(
                model_id=MODEL_ID,
                config=config,
                device=device.type,
            )
            
            trimmed_prompt = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                
            # Prefill (first token generation)
            t0 = time.perf_counter()
            _ = wrapper.generate(prompt=trimmed_prompt, max_new_tokens=1, temperature=0.0)
            t_prefill = time.perf_counter() - t0
            
            # Clear default session to isolate decode performance
            wrapper.manager.clear_session("default")
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                
            # Decode (64 tokens generation)
            t1 = time.perf_counter()
            _ = wrapper.generate(prompt=trimmed_prompt, max_new_tokens=64, temperature=0.0)
            t_total = time.perf_counter() - t1
            
            t_decode = t_total - t_prefill
            tps = 64 / max(t_decode, 0.001)
            
            # Get Quality Cosine Similarity
            summary = wrapper.manager.runtime_summary()
            avg_cos_sim = summary.get("avg_cosine_sim", 0.0)
            
            wrapper.stop()
            tracker.stop()
            tracker.join()
            
            res = {
                "prefill_s": t_prefill,
                "decode_tps": tps,
                "peak_rss_mb": tracker.peak_rss,
                "peak_allocated_mb": tracker.peak_allocated,
                "peak_reserved_mb": tracker.peak_reserved,
                "avg_cos_sim": avg_cos_sim,
            }
            print(json.dumps(res))
            
    except Exception as e:
        tracker.stop()
        try:
            tracker.join()
        except Exception:
            pass
        import traceback
        res = {"error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(res))


# ── Subprocess Runner & Aggregator ───────────────────────────────────────────

def run_subprocess_run(mode, context_len, rank=16, micro_block_size=16):
    cmd = [
        sys.executable,
        __file__,
        "--run-single",
        "--mode", mode,
        "--context", str(context_len),
        "--rank", str(rank),
        "--micro-block-size", str(micro_block_size)
    ]
    try:
        # Run with 150 seconds timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=150,
            cwd=_parent_dir
        )
        if result.returncode != 0:
            return {"error": f"Process exited with code {result.returncode}", "stderr": result.stderr}
        
        # Parse output for JSON line
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        return {"error": "No JSON output found", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "TimeoutExpired after 150s"}
    except Exception as e:
        return {"error": str(e)}


# ── Plotting logic ───────────────────────────────────────────────────────────

def generate_plots(results_data, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract data for primary graphs
    contexts = [512, 1024, 2048, 4096, 8192]
    
    dense_mem = []
    diff_mem = []
    
    dense_cos = []
    diff_cos = []
    
    dense_prefill = []
    diff_prefill = []
    
    dense_tps = []
    diff_tps = []
    
    for c in contexts:
        d_res = results_data.get("standard", {}).get(str(c), {})
        df_res = results_data.get("diffkv", {}).get(str(c), {})
        
        # Dense Peak RAM (RSS) or VRAM in GB
        if "error" in d_res or not d_res:
            dense_mem.append(None)
            dense_cos.append(None)
            dense_prefill.append(None)
            dense_tps.append(None)
        else:
            dense_mem.append(max(d_res.get("peak_rss_mb", 0.0), d_res.get("peak_reserved_mb", 0.0)) / 1024.0)
            dense_cos.append(1.0)
            dense_prefill.append(d_res.get("prefill_s", 0.0))
            dense_tps.append(d_res.get("decode_tps", 0.0))
            
        if "error" in df_res or not df_res:
            diff_mem.append(None)
            diff_cos.append(None)
            diff_prefill.append(None)
            diff_tps.append(None)
        else:
            diff_mem.append(max(df_res.get("peak_rss_mb", 0.0), df_res.get("peak_reserved_mb", 0.0)) / 1024.0)
            diff_cos.append(df_res.get("avg_cos_sim", 1.0))
            diff_prefill.append(df_res.get("prefill_s", 0.0))
            diff_tps.append(df_res.get("decode_tps", 0.0))

    # Apply stylish dark/clean mode formatting
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # ── Graph 1: Memory vs Context Length ─────────────────────────────────────
    plt.figure(figsize=(8, 5))
    valid_d_mem = [(c, m) for c, m in zip(contexts, dense_mem) if m is not None]
    valid_df_mem = [(c, m) for c, m in zip(contexts, diff_mem) if m is not None]
    
    if valid_d_mem:
        plt.plot([x[0] for x in valid_d_mem], [x[1] for x in valid_d_mem], 'o-', label='Dense (Baseline)', color='#e056fd', linewidth=2.5, markersize=8)
    if valid_df_mem:
        plt.plot([x[0] for x in valid_df_mem], [x[1] for x in valid_df_mem], 's-', label='DiffKV (O(seq * rank))', color='#00d2d3', linewidth=2.5, markersize=8)
        
    # Mark standard OOM / Swap boundary
    plt.axhline(y=8.0, color='#ff7675', linestyle='--', alpha=0.7, label='M3 MacBook RAM Limit (8GB)')
    
    plt.title('Memory Footprint vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Peak RAM/VRAM Footprint (GB)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph1_memory.png'), dpi=200)
    plt.close()
    
    # ── Graph 2: Quality vs Context Length ────────────────────────────────────
    plt.figure(figsize=(8, 5))
    valid_df_cos = [(c, q) for c, q in zip(contexts, diff_cos) if q is not None]
    
    plt.axhline(y=1.0, color='#e056fd', linestyle='-', linewidth=2.0, label='Dense baseline (100% Fidelity)')
    if valid_df_cos:
        # Ignore 512 for quality since it is not compressed (100% dense)
        valid_comp_cos = [x for x in valid_df_cos if x[0] > 512]
        if valid_comp_cos:
            plt.plot([x[0] for x in valid_comp_cos], [x[1] for x in valid_comp_cos], 's-', color='#00d2d3', linewidth=2.5, markersize=8, label='DiffKV Compressed Blocks')
            
    plt.title('Output Quality (Cosine Similarity) vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Cosine Similarity with Dense Keys', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.ylim(0.990, 1.002)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph2_quality.png'), dpi=200)
    plt.close()

    # ── Graph 3: Latency vs Context Length ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prefill Time
    if valid_d_mem:
        ax1.plot([x[0] for x in valid_d_mem], [dense_prefill[contexts.index(x[0])] for x in valid_d_mem], 'o-', label='Dense Prefill', color='#e056fd', linewidth=2.0)
    if valid_df_mem:
        ax1.plot([x[0] for x in valid_df_mem], [diff_prefill[contexts.index(x[0])] for x in valid_df_mem], 's-', label='DiffKV (Prefill + SVD)', color='#00d2d3', linewidth=2.0)
    ax1.set_title('Prefill Latency', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Context Length (tokens)', fontsize=10)
    ax1.set_ylabel('Prefill Execution Time (seconds)', fontsize=10)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(contexts)
    ax1.set_xticklabels([str(c) for c in contexts])
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(frameon=True)
    
    # Decode TPS
    if valid_d_mem:
        ax2.plot([x[0] for x in valid_d_mem], [dense_tps[contexts.index(x[0])] for x in valid_d_mem], 'o-', label='Dense Decode', color='#e056fd', linewidth=2.0)
    if valid_df_mem:
        ax2.plot([x[0] for x in valid_df_mem], [diff_tps[contexts.index(x[0])] for x in valid_df_mem], 's-', label='DiffKV Decode (Zero-Sync)', color='#00d2d3', linewidth=2.0)
    ax2.set_title('Decode Throughput (TPS)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Context Length (tokens)', fontsize=10)
    ax2.set_ylabel('Throughput (tokens/second)', fontsize=10)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(contexts)
    ax2.set_xticklabels([str(c) for c in contexts])
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(frameon=True)
    
    plt.suptitle('Latency and Throughput Benchmarks', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph3_latency.png'), dpi=200)
    plt.close()
    
    # ── Graph 4: Ablation Study ───────────────────────────────────────────────
    ablation_data = results_data.get("ablation", {})
    ranks = [int(r) for r in sorted(ablation_data.keys(), key=int)]
    ab_cos = [ablation_data[str(r)].get("avg_cos_sim", 0.0) for r in ranks]
    ab_vram = [ablation_data[str(r)].get("peak_reserved_mb", 0.0) for r in ranks]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = '#10ac84'
    ax1.set_xlabel('Compression Rank (R)', fontsize=12)
    ax1.set_ylabel('Quality (Cosine Similarity)', color=color, fontsize=12)
    ax1.plot(ranks, ab_cos, 'o-', color=color, linewidth=2.5, markersize=8, label='Quality')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ranks)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    ax2 = ax1.twinx()  
    color = '#2e86de'
    ax2.set_ylabel('KV Cache peak VRAM (MB)', color=color, fontsize=12)
    ax2.plot(ranks, ab_vram, 's--', color=color, linewidth=2.0, markersize=8, label='VRAM')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Ablation Study: Rank (R) vs. Quality vs. Memory (Context 2048)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph4_ablation.png'), dpi=200)
    plt.close()


# ── Main Control Loop ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--mode", choices=["standard", "diffkv"])
    parser.add_argument("--context", type=int)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--micro-block-size", type=int, default=16)
    args = parser.parse_args()
    
    if args.run_single:
        # Run one single combination in the current python process
        run_single_benchmark(args.mode, args.context, args.rank, args.micro_block_size)
        sys.exit(0)
        
    # Standard workflow: run all combinations via isolated subprocesses
    print("======================================================================")
    print("STARTING MULTI-PROCESS BENCHMARK SUITE")
    print("======================================================================")
    
    contexts = [512, 1024, 2048, 4096, 8192]
    modes = ["standard", "diffkv"]
    
    results = {"standard": {}, "diffkv": {}, "ablation": {}}
    
    # 1. Run standard dense baseline
    for c in contexts:
        print(f"\n[RUN] Mode: standard | Context: {c}")
        # For dense at 4096 and 8192, we'll try to run them but with a lower timeout/limit 
        # or catch OOM immediately. Since they are run in subprocesses, any OOM won't crash this script!
        res = run_subprocess_run("standard", c)
        if "error" in res:
            print(f"  --> Failed/Skipped: {res['error']}")
            results["standard"][str(c)] = {"error": res["error"]}
        else:
            print(f"  --> Success! Prefill={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB")
            results["standard"][str(c)] = res
            
    # 2. Run DiffKV benchmarks
    for c in contexts:
        print(f"\n[RUN] Mode: diffkv | Context: {c}")
        res = run_subprocess_run("diffkv", c, rank=16)
        if "error" in res:
            print(f"  --> Failed/Skipped: {res['error']}")
            results["diffkv"][str(c)] = {"error": res["error"]}
        else:
            print(f"  --> Success! Prefill={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | CosSim={res['avg_cos_sim']:.4f}")
            results["diffkv"][str(c)] = res
            
    # 3. Run Ablation Study (R = [8, 16, 32, 64] at context=2048)
    ablation_ranks = [8, 16, 32, 64]
    for r in ablation_ranks:
        print(f"\n[RUN-ABLATION] Mode: diffkv | Context: 2048 | Rank: {r}")
        res = run_subprocess_run("diffkv", 2048, rank=r)
        if "error" in res:
            print(f"  --> Failed/Skipped: {res['error']}")
        else:
            print(f"  --> Success! CosSim={res['avg_cos_sim']:.4f} | PeakReserved={res['peak_reserved_mb']:.1f}MB")
            results["ablation"][str(r)] = res

    # 4. Save results to JSON
    json_path = os.path.join(_parent_dir, "benchmark_clean_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw clean results saved to {json_path}")
    
    # 5. Generate plots
    artifact_dir = "/Users/omchimurkar1/.gemini/antigravity/brain/f0a31b1f-780f-434a-9632-21506ac9a8ad"
    if not os.path.exists(artifact_dir):
        # Fallback to current dir if not in conversational agent app environment
        artifact_dir = os.path.join(_parent_dir, "benchmark_plots")
        
    print(f"\nGenerating plots in {artifact_dir}...")
    generate_plots(results, artifact_dir)
    print("Done! All plots generated successfully.")
