import os
import sys
import time
import gc
import json
import argparse
import subprocess
import threading
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Subprocess Memory Tracker ─────────────────────────────────────────────────

class MemoryTracker(threading.Thread):
    def __init__(self, interval=0.005):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0.0
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                mem = {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'rss_mb': 0.0}
                try:
                    import psutil
                    mem['rss_mb'] = psutil.Process().memory_info().rss / 1e6
                except Exception:
                    pass
                if torch.backends.mps.is_available():
                    try:
                        mem['allocated_mb'] = torch.mps.current_allocated_memory() / 1e6
                        mem['reserved_mb'] = torch.mps.driver_allocated_memory() / 1e6
                    except Exception:
                        pass
                
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

# ── NIAH Prompt Builder ────────────────────────────────────────────────────────

def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    
    # Estimate remaining room for templates and questions
    target_filler_tokens = context_length - len(needle_tokens) - 100
    if target_filler_tokens < 0:
        target_filler_tokens = 100
        
    num_repeats = (target_filler_tokens // len(filler_tokens)) + 1
    all_filler_tokens = (filler_tokens * num_repeats)[:target_filler_tokens]
    
    insert_idx = int(len(all_filler_tokens) * depth)
    part1_tokens = all_filler_tokens[:insert_idx]
    part2_tokens = all_filler_tokens[insert_idx:]
    
    part1_text = tokenizer.decode(part1_tokens)
    part2_text = tokenizer.decode(part2_tokens)
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1_text + "\n"
        + needle + "\n"
        + part2_text + "\n\n"
        + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

# ── Worker Logic ──────────────────────────────────────────────────────────────

def run_single_benchmark(mode, context_len, rank, micro_block_size, preset, approximate_attn):
    # Set the overrides
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"  # Ensure compression engages for all context sizes
    
    if preset:
        os.environ["DIFFKV_PRESET"] = preset
    if approximate_attn is not None:
        os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1" if approximate_attn else "0"
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load the tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Create the needle and question
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    
    # Build prompt
    prompt = make_niah_prompt(tokenizer, context_len, 0.5, needle, question)
    
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        if mode == "standard":
            # Standard model uses standard Transformers
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            ).to(device)
            model.eval()
            
            # Warmup pass to compile/initialize GPU kernels
            with torch.no_grad():
                _ = model(tokenizer("Warmup.", return_tensors="pt").input_ids.to(device), use_cache=True)
            
            # Reset cache/collect
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]
            
            # Prefill time measurement
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(inputs.input_ids, use_cache=True)
            if device.type == "mps":
                torch.mps.synchronize()
            t_prefill = time.perf_counter() - t0
            
            # Decode time measurement (generate 64 tokens)
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
            t1 = time.perf_counter()
            current_past = past_key_values
            current_input = next_token_id
            
            generated_tokens = []
            with torch.no_grad():
                for _ in range(64):
                    outputs = model(
                        input_ids=current_input,
                        past_key_values=current_past,
                        use_cache=True
                    )
                    current_past = outputs.past_key_values
                    current_input = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                    generated_tokens.append(current_input.item())
            if device.type == "mps":
                torch.mps.synchronize()
            t_decode = time.perf_counter() - t1
            tps = 64 / max(t_decode, 0.001)
            
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            accuracy = 1.0 if "847291" in response else 0.0
            
            tracker.stop()
            tracker.join()
            
            res = {
                "prefill_s": t_prefill,
                "decode_tps": tps,
                "peak_rss_mb": tracker.peak_rss,
                "peak_allocated_mb": tracker.peak_allocated,
                "peak_reserved_mb": tracker.peak_reserved,
                "avg_cos_sim": 1.0,
                "accuracy": accuracy,
                "response": response
            }
            print(json.dumps(res))
            
        elif mode == "diffkv":
            # Import DiffKV HF Wrapper from ACTIVE_RUNTIME
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))
            from serving.hf_diffkv_wrapper import DiffKVHFWrapper
            
            config = {}
            if rank:
                config["rank"] = rank
            if micro_block_size:
                config["block_size"] = micro_block_size  # MLX uses block_size parameter
                config["micro_block_size"] = micro_block_size
            if preset:
                config["preset"] = preset
                
            wrapper = DiffKVHFWrapper(
                model_id=MODEL_ID,
                config=config,
                device=device.type,
            )
            
            # Warmup pass to compile/initialize MLX/MPS kernels
            _ = wrapper.generate(prompt="Warmup run.", max_new_tokens=1, temperature=0.0)
            wrapper.manager.clear_session("default")
            
            # Reset cache/collect
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
                
            # Measure prefill (warm up + first token generation)
            t0 = time.perf_counter()
            _ = wrapper.generate(prompt=prompt, max_new_tokens=1, temperature=0.0)
            if device.type == "mps":
                torch.mps.synchronize()
            t_prefill = time.perf_counter() - t0
            
            # Clear default session to isolate decode performance
            wrapper.manager.clear_session("default")
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
                
            # Decode time measurement (generate 64 tokens)
            t1 = time.perf_counter()
            response_full = wrapper.generate(prompt=prompt, max_new_tokens=64, temperature=0.0)
            if device.type == "mps":
                torch.mps.synchronize()
            t_total = time.perf_counter() - t1
            
            t_decode = t_total - t_prefill
            tps = 64 / max(t_decode, 0.001)
            
            # Extract newly generated tokens
            prompt_len = len(wrapper.tokenizer(prompt).input_ids)
            session_id = wrapper.active_session or "default"
            stored_ids = getattr(wrapper, "_session_token_ids", {}).get(session_id, [])
            new_tokens = stored_ids[prompt_len:]
            response = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            accuracy = 1.0 if "847291" in response else 0.0
            
            # Get Quality Cosine Similarity (fallback to 1.0 if not supported by manager)
            avg_cos_sim = 1.0
            if hasattr(wrapper.manager, "runtime_summary"):
                summary = wrapper.manager.runtime_summary()
                avg_cos_sim = summary.get("avg_cosine_sim", 1.0)
            
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
                "accuracy": accuracy,
                "response": response
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

# ── Parent Orchestrator ───────────────────────────────────────────────────────

def run_subprocess_run(mode, context_len, rank=16, micro_block_size=16, preset=None, approximate_attn=None):
    cmd = [
        sys.executable,
        __file__,
        "--run-single",
        "--mode", mode,
        "--context", str(context_len),
        "--rank", str(rank),
        "--micro-block-size", str(micro_block_size)
    ]
    if preset:
        cmd += ["--preset", preset]
    if approximate_attn is not None:
        cmd += ["--approximate-attn", "1" if approximate_attn else "0"]
        
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode != 0:
            return {"error": f"Process exited with code {result.returncode}", "stderr": result.stderr}
        
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        return {"error": "No JSON output found", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "TimeoutExpired after 180s"}
    except Exception as e:
        return {"error": str(e)}

# ── Plotting Logic ────────────────────────────────────────────────────────────

def generate_plots(results_data, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    
    contexts = [1024, 2048, 4096, 8192, 16384]
    
    dense_mem = []
    diff_mem = []
    
    dense_prefill = []
    diff_prefill = []
    
    dense_tps = []
    diff_tps = []
    
    dense_acc = []
    diff_acc = []
    
    for c in contexts:
        d_res = results_data.get("standard", {}).get(str(c), {})
        df_res = results_data.get("diffkv", {}).get(str(c), {})
        
        if "error" in d_res or not d_res:
            dense_mem.append(None)
            dense_prefill.append(None)
            dense_tps.append(None)
            dense_acc.append(None)
        else:
            dense_mem.append(max(d_res.get("peak_rss_mb", 0.0), d_res.get("peak_reserved_mb", 0.0)) / 1024.0)
            dense_prefill.append(d_res.get("prefill_s", 0.0))
            dense_tps.append(d_res.get("decode_tps", 0.0))
            dense_acc.append(d_res.get("accuracy", 0.0))
            
        if "error" in df_res or not df_res:
            diff_mem.append(None)
            diff_prefill.append(None)
            diff_tps.append(None)
            diff_acc.append(None)
        else:
            diff_mem.append(max(df_res.get("peak_rss_mb", 0.0), df_res.get("peak_reserved_mb", 0.0)) / 1024.0)
            diff_prefill.append(df_res.get("prefill_s", 0.0))
            diff_tps.append(df_res.get("decode_tps", 0.0))
            diff_acc.append(df_res.get("accuracy", 0.0))

    plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. Peak Memory Footprint
    plt.figure(figsize=(8, 5))
    valid_d_mem = [(c, m) for c, m in zip(contexts, dense_mem) if m is not None]
    valid_df_mem = [(c, m) for c, m in zip(contexts, diff_mem) if m is not None]
    
    if valid_d_mem:
        plt.plot([x[0] for x in valid_d_mem], [x[1] for x in valid_d_mem], 'o-', label='Dense baseline', color='#ff7675', linewidth=2.5, markersize=8)
    if valid_df_mem:
        plt.plot([x[0] for x in valid_df_mem], [x[1] for x in valid_df_mem], 's-', label='DiffKV', color='#0984e3', linewidth=2.5, markersize=8)
        
    plt.title('Peak RAM/VRAM Footprint vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Peak Memory (GB)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph_memory.png'), dpi=200)
    plt.close()
    
    # 2. Prefill Latency
    plt.figure(figsize=(8, 5))
    valid_d_pre = [(c, m) for c, m in zip(contexts, dense_prefill) if m is not None]
    valid_df_pre = [(c, m) for c, m in zip(contexts, diff_prefill) if m is not None]
    
    if valid_d_pre:
        plt.plot([x[0] for x in valid_d_pre], [x[1] for x in valid_d_pre], 'o-', label='Dense baseline', color='#ff7675', linewidth=2.5, markersize=8)
    if valid_df_pre:
        plt.plot([x[0] for x in valid_df_pre], [x[1] for x in valid_df_pre], 's-', label='DiffKV', color='#0984e3', linewidth=2.5, markersize=8)
        
    plt.title('Prefill Latency vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Prefill Execution Time (seconds)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph_prefill_latency.png'), dpi=200)
    plt.close()

    # 3. Decode TPS
    plt.figure(figsize=(8, 5))
    valid_d_tps = [(c, m) for c, m in zip(contexts, dense_tps) if m is not None]
    valid_df_tps = [(c, m) for c, m in zip(contexts, diff_tps) if m is not None]
    
    if valid_d_tps:
        plt.plot([x[0] for x in valid_d_tps], [x[1] for x in valid_d_tps], 'o-', label='Dense baseline', color='#ff7675', linewidth=2.5, markersize=8)
    if valid_df_tps:
        plt.plot([x[0] for x in valid_df_tps], [x[1] for x in valid_df_tps], 's-', label='DiffKV', color='#0984e3', linewidth=2.5, markersize=8)
        
    plt.title('Decode Throughput (TPS) vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Throughput (tokens/second)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph_decode_tps.png'), dpi=200)
    plt.close()

    # 4. Retrieval Accuracy
    plt.figure(figsize=(8, 5))
    valid_d_acc = [(c, m) for c, m in zip(contexts, dense_acc) if m is not None]
    valid_df_acc = [(c, m) for c, m in zip(contexts, diff_acc) if m is not None]
    
    if valid_d_acc:
        plt.plot([x[0] for x in valid_d_acc], [x[1] for x in valid_d_acc], 'o-', label='Dense baseline', color='#ff7675', linewidth=2.5, markersize=8)
    if valid_df_acc:
        plt.plot([x[0] for x in valid_df_acc], [x[1] for x in valid_df_acc], 's-', label='DiffKV', color='#0984e3', linewidth=2.5, markersize=8)
        
    plt.title('NIAH Retrieval Accuracy vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Accuracy (0.0 to 1.0)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.ylim(-0.1, 1.1)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph_accuracy.png'), dpi=200)
    plt.close()

    # 5. Ablation: Rank Study
    ablation_rank_data = results_data.get("ablation_rank", {})
    ranks = [int(r) for r in sorted(ablation_rank_data.keys(), key=int)]
    ab_cos = [ablation_rank_data[str(r)].get("avg_cos_sim", 0.0) for r in ranks]
    ab_tps = [ablation_rank_data[str(r)].get("decode_tps", 0.0) for r in ranks]
    
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
    ax2.set_ylabel('Decode Speed (TPS)', color=color, fontsize=12)
    ax2.plot(ranks, ab_tps, 's--', color=color, linewidth=2.0, markersize=8, label='Decode TPS')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Ablation: Rank (R) vs. Quality vs. Speed (Context 4096)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph_ablation_rank.png'), dpi=200)
    plt.close()

    # 6. Ablation: Block Size Study
    ablation_bs_data = results_data.get("ablation_bs", {})
    blocks = [int(b) for b in sorted(ablation_bs_data.keys(), key=int)]
    ab_bs_tps = [ablation_bs_data[str(b)].get("decode_tps", 0.0) for b in blocks]
    ab_bs_mem = [max(ablation_bs_data[str(b)].get("peak_rss_mb", 0.0), ablation_bs_data[str(b)].get("peak_reserved_mb", 0.0)) / 1024.0 for b in blocks]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = '#ff6b6b'
    ax1.set_xlabel('Block Size (B)', fontsize=12)
    ax1.set_ylabel('Decode Speed (TPS)', color=color, fontsize=12)
    ax1.plot(blocks, ab_bs_tps, 'o-', color=color, linewidth=2.5, markersize=8, label='TPS')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(blocks)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    ax2 = ax1.twinx()  
    color = '#4d3f72'
    ax2.set_ylabel('Peak Memory (GB)', color=color, fontsize=12)
    ax2.plot(blocks, ab_bs_mem, 's--', color=color, linewidth=2.0, markersize=8, label='Memory (GB)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Ablation: Block Size (B) vs. Speed vs. Memory (Context 4096)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'graph_ablation_block_size.png'), dpi=200)
    plt.close()

# ── Main Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--mode", choices=["standard", "diffkv"])
    parser.add_argument("--context", type=int)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--micro-block-size", type=int, default=16)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--approximate-attn", type=str, default=None)
    args = parser.parse_args()
    
    if args.run_single:
        attn_val = None
        if args.approximate_attn is not None:
            attn_val = (args.approximate_attn == "1")
        run_single_benchmark(args.mode, args.context, args.rank, args.micro_block_size, args.preset, attn_val)
        sys.exit(0)
        
    print("=" * 80)
    print("      LAUNCHING DENSE VS DIFFKV BENCHMARK SUITE (1K TO 16K)")
    print("=" * 80)
    
    contexts = [1024, 2048, 4096, 8192, 16384]
    results = {"standard": {}, "diffkv": {}, "ablation_rank": {}, "ablation_bs": {}}
    
    # 1. Run standard dense baseline one-by-one
    print("\n--- Phase 1: Running Standard Dense Baselines ---")
    for c in contexts:
        print(f"Running Dense | Ctx: {c}...")
        res = run_subprocess_run("standard", c)
        if "error" in res:
            print(f"  --> Skip/OOM: {res['error']}")
            results["standard"][str(c)] = {"error": res['error']}
        else:
            print(f"  --> Done: TTFT={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | Acc={res['accuracy']:.1f}")
            results["standard"][str(c)] = res
            
    # 2. Run DiffKV baseline one-by-one (default rank 16)
    print("\n--- Phase 2: Running DiffKV Baselines ---")
    for c in contexts:
        print(f"Running DiffKV | Ctx: {c}...")
        res = run_subprocess_run("diffkv", c, rank=16, preset="low")
        if "error" in res:
            print(f"  --> Skip/OOM: {res['error']}")
            results["diffkv"][str(c)] = {"error": res['error']}
        else:
            print(f"  --> Done: TTFT={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | Cos={res['avg_cos_sim']:.4f} | Acc={res['accuracy']:.1f}")
            results["diffkv"][str(c)] = res
            
    # 3. Ablation Rank Study (R = [8, 16, 32, 64] at Context 4096)
    print("\n--- Phase 3: Running Rank Ablations (Context 4096) ---")
    for r in [8, 16, 32, 64]:
        print(f"Running DiffKV Rank Ablation | R={r} | Ctx: 4096...")
        res = run_subprocess_run("diffkv", 4096, rank=r, preset="low")
        if "error" in res:
            print(f"  --> Skip/OOM: {res['error']}")
        else:
            print(f"  --> Done: Cos={res['avg_cos_sim']:.4f} | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB")
            results["ablation_rank"][str(r)] = res
            
    # 4. Ablation Block Size Study (B = [64, 128, 256, 512] at Context 4096)
    print("\n--- Phase 4: Running Block Size Ablations (Context 4096) ---")
    for b in [64, 128, 256, 512]:
        print(f"Running DiffKV Block Size Ablation | B={b} | Ctx: 4096...")
        res = run_subprocess_run("diffkv", 4096, rank=16, micro_block_size=b, preset="low")
        if "error" in res:
            print(f"  --> Skip/OOM: {res['error']}")
        else:
            print(f"  --> Done: TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB")
            results["ablation_bs"][str(b)] = res
            
    # Save raw results
    out_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results_custom.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {out_json}")
    
    # Generate plots in the current artifact folder
    artifact_dir = "/Users/omchimurkar1/.gemini/antigravity/brain/ada31170-301d-45cf-bbdf-321c6b861dbc"
    if not os.path.exists(artifact_dir):
        artifact_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_plots")
        
    print(f"Generating plots in {artifact_dir}...")
    try:
        generate_plots(results, artifact_dir)
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
