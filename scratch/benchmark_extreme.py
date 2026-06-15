import os
import sys
import time
import gc
import json
import argparse
import subprocess
import threading
import psutil
import numpy as np

# Create the memory tracker
class MemoryTracker(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0.0
        self.process = psutil.Process()

    def run(self):
        while not self.stop_event.is_set():
            try:
                if self.process.is_running():
                    rss = self.process.memory_info().rss / 1e6  # MB
                    if rss > self.peak_rss:
                        self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# Helper to build the exact prompt
def build_exact_prompt(tokenizer, book_tokens, L, depth, needle, prefix, suffix):
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    T_meta = len(prefix_ids) + len(needle_ids) + len(suffix_ids)
    T_book = L - T_meta
    
    if T_book < 100:
        raise ValueError(f"Context length L={L} is too small to contain metadata and needle.")
        
    adj = 0
    for _ in range(5):
        curr_T_book = T_book + adj
        book_slice = book_tokens[:curr_T_book]
        insert_idx = int(len(book_slice) * depth)
        prompt_ids = prefix_ids + book_slice[:insert_idx] + needle_ids + book_slice[insert_idx:] + suffix_ids
        prompt_text = tokenizer.decode(prompt_ids)
        re_encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
        diff = L - len(re_encoded)
        if diff == 0:
            return prompt_text, re_encoded
        else:
            adj += diff
            
    return prompt_text, re_encoded

# Single execution logic
def run_single_benchmark(mode, context_len, model_id, rank, block_size):
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        import mlx.core as mx
        from transformers import AutoTokenizer
        
        # Setup actual model ID
        if "0.5b" in model_id.lower():
            actual_model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        elif "1.5b" in model_id.lower():
            actual_model_id = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        else:
            actual_model_id = model_id
            
        tokenizer = AutoTokenizer.from_pretrained(actual_model_id)
        
        # Load book tokens
        with open("scratch/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
            book_text = f.read()
        book_tokens = tokenizer.encode(book_text, add_special_tokens=False)
        
        # Build prompt
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        needle = "The special code is 847291."
        suffix = "\n\nWhat is the special code? Answer in exactly the 6-digit code number.<|im_end|>\n<|im_start|>assistant\n"
        
        prompt_text, prompt_ids = build_exact_prompt(tokenizer, book_tokens, context_len, 0.5, needle, prefix, suffix)
        
        # Collect garbage and empty caches
        gc.collect()
        mx.metal.clear_cache()
        
        if mode == "diffkv":
            # Add ACTIVE_RUNTIME to path
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))
            import torch
            from serving.hf_diffkv_wrapper import DiffKVHFWrapper
            
            # Subclass wrapper to accept prompt_ids directly
            class BenchmarkedDiffKVWrapper(DiffKVHFWrapper):
                def generate_benchmark_ids(self, prompt_ids, max_new_tokens=64):
                    self.ensure_loaded()
                    session_id = self.active_session or "default"
                    
                    self.manager.clear_session(session_id)
                    self._session_token_ids[session_id] = []
                    
                    self.manager.init_session(session_id, prefill_len=len(prompt_ids))
                    self.manager.register_prefill_tokens(session_id, torch.tensor(prompt_ids, dtype=torch.long))
                    self.model._diffkv_session_ids = [session_id]

                    PREFILL_CHUNK = 512
                    output = None
                    
                    # Reset peak memory
                    mx.reset_peak_memory()
                    t0 = time.perf_counter()
                    
                    for chunk_start in range(0, len(prompt_ids), PREFILL_CHUNK):
                        chunk = prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
                        clen = len(chunk)
                        abs_start = chunk_start
                        
                        chunk_tensor = torch.tensor([chunk], dtype=torch.long)
                        pos_tensor = torch.tensor([list(range(abs_start, abs_start + clen))], dtype=torch.long)
                        
                        output = self.model(chunk_tensor, pos_tensor)
                        self.manager.compress_deferred_prefill_blocks(session_id)
                    
                    mx.eval()
                    t_prefill = time.perf_counter() - t0

                    generated = prompt_ids.copy()
                    cur_pos = len(prompt_ids)
                    logits = output.logits[0, -1].cpu().numpy()

                    t1 = time.perf_counter()
                    for step_idx in range(max_new_tokens):
                        next_id = int(np.argmax(logits))
                        generated.append(next_id)
                        self.manager.register_prefill_tokens(session_id, torch.tensor([next_id], dtype=torch.long))

                        input_ids = torch.tensor([[next_id]], dtype=torch.long)
                        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long)
                        
                        output = self.model(input_ids, pos_tensor)
                        logits = output.logits[0, -1].cpu().numpy()
                        cur_pos += 1
                    
                    mx.eval()
                    t_decode = time.perf_counter() - t1
                    
                    response_text = self.tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).strip()
                    
                    return {
                        "prefill_s": t_prefill,
                        "decode_tps": max_new_tokens / max(t_decode, 0.001),
                        "response": response_text,
                        "peak_mlx_mb": mx.get_peak_memory() / 1e6
                    }
            
            config = {
                "quantization": "int4",
                "rank": rank,
                "block_size": block_size,
                "micro_block_size": block_size,
            }
            
            wrapper = BenchmarkedDiffKVWrapper(
                model_id=actual_model_id,
                config=config,
                device="mps",
            )
            
            res_bench = wrapper.generate_benchmark_ids(prompt_ids, max_new_tokens=64)
            wrapper.stop()
            
        elif mode == "dense":
            from mlx_lm.utils import load as mlx_load
            from mlx_lm.models.cache import make_prompt_cache
            
            model, tokenizer = mlx_load(actual_model_id)
            cache = make_prompt_cache(model)
            
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            logits = None
            PREFILL_CHUNK = 512
            for chunk_start in range(0, len(prompt_ids), PREFILL_CHUNK):
                chunk = prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
                chunk_arr = mx.array([chunk])
                logits = model(chunk_arr, cache=cache)
                mx.eval(logits)
                
            t_prefill = time.perf_counter() - t0
            
            generated = list(prompt_ids)
            cur_token = int(mx.argmax(logits[0, -1]).item())
            generated.append(cur_token)
            
            t1 = time.perf_counter()
            for _ in range(63):
                token_arr = mx.array([[cur_token]])
                logits = model(token_arr, cache=cache)
                mx.eval(logits)
                cur_token = int(mx.argmax(logits[0, -1]).item())
                generated.append(cur_token)
                
            mx.eval()
            t_decode = time.perf_counter() - t1
            
            response_text = tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).strip()
            
            res_bench = {
                "prefill_s": t_prefill,
                "decode_tps": 64 / max(t_decode, 0.001),
                "response": response_text,
                "peak_mlx_mb": mx.get_peak_memory() / 1e6
            }
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        tracker.stop()
        tracker.join()
        
        accuracy = 1.0 if "847291" in res_bench["response"] else 0.0
        
        res = {
            "prefill_s": res_bench["prefill_s"],
            "decode_tps": res_bench["decode_tps"],
            "peak_rss_mb": tracker.peak_rss,
            "peak_mlx_mb": res_bench["peak_mlx_mb"],
            "accuracy": accuracy,
            "response": res_bench["response"]
        }
        
        print(json.dumps(res))
        
    except Exception as e:
        tracker.stop()
        try:
            tracker.join()
        except Exception:
            pass
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))

# Parent Orchestrator execution
def run_subprocess_run(mode, context_len, model_id, rank=16, block_size=256):
    cmd = [
        sys.executable,
        __file__,
        "--run-single",
        "--mode", mode,
        "--context", str(context_len),
        "--model-id", model_id,
        "--rank", str(rank),
        "--block-size", str(block_size)
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=400,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode != 0:
            return {"error": f"ExitCode {result.returncode}", "stderr": result.stderr}
            
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        return {"error": "No JSON output found", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "TimeoutExpired"}
    except Exception as e:
        return {"error": str(e)}

# Generate and save plots
def generate_plots(results_path, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    contexts = [32768, 49152, 65536]
    models = ["0.5b", "1.5b"]
    modes = ["dense", "diffkv"]
    
    labels = {
        ("0.5b", "dense"): "Dense 0.5B",
        ("0.5b", "diffkv"): "DiffKV 0.5B",
        ("1.5b", "dense"): "Dense 1.5B",
        ("1.5b", "diffkv"): "DiffKV 1.5B",
    }
    
    colors = {
        ("0.5b", "dense"): "#ff7675",
        ("0.5b", "diffkv"): "#0984e3",
        ("1.5b", "dense"): "#d63031",
        ("1.5b", "diffkv"): "#00cec9",
    }
    
    markers = {
        ("0.5b", "dense"): "o",
        ("0.5b", "diffkv"): "s",
        ("1.5b", "dense"): "^",
        ("1.5b", "diffkv"): "d",
    }
    
    linestyles = {
        ("0.5b", "dense"): "--",
        ("0.5b", "diffkv"): "-",
        ("1.5b", "dense"): "--",
        ("1.5b", "diffkv"): "-",
    }

    plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    metrics = ["prefill_s", "decode_tps", "peak_rss_mb", "accuracy"]
    titles = {
        "prefill_s": "Prefill Latency (TTFT) vs. Context Length",
        "decode_tps": "Decode Throughput (TPS) vs. Context Length",
        "peak_rss_mb": "Peak Host RSS Memory vs. Context Length",
        "accuracy": "Needle Retrieval Accuracy vs. Context Length"
    }
    y_labels = {
        "prefill_s": "Latency (seconds)",
        "decode_tps": "Throughput (tokens/sec)",
        "peak_rss_mb": "Memory (MB)",
        "accuracy": "Accuracy (0.0 - 1.0)"
    }
    
    for metric in metrics:
        plt.figure(figsize=(9, 6))
        
        for model in models:
            for mode in modes:
                key = (model, mode)
                x_vals = []
                y_vals = []
                for c in contexts:
                    val = data.get(model, {}).get(mode, {}).get(str(c), {}).get(metric)
                    if val is not None and not isinstance(val, str):
                        x_vals.append(c)
                        y_vals.append(val)
                
                if x_vals:
                    plt.plot(
                        x_vals,
                        y_vals,
                        marker=markers[key],
                        linestyle=linestyles[key],
                        color=colors[key],
                        label=labels[key],
                        linewidth=2.0,
                        markersize=8
                    )
                    
        plt.title(titles[metric], fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Context Length (tokens)', fontsize=12)
        plt.ylabel(y_labels[metric], fontsize=12)
        plt.xticks(contexts, ["32K", "48K", "64K"])
        
        if metric == "accuracy":
            plt.ylim(-0.1, 1.1)
        elif metric == "decode_tps":
            plt.ylim(0, None)
            
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'compare_extreme_{metric}.png'), dpi=200)
        plt.close()
        
    print("Extreme context comparative plots generated successfully.")

# Main coordinator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--mode", choices=["dense", "diffkv"])
    parser.add_argument("--context", type=int)
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    
    if args.run_single:
        run_single_benchmark(args.mode, args.context, args.model_id, args.rank, args.block_size)
        sys.exit(0)
        
    print("=" * 80)
    print("      LAUNCHING EXTREME CONTEXT BENCHMARK SUITE (32K, 48K, 64K)")
    print("=" * 80)
    
    contexts = [32768, 49152, 65536]
    models = {
        "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
    }
    modes = ["dense", "diffkv"]
    
    results = {
        "0.5b": {"dense": {}, "diffkv": {}},
        "1.5b": {"dense": {}, "diffkv": {}}
    }
    
    for model_key, model_id in models.items():
        for mode in modes:
            for c in contexts:
                print(f"\n>>> Running: Model={model_key} | Mode={mode} | Context={c} ...")
                res = run_subprocess_run(mode, c, model_id)
                print(f"Result: {res}")
                results[model_key][mode][c] = res
                
    # Save results
    results_path = "benchmark_results_extreme.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")
    
    # Generate plots
    artifact_dir = "/Users/omchimurkar1/.gemini/antigravity/brain/ada31170-301d-45cf-bbdf-321c6b861dbc"
    generate_plots(results_path, artifact_dir)
    generate_plots(results_path, "benchmark_plots")
