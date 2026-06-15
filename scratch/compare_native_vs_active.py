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
def run_single_benchmark(mode, context_len):
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        from transformers import AutoTokenizer
        
        # We always use the 0.5B model for this comparison
        tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        
        # Load book tokens
        with open("scratch/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
            book_text = f.read()
        book_tokens = tokenizer.encode(book_text, add_special_tokens=False)
        
        # Build prompt
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        needle = "The special code is 847291."
        suffix = "\n\nWhat is the special code? Answer in exactly the 6-digit code number.<|im_end|>\n<|im_start|>assistant\n"
        
        prompt_text, prompt_ids = build_exact_prompt(tokenizer, book_tokens, context_len, 0.5, needle, prefix, suffix)
        
        if mode == "native":
            binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
            model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
            
            cmd = [binary_path, model_path, prompt_text]
            env = os.environ.copy()
            env["DIFFKV_USE_GPU"] = "1"
            env["DIFFKV_TIME_DECODE"] = "1"
            env["DIFFKV_MICRO_BLOCK_SIZE"] = "64"
            env["DIFFKV_MAX_CONTEXT_SLOTS"] = "512"
            
            t_prefill_start = None
            t_prefill_end = None
            decode_step_times = []
            stdout_lines = []
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            
            # Read stdout and stderr concurrently to prevent blocks and measure timings
            def read_stdout():
                for line in proc.stdout:
                    stdout_lines.append(line)
                    
            def read_stderr():
                nonlocal t_prefill_start, t_prefill_end
                for line in proc.stderr:
                    if "[DiffKV Native] Running Prefill phase in chunks..." in line:
                        t_prefill_start = time.perf_counter()
                    elif "[Prefill Phase Top predictions]" in line:
                        t_prefill_end = time.perf_counter()
                    elif "[Timing Step" in line:
                        try:
                            parts = line.split("Total: ")
                            if len(parts) > 1:
                                ms_val = float(parts[1].split("ms")[0])
                                decode_step_times.append(ms_val)
                        except Exception:
                            pass

            t_out = threading.Thread(target=read_stdout)
            t_err = threading.Thread(target=read_stderr)
            t_out.start()
            t_err.start()
            
            proc.wait()
            t_out.join()
            t_err.join()
            
            tracker.stop()
            tracker.join()
            
            # Compute timings
            if t_prefill_start and t_prefill_end:
                t_prefill = t_prefill_end - t_prefill_start
            else:
                t_prefill = 0.0
                
            if decode_step_times:
                avg_step_ms = sum(decode_step_times) / len(decode_step_times)
                tps = 1000.0 / avg_step_ms
            else:
                tps = 0.0
                
            merged_stdout = "".join(stdout_lines)
            response = ""
            if "[Response]" in merged_stdout:
                response = merged_stdout.split("[Response]")[1].strip()
                
            accuracy = 1.0 if "847291" in response else 0.0
            
            res = {
                "prefill_s": t_prefill,
                "decode_tps": tps,
                "peak_rss_mb": tracker.peak_rss,
                "accuracy": accuracy,
                "response": response
            }
            
        elif mode == "active":
            import mlx.core as mx
            # Add ACTIVE_RUNTIME to path
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))
            import torch
            from serving.hf_diffkv_wrapper import DiffKVHFWrapper
            
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
                        "response": response_text
                    }
            
            # Configure exact same rank=32 and block_size=64
            config = {
                "quantization": "int4",
                "rank": 32,
                "block_size": 64,
                "micro_block_size": 64,
            }
            
            wrapper = BenchmarkedDiffKVWrapper(
                model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                config=config,
                device="mps",
            )
            
            res_bench = wrapper.generate_benchmark_ids(prompt_ids, max_new_tokens=64)
            wrapper.stop()
            
            tracker.stop()
            tracker.join()
            
            accuracy = 1.0 if "847291" in res_bench["response"] else 0.0
            
            res = {
                "prefill_s": res_bench["prefill_s"],
                "decode_tps": res_bench["decode_tps"],
                "peak_rss_mb": tracker.peak_rss,
                "accuracy": accuracy,
                "response": res_bench["response"]
            }
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        print(json.dumps(res))
        
    except Exception as e:
        tracker.stop()
        try:
            tracker.join()
        except Exception:
            pass
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))

# Parent Orchestrator
def run_subprocess_run(mode, context_len):
    cmd = [
        sys.executable,
        __file__,
        "--run-single",
        "--mode", mode,
        "--context", str(context_len)
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=250,
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
        
    contexts = [2048, 4096, 8192]
    modes = ["native", "active"]
    
    labels = {
        "native": "diffkv_native (C++)",
        "active": "ACTIVE_RUNTIME (Python MLX)"
    }
    
    colors = {
        "native": "#0984e3",  # C++ blue
        "active": "#e056fd"   # Python purple
    }
    
    markers = {
        "native": "o",
        "active": "s"
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
        
        for mode in modes:
            x_vals = []
            y_vals = []
            for c in contexts:
                val = data.get(mode, {}).get(str(c), {}).get(metric)
                if val is not None and not isinstance(val, str):
                    x_vals.append(c)
                    y_vals.append(val)
            
            if x_vals:
                plt.plot(
                    x_vals,
                    y_vals,
                    marker=markers[mode],
                    linestyle="-",
                    color=colors[mode],
                    label=labels[mode],
                    linewidth=2.0,
                    markersize=8
                )
                    
        plt.title(titles[metric], fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Context Length (tokens)', fontsize=12)
        plt.ylabel(y_labels[metric], fontsize=12)
        plt.xticks(contexts, ["2K", "4K", "8K"])
        
        if metric == "accuracy":
            plt.ylim(-0.1, 1.1)
        elif metric == "decode_tps":
            plt.ylim(0, None)
            
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'compare_runtime_{metric}.png'), dpi=200)
        plt.close()
        
    print("Comparative runtime plots generated successfully.")

# Main coordinator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--mode", choices=["native", "active"])
    parser.add_argument("--context", type=int)
    args = parser.parse_args()
    
    if args.run_single:
        run_single_benchmark(args.mode, args.context)
        sys.exit(0)
        
    print("=" * 80)
    print("      LAUNCHING NATIVE C++ VS PYTHON MLX COMPARISON BENCHMARK SUITE")
    print("=" * 80)
    
    contexts = [2048, 4096, 8192]
    modes = ["native", "active"]
    
    results = {
        "native": {},
        "active": {}
    }
    
    for mode in modes:
        for c in contexts:
            print(f"\n>>> Running: Mode={mode} | Context={c} ...")
            res = run_subprocess_run(mode, c)
            print(f"Result: {res}")
            results[mode][c] = res
                
    # Save results
    results_path = "compare_native_active_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")
    
    # Generate plots
    artifact_dir = "/Users/omchimurkar1/.gemini/antigravity/brain/ada31170-301d-45cf-bbdf-321c6b861dbc"
    generate_plots(results_path, artifact_dir)
    generate_plots(results_path, "benchmark_plots")
