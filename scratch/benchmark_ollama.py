import os
import sys
import time
import json
import urllib.request
import threading
import psutil

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_FP16 = "qwen2.5:0.5b-instruct-fp16"
MODEL_QUANT = "qwen2.5:0.5b-instruct"

# ── Process Finder & Memory Tracker ───────────────────────────────────────────

def find_ollama_runner():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any('ollama' in part.lower() for part in cmd) and any('serve' in part.lower() for part in cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

class MemoryTracker(threading.Thread):
    def __init__(self, process, interval=0.01):
        super().__init__()
        self.process = process
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                if self.process.is_running():
                    mem = self.process.memory_info()
                    rss = mem.rss / 1e6
                    if rss > self.peak_rss:
                        self.peak_rss = rss
                else:
                    break
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# ── NIAH Prompt Builder ────────────────────────────────────────────────────────

def make_niah_prompt(context_length, depth, needle, question):
    # Simulating tokenizer token counting (roughly 4 characters per token)
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    char_len_per_token = 4
    needle_len_tokens = len(needle) // char_len_per_token
    question_len_tokens = len(question) // char_len_per_token
    
    target_filler_tokens = context_length - needle_len_tokens - question_len_tokens - 100
    if target_filler_tokens < 0:
        target_filler_tokens = 100
        
    filler_char_len = target_filler_tokens * char_len_per_token
    num_repeats = (filler_char_len // len(filler)) + 1
    all_filler = (filler * num_repeats)[:filler_char_len]
    
    insert_idx = int(len(all_filler) * depth)
    part1_text = all_filler[:insert_idx]
    part2_text = all_filler[insert_idx:]
    
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

# ── Ollama HTTP Client ────────────────────────────────────────────────────────

def query_ollama(model_name, prompt, length, max_tokens=64):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
            "num_ctx": max(2048, length + 200)  # Allocate context space in Ollama
        }
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

def unload_model(model_name):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": "",
        "keep_alive": 0
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
    except Exception:
        pass

# ── Main Run Logic ────────────────────────────────────────────────────────────

def run_model_benchmark(model_name, contexts):
    print(f"\n--- Benchmarking Ollama model: {model_name} ---")
    results = {}
    
    # 1. Warmup to load the model
    print("Warming up model to trigger runner initialization...")
    warmup_res = query_ollama(model_name, "Warm up prompt", 128, max_tokens=1)
    if "error" in warmup_res:
        print(f"  --> Failed to warm up model: {warmup_res['error']}")
        return {"error": warmup_res["error"]}
        
    time.sleep(2.0)
    
    # 2. Find Ollama runner process
    runner = find_ollama_runner()
    if runner is None:
        print("  --> Warning: Could not locate Ollama runner process. Benchmarking without RAM tracking.")
    else:
        print(f"  --> Found Ollama runner process PID: {runner.pid}")
        
    for c in contexts:
        print(f"Running Ctx: {c}...")
        needle = "The special code is 847291."
        question = "What is the special code? Answer in exactly the 6-digit code number."
        
        prompt = make_niah_prompt(c, 0.5, needle, question)
        
        # Start memory tracking
        tracker = None
        if runner is not None:
            tracker = MemoryTracker(runner)
            tracker.start()
            
        t_start = time.perf_counter()
        res = query_ollama(model_name, prompt, c, max_tokens=64)
        t_elapsed = time.perf_counter() - t_start
        
        if tracker is not None:
            tracker.stop()
            tracker.join()
            peak_rss = tracker.peak_rss
        else:
            peak_rss = 0.0
            
        if "error" in res:
            print(f"  --> Skip/OOM/Error: {res['error']}")
            results[str(c)] = {"error": res["error"]}
        else:
            # Extract internal Ollama nanosecond metrics
            # prompt_eval_duration = prefill time
            # eval_duration = decode time
            # eval_count = decoded tokens
            prompt_eval_ns = res.get("prompt_eval_duration", 0.0)
            eval_ns = res.get("eval_duration", 0.0)
            eval_count = res.get("eval_count", 1.0)
            
            prefill_s = prompt_eval_ns / 1e9
            decode_tps = eval_count / (max(1.0, eval_ns) / 1e9)
            
            response = res.get("response", "").strip()
            accuracy = 1.0 if "847291" in response else 0.0
            
            # If Ollama didn't return nanosecond metrics (older versions), fallback to wall-clock time
            if prompt_eval_ns == 0.0 or eval_ns == 0.0:
                prefill_s = t_elapsed
                decode_tps = 64.0 / t_elapsed
                
            print(f"  --> Done: TTFT={prefill_s:.3f}s | TPS={decode_tps:.1f} | PeakRSS={peak_rss:.1f}MB | Acc={accuracy:.1f} | Resp={repr(response[:40])}")
            
            results[str(c)] = {
                "prefill_s": prefill_s,
                "decode_tps": decode_tps,
                "peak_rss_mb": peak_rss,
                "accuracy": accuracy,
                "response": response
            }
            
        time.sleep(2.0)
        
    print(f"Unloading model {model_name}...")
    unload_model(model_name)
    time.sleep(3.0)
    
    return results

if __name__ == "__main__":
    contexts = [1024, 2048, 4096, 8192, 16384]
    
    # 1. Run FP16 Model
    fp16_results = run_model_benchmark(MODEL_FP16, contexts)
    
    # 2. Run Quantized Model
    quant_results = run_model_benchmark(MODEL_QUANT, contexts)
    
    # 3. Load previous results to merge
    custom_results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results_custom.json")
    if os.path.exists(custom_results_path):
        with open(custom_results_path, "r") as f:
            custom_data = json.load(f)
    else:
        custom_data = {"standard": {}, "diffkv": {}}
        
    combined_results = {
        "dense_pytorch": custom_data.get("standard", {}),
        "diffkv_mlx": custom_data.get("diffkv", {}),
        "ollama_fp16": fp16_results,
        "ollama_quant": quant_results
    }
    
    # Save combined results
    out_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results_ollama.json")
    with open(out_json, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nConsolidated results saved to {out_json}")
