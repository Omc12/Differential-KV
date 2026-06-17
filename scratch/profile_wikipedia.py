import os
import sys
import time
import json
import subprocess
import threading
import psutil

class MemoryTracker(threading.Thread):
    def __init__(self, pid, interval=0.05):
        super().__init__()
        self.pid = pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0.0

    def run(self):
        try:
            p = psutil.Process(self.pid)
            while not self.stop_event.is_set():
                if not p.is_running() or p.status() == "zombie":
                    break
                total = p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass
                rss_mb = total / (1024 * 1024)
                if rss_mb > self.peak_rss:
                    self.peak_rss = rss_mb
                time.sleep(self.interval)
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()

def run_cpp_benchmark(model_path, prompt, max_tokens=32, preset="low"):
    print(f"\n--- [C++ Native Benchmark] Model={model_path} ---")
    binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
    
    env = os.environ.copy()
    env["DIFFKV_USE_GPU"] = "1"
    env["DIFFKV_VERBOSE"] = "1"
    env["DIFFKV_MAX_TOKENS"] = str(max_tokens)
    env["DIFFKV_PRESET"] = preset
    env["max_ctx_tk"] = "16384"
    env["DIFFKV_TIME_DECODE"] = "1"
    
    # We pass the prompt via argv[2]
    cmd = [binary_path, model_path, prompt]
    
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    tracker = MemoryTracker(proc.pid)
    tracker.start()
    
    t_prefill_start = None
    t_prefill_end = None
    decode_step_times = []
    
    # We set a timeout of 120 seconds to prevent hanging
    stdout_lines = []
    def read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)
            
    def read_stderr():
        nonlocal t_prefill_start, t_prefill_end
        for line in proc.stderr:
            # Print stderr in real-time to see progress
            print(f"  [stderr] {line.strip()}")
            if "Running Prefill phase in chunks" in line:
                t_prefill_start = time.perf_counter()
            elif "Prefill Phase Top predictions" in line:
                t_prefill_end = time.perf_counter()
            elif "Timing Step" in line:
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
    
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("[!] C++ process timed out after 120 seconds!")
        proc.terminate()
        proc.wait()
        
    tracker.stop()
    tracker.join()
    t_out.join()
    t_err.join()
    
    total_time = time.time() - start_time
    prefill_time = (t_prefill_end - t_prefill_start) if (t_prefill_start and t_prefill_end) else None
    
    avg_step_ms = sum(decode_step_times) / len(decode_step_times) if decode_step_times else None
    decode_tps = (1000.0 / avg_step_ms) if avg_step_ms else None
    
    print(f"C++ Peak Memory: {tracker.peak_rss:.2f} MB")
    print(f"C++ Total Time:   {total_time:.2f} s")
    if prefill_time:
        print(f"C++ Prefill Time: {prefill_time:.2f} s")
    if decode_tps:
        print(f"C++ Decode TPS:   {decode_tps:.2f}")
    return {
        "peak_rss": tracker.peak_rss,
        "total_time": total_time,
        "prefill_time": prefill_time,
        "decode_tps": decode_tps,
        "response": "".join(stdout_lines)
    }

def run_python_benchmark(model_id, prompt, max_tokens=32, preset="low"):
    print(f"\n--- [Python MLX Benchmark] Model={model_id} ---")
    
    # We will write a tiny runner script to isolate the Python subprocess
    runner_code = f"""
import os
import sys
import time
import gc
import numpy as np
import torch
import mlx.core as mx

# Add ACTIVE_RUNTIME to path
sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

with open("scratch/wiki_prompt.txt", "r") as f:
    prompt_text = f.read()

config = {{
    "quantization": "int4" if "{model_id}".endswith("4bit") else None,
    "rank": 16 if "{preset}" == "low" else 32,
    "block_size": 256,
    "micro_block_size": 256,
    "preset": "{preset}",
    "serving_mode": "lightweight" if "{preset}" == "low" else "balanced"
}}

print("Loading model...", flush=True)
wrapper = DiffKVHFWrapper(
    "{model_id}",
    config=config,
    device="mps",
)

wrapper.ensure_loaded()
prompt_ids = wrapper.tokenizer.encode(prompt_text, add_special_tokens=False)
print("Prefilling", len(prompt_ids), "tokens...", flush=True)

session_id = "bench_session"
wrapper.manager.clear_session(session_id)
wrapper.manager.init_session(session_id, prefill_len=len(prompt_ids))
wrapper.manager.register_prefill_tokens(session_id, torch.tensor(prompt_ids, dtype=torch.long))
wrapper.model._diffkv_session_ids = [session_id]

PREFILL_CHUNK = 512
output = None
mx.eval()
t0 = time.perf_counter()

for chunk_start in range(0, len(prompt_ids), PREFILL_CHUNK):
    chunk = prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
    clen = len(chunk)
    abs_start = chunk_start
    chunk_tensor = torch.tensor([chunk], dtype=torch.long)
    pos_tensor = torch.tensor([list(range(abs_start, abs_start + clen))], dtype=torch.long)
    output = wrapper.model(chunk_tensor, pos_tensor)
    wrapper.manager.compress_deferred_prefill_blocks(session_id)

mx.eval()
t_prefill = time.perf_counter() - t0
print(f"Prefill done in {{t_prefill:.3f}} seconds", flush=True)

# Decode phase
generated = []
cur_pos = len(prompt_ids)
logits = output.logits[0, -1].cpu().numpy()

t1 = time.perf_counter()
for step_idx in range({max_tokens}):
    next_id = int(np.argmax(logits))
    generated.append(next_id)
    wrapper.manager.register_prefill_tokens(session_id, torch.tensor([next_id], dtype=torch.long))
    input_ids = torch.tensor([[next_id]], dtype=torch.long)
    pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long)
    output = wrapper.model(input_ids, pos_tensor)
    logits = output.logits[0, -1].cpu().numpy()
    cur_pos += 1

mx.eval()
t_decode = time.perf_counter() - t1
tps = {max_tokens} / max(t_decode, 0.001)
print(f"Decode TPS: {{tps:.2f}}", flush=True)
decoded_text = wrapper.tokenizer.decode(generated)
print(f"PYTHON_DECODED: {{decoded_text}}", flush=True)
"""
    
    with open("scratch/run_py_bench_temp.py", "w") as f:
        f.write(runner_code)
        
    cmd = ["./diffkv_venv/bin/python3", "scratch/run_py_bench_temp.py"]
    
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    tracker = MemoryTracker(proc.pid)
    tracker.start()
    
    prefill_time = None
    decode_tps = None
    
    stdout_lines = []
    for line in proc.stdout:
        print(f"  [stdout] {line.strip()}")
        stdout_lines.append(line)
        if "Prefill done in" in line:
            try:
                prefill_time = float(line.split("Prefill done in ")[1].split(" seconds")[0])
            except Exception:
                pass
        elif "Decode TPS:" in line:
            try:
                decode_tps = float(line.split("Decode TPS: ")[1])
            except Exception:
                pass
                
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("[!] Python process timed out!")
        proc.terminate()
        proc.wait()
        
    tracker.stop()
    tracker.join()
    
    total_time = time.time() - start_time
    print(f"Python Peak Memory: {tracker.peak_rss:.2f} MB")
    print(f"Python Total Time:   {total_time:.2f} s")
    
    # cleanup temp script
    if os.path.exists("scratch/run_py_bench_temp.py"):
        os.remove("scratch/run_py_bench_temp.py")
        
    py_response = ""
    for line in stdout_lines:
        if "PYTHON_DECODED:" in line:
            py_response = line.split("PYTHON_DECODED:")[1].strip()

    return {
        "peak_rss": tracker.peak_rss,
        "total_time": total_time,
        "prefill_time": prefill_time,
        "decode_tps": decode_tps,
        "response": py_response
    }

if __name__ == "__main__":
    with open("scratch/wiki_prompt.txt", "r") as f:
        prompt = f.read()
        
    results = {}
    
    # We will test:
    # 1. Python MLX Qwen 2.5 1.5B Instruct 4bit
    results["py_1.5b_4bit"] = run_python_benchmark("mlx-community/Qwen2.5-1.5B-Instruct-4bit", prompt, max_tokens=32, preset="low")
    
    # 2. C++ Native Qwen 2.5 1.5B Instruct Q8_0
    model_1_5b = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
    results["cpp_1.5b_q8_0"] = run_cpp_benchmark(model_1_5b, prompt, max_tokens=32, preset="low")
    
    print("\n" + "="*50)
    print("           BENCHMARK RESULTS (10,712 tokens)")
    print("="*50)
    print(f"{'Metric':<25} | {'Python 1.5B 4-bit':<20} | {'C++ 1.5B Q8_0':<20}")
    print("-" * 71)
    
    py = results["py_1.5b_4bit"]
    cpp = results["cpp_1.5b_q8_0"]
    
    py_mem = f"{py['peak_rss']:.2f} MB"
    cpp_mem = f"{cpp['peak_rss']:.2f} MB"
    
    py_pref = f"{py.get('prefill_time'):.2f} s" if py.get('prefill_time') is not None else 'N/A'
    cpp_pref = f"{cpp.get('prefill_time'):.2f} s" if cpp.get('prefill_time') is not None else 'N/A'
    
    py_dec = f"{py.get('decode_tps'):.2f} tps" if py.get('decode_tps') is not None else 'N/A'
    cpp_dec = f"{cpp.get('decode_tps'):.2f} tps" if cpp.get('decode_tps') is not None else 'N/A'
    
    py_tot = f"{py['total_time']:.2f} s"
    cpp_tot = f"{cpp['total_time']:.2f} s"
    
    print(f"{'Peak Memory (RSS)':<25} | {py_mem:<20} | {cpp_mem:<20}")
    print(f"{'Prefill Time':<25} | {py_pref:<20} | {cpp_pref:<20}")
    print(f"{'Decode Throughput (TPS)':<25} | {py_dec:<20} | {cpp_dec:<20}")
    print(f"{'Total Execution Time':<25} | {py_tot:<20} | {cpp_tot:<20}")
    print("="*50)
    
    print("\n[Python 1.5B Response]:")
    print(py.get("response"))
    print("\n[C++ 1.5B Response]:")
    print(cpp.get("response"))
    print("="*50)
