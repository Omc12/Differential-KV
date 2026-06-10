import os
import sys
import time
import json
import subprocess
import asyncio
import urllib.request
import psutil

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_ollama_runner():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any('ollama runner' in ' '.join(cmd) or 'ollama' in part for part in cmd) and any('runner' in part for part in cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def find_gateway_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any('openai_compatible_api_gateway.py' in part for part in cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def get_process_memory_info(proc):
    if proc is None:
        return 0.0, 0.0
    try:
        mem = proc.memory_info()
        return mem.rss / (1024**2), mem.vms / (1024**2)
    except Exception:
        return 0.0, 0.0

async def query_ollama(prompt, length, num_tokens=30):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "qwen2.5:1.5b-instruct-fp16",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_tokens,
            "temperature": 0.0,
            "num_ctx": max(2048, length + 100)  # Make sure Ollama has enough context space
        }
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    
    # Run in executor to avoid blocking asyncio loop
    loop = asyncio.get_event_loop()
    def _send():
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            return {"error": str(e)}
            
    return await loop.run_in_executor(None, _send)

async def query_diffkv(prompt, num_tokens=30):
    url = "http://localhost:8000/v1/chat/completions"
    data = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_tokens,
        "temperature": 0.0
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    
    loop = asyncio.get_event_loop()
    def _send():
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            return {"error": str(e)}
            
    return await loop.run_in_executor(None, _send)

def get_diffkv_runtime_info():
    url = "http://localhost:8000/v1/runtime_info"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=1.0) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return {}

async def run_ollama_benchmark(lengths):
    print("\n=== BENCHMARKING OLLAMA ===")
    results = {}
    
    # Warm up Ollama
    print("Warming up Ollama with short request...")
    await query_ollama("Warm up prompt", 10)
    time.sleep(2.0)
    
    runner = find_ollama_runner()
    if runner is None:
        print("Error: Could not find Ollama runner process!")
        return {}
        
    print(f"Found Ollama runner PID: {runner.pid}")
    
    # Baseline idle memory
    base_rss, _ = get_process_memory_info(runner)
    print(f"Ollama Idle Memory: {base_rss:.1f} MB")
    
    for length in lengths:
        print(f"\nTesting context length: {length} tokens...")
        # Create a prompt of approximately `length` tokens (each word is roughly 1.3 tokens)
        word_count = int(length / 1.3)
        dummy_prompt = "hello " * word_count
        
        # Track memory concurrently during the query
        rss_samples = []
        
        async def monitor():
            while True:
                rss, _ = get_process_memory_info(runner)
                if rss > 0:
                    rss_samples.append(rss)
                await asyncio.sleep(0.1)
                
        monitor_task = asyncio.create_task(monitor())
        
        start_time = time.time()
        res = await query_ollama(dummy_prompt, length)
        elapsed = time.time() - start_time
        
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
            
        if "error" in res:
            print(f"Ollama Error at {length} tokens: {res['error']}")
            results[length] = {"error": res['error']}
        else:
            peak_rss = max(rss_samples) if rss_samples else base_rss
            delta_rss = peak_rss - base_rss
            print(f"Completed in {elapsed:.2f}s | Peak RSS: {peak_rss:.1f} MB | Delta: {delta_rss:.1f} MB")
            results[length] = {
                "peak_rss_mb": peak_rss,
                "delta_rss_mb": delta_rss,
                "time_s": elapsed
            }
            
        # Let Ollama settle
        time.sleep(3.0)

    # Force unload model from Ollama
    print("Unloading Ollama model...")
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "qwen2.5:1.5b-instruct-fp16",
            "prompt": "",
            "keep_alive": 0
        }
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode('utf-8'), 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
        print("Successfully sent unload request to Ollama.")
    except Exception as e:
        print(f"Warning: Failed to send unload request to Ollama: {e}")

    # Terminate runner if still present
    runner = find_ollama_runner()
    if runner is not None:
        try:
            runner.terminate()
            runner.wait(timeout=5)
            print("Terminated lingering Ollama runner process.")
        except Exception:
            pass
        
    return results

async def run_diffkv_benchmark(lengths):
    print("\n=== BENCHMARKING DIFFKV ===")
    results = {}
    
    for length in lengths:
        print(f"\nTesting context length: {length} tokens...")
        
        # Start DiffKV server in background
        print("Starting DiffKV API Gateway server...")
        cmd = [
            "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_venv/bin/python3",
            "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py",
            "--model", "Qwen/Qwen2.5-1.5B-Instruct",
            "--preset", "low",
            "--rank", "16"
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME"
        env["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
        env["DIFFKV_USE_TORCH_COMPILE"] = "0"
        
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gateway_log.txt")
        log_file = open(log_path, "w")
        proc_srv = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for server to start
        gateway_proc = None
        for i in range(120):
            time.sleep(1.0)
            gateway_proc = find_gateway_process()
            if gateway_proc is not None:
                # Check if port 8000 is responding
                info = get_diffkv_runtime_info()
                if info:
                    print(f"DiffKV server started on PID: {gateway_proc.pid}")
                    break
        else:
            print("Error starting DiffKV server!")
            proc_srv.terminate()
            log_file.close()
            continue
            
        try:
            # Baseline idle memory
            base_rss, _ = get_process_memory_info(gateway_proc)
            info = get_diffkv_runtime_info()
            base_mps_driver = info.get("mps_driver_gb", 0.0) * 1024.0
            print(f"DiffKV Idle CPU RSS: {base_rss:.1f} MB | MPS Driver: {base_mps_driver:.1f} MB")
            
            word_count = int(length / 1.3)
            dummy_prompt = "hello " * word_count
            
            # Track memory concurrently
            rss_samples = []
            mps_samples = []
            
            async def monitor():
                while True:
                    rss, _ = get_process_memory_info(gateway_proc)
                    if rss > 0:
                        rss_samples.append(rss)
                    info = get_diffkv_runtime_info()
                    mps_driver = info.get("mps_driver_gb", 0.0) * 1024.0
                    if mps_driver > 0:
                        mps_samples.append(mps_driver)
                    await asyncio.sleep(0.1)
                    
            monitor_task = asyncio.create_task(monitor())
            
            start_time = time.time()
            res = await query_diffkv(dummy_prompt)
            elapsed = time.time() - start_time
            
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
                
            if "error" in res:
                print(f"DiffKV Error at {length} tokens: {res['error']}")
                results[length] = {"error": res['error']}
            else:
                peak_rss = max(rss_samples) if rss_samples else base_rss
                peak_mps = max(mps_samples) if mps_samples else base_mps_driver
                
                # In macOS, the footprint is CPU RSS + MPS Driver.
                total_footprint = peak_rss + peak_mps
                delta_footprint = total_footprint - (base_rss + base_mps_driver)
                
                print(f"Completed in {elapsed:.2f}s | Peak VRAM/MPS: {peak_mps:.1f} MB | CPU RSS: {peak_rss:.1f} MB | Total: {total_footprint:.1f} MB")
                results[length] = {
                    "peak_rss_mb": peak_rss,
                    "peak_mps_mb": peak_mps,
                    "total_footprint_mb": total_footprint,
                    "delta_footprint_mb": delta_footprint,
                    "time_s": elapsed
                }
                
        finally:
            # Kill server
            print("Shutting down DiffKV server...")
            proc_srv.terminate()
            proc_srv.wait()
            try:
                log_file.close()
            except Exception:
                pass
            time.sleep(3.0)
        
    return results

async def main():
    lengths = [2000, 4000, 8000]
    
    # Run Ollama
    ollama_res = await run_ollama_benchmark(lengths)
    
    # Run DiffKV
    diffkv_res = await run_diffkv_benchmark(lengths)
    
    # Print comparison table
    print("\n" + "="*80)
    print("                 REAL-WORLD BENCHMARK RESULTS (Qwen-2.5 1.5B FP16)")
    print("="*80)
    print(f"{'Length':<10} | {'Ollama Peak RAM':<18} | {'DiffKV VRAM/MPS':<18} | {'DiffKV CPU RSS':<15} | {'DiffKV Total'}")
    print("-"*80)
    for length in lengths:
        o = ollama_res.get(length, {})
        d = diffkv_res.get(length, {})
        
        if "error" in o or not o:
            o_ram = "Error/N/A"
        else:
            o_ram = f"{o['peak_rss_mb']:.1f} MB"
            
        if "error" in d or not d:
            d_mps = "Error/N/A"
            d_rss = "N/A"
            d_tot = "N/A"
        else:
            d_mps = f"{d['peak_mps_mb']:.1f} MB"
            d_rss = f"{d['peak_rss_mb']:.1f} MB"
            d_tot = f"{d['total_footprint_mb']:.1f} MB"
            
        print(f"{length:<10} | {o_ram:<18} | {d_mps:<18} | {d_rss:<15} | {d_tot}")
    print("="*80)
    
    # Print Delta scaling table
    print("\n" + "="*80)
    print("                 KV CACHE SCALING SCENARIO (DELTA MEMORY GROWTH)")
    print("="*80)
    print(f"{'Length':<10} | {'Ollama Delta RAM':<20} | {'DiffKV Delta Memory':<25} | {'KV Cache VRAM Saved'}")
    print("-"*80)
    for length in lengths:
        o = ollama_res.get(length, {})
        d = diffkv_res.get(length, {})
        
        if "error" in o or not o:
            o_delta = "N/A"
            o_val = 0.0
        else:
            o_delta = f"+{o['delta_rss_mb']:.1f} MB"
            o_val = o['delta_rss_mb']
            
        if "error" in d or not d:
            d_delta = "N/A"
            d_val = 0.0
        else:
            d_delta = f"+{d['delta_footprint_mb']:.1f} MB"
            d_val = d['delta_footprint_mb']
            
        saved = o_val - d_val
        saved_str = f"{saved:.1f} MB" if saved > 0 else "0.0 MB"
        
        print(f"{length:<10} | {o_delta:<20} | {d_delta:<25} | {saved_str}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
