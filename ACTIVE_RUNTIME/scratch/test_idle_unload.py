import os
import sys
import time
import json
import subprocess
import urllib.request

def get_runtime_info():
    try:
        url = "http://localhost:8000/v1/runtime_info"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=1.0) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

def query_chat():
    try:
        url = "http://localhost:8000/v1/chat/completions"
        data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 10,
            "temperature": 0.0
        }
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode('utf-8'), 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30.0) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

def main():
    print("="*60)
    print("  TESTING AUTOMATIC IDLE MODEL UNLOADING & ON-DEMAND RELOAD")
    print("="*60)
    
    # Start server in background with a 5-second idle timeout
    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME"
    env["DIFFKV_MODEL_IDLE_TIMEOUT"] = "5"
    env["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
    env["DIFFKV_USE_TORCH_COMPILE"] = "0"
    
    cmd = [
        "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_venv/bin/python3",
        "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py",
        "--preset", "low",
        "--rank", "16"
    ]
    
    print("Launching server with DIFFKV_MODEL_IDLE_TIMEOUT=5...")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to be responsive
    for i in range(30):
        time.sleep(1.0)
        info = get_runtime_info()
        if "error" not in info:
            print("Server is responsive!")
            break
    else:
        print("Error: Server did not become responsive!")
        proc.terminate()
        sys.exit(1)
        
    try:
        # Phase 1: Loaded state immediately after startup
        info = get_runtime_info()
        mps_alloc = info.get("mps_allocated_gb", 0.0) * 1024.0
        mps_driver = info.get("mps_driver_gb", 0.0) * 1024.0
        print(f"\n[Phase 1] Loaded Startup Memory:")
        print(f"  PyTorch MPS Allocated: {mps_alloc:.1f} MB")
        print(f"  Metal Driver Resident: {mps_driver:.1f} MB")
        
        # Verify that weights are loaded (>0 MB allocated)
        if mps_alloc < 100:
            print("WARNING: Expected model weights to be loaded initially, but allocation is low!")
            
        # Phase 2: Sleep and let idle timeout fire
        sleep_time = 8.0
        print(f"\n[Phase 2] Sleeping for {sleep_time} seconds to let idle timeout trigger...")
        time.sleep(sleep_time)
        
        info = get_runtime_info()
        mps_alloc_idle = info.get("mps_allocated_gb", 0.0) * 1024.0
        mps_driver_idle = info.get("mps_driver_gb", 0.0) * 1024.0
        print(f"Idle Memory after sleep:")
        print(f"  PyTorch MPS Allocated: {mps_alloc_idle:.1f} MB")
        print(f"  Metal Driver Resident: {mps_driver_idle:.1f} MB")
        
        # Verify that VRAM has been released (MPS allocated drops to 0)
        if mps_alloc_idle == 0.0:
            print("SUCCESS: Model weights successfully unloaded from PyTorch memory!")
        else:
            print("ERROR: Model weights still present in memory after idle period!")
            
        # Phase 3: Send query to trigger reload
        print("\n[Phase 3] Sending request to trigger auto-reload...")
        start_t = time.time()
        res = query_chat()
        elapsed = time.time() - start_t
        
        if "error" in res:
            print(f"Query Error during reload: {res['error']}")
        else:
            ans = res["choices"][0]["message"]["content"]
            print(f"Query Succeeded in {elapsed:.2f}s!")
            print(f"Response: {repr(ans)}")
            
        # Check memory again after reload
        info = get_runtime_info()
        mps_alloc_reloaded = info.get("mps_allocated_gb", 0.0) * 1024.0
        mps_driver_reloaded = info.get("mps_driver_gb", 0.0) * 1024.0
        print(f"\nMemory after reloading:")
        print(f"  PyTorch MPS Allocated: {mps_alloc_reloaded:.1f} MB")
        print(f"  Metal Driver Resident: {mps_driver_reloaded:.1f} MB")
        
        if mps_alloc_reloaded > 100:
            print("SUCCESS: Model weights successfully reloaded back into memory!")
        else:
            print("ERROR: Model weights not active after query!")
            
        # Phase 4: Wait for it to unload again
        print(f"\n[Phase 4] Sleeping for another {sleep_time} seconds to verify second unload...")
        time.sleep(sleep_time)
        
        info = get_runtime_info()
        mps_alloc_idle2 = info.get("mps_allocated_gb", 0.0) * 1024.0
        print(f"Memory after second idle period:")
        print(f"  PyTorch MPS Allocated: {mps_alloc_idle2:.1f} MB")
        
        if mps_alloc_idle2 == 0.0:
            print("SUCCESS: Model weights unloaded again on second idle period!")
        else:
            print("ERROR: Failed to unload model weights on second idle period!")
            
    finally:
        print("\nTerminating background server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
