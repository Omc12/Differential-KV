import time
import json
import os
import sys
import psutil
import urllib.request

def find_gateway_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any('openai_compatible_api_gateway.py' in part for part in cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def main():
    print("\033[H\033[J", end="") # Clear screen
    print("=========================================================")
    print("      DIFFERENTIAL KV LIVE MEMORY MONITOR & TRACKER      ")
    print("=========================================================")
    print("Waiting for gateway process to start...")
    
    proc = None
    for _ in range(30): # wait up to 15 seconds
        proc = find_gateway_process()
        if proc is not None:
            break
        time.sleep(0.5)
        
    if proc is None:
        print("Error: Gateway process (openai_compatible_api_gateway.py) not found!")
        print("Please launch the gateway script first, or make sure it is running.")
        sys.exit(1)
        
    pid = proc.pid
    print(f"Found process PID: {pid}")
    print("Connecting to /v1/runtime_info...")
    
    log_file = "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/scratch/memory_log.json"
    history = []
    
    # We run until the script is interrupted or process exits
    start_time = time.time()
    
    try:
        while True:
            if not proc.is_running():
                print("\n[Monitor] Process terminated.")
                break
                
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Query memory info using psutil
            try:
                mem_info = proc.memory_info()
                rss_gb = mem_info.rss / 1e9
                vms_gb = mem_info.vms / 1e9
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("\n[Monitor] Process lost.")
                break
                
            # Query runtime info from the gateway
            runtime_info = {}
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/v1/runtime_info")
                with urllib.request.urlopen(req, timeout=0.3) as response:
                    runtime_info = json.loads(response.read().decode())
            except Exception:
                pass
                
            entry = {
                "timestamp": elapsed,
                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rss_gb": round(rss_gb, 4),
                "vms_gb": round(vms_gb, 4),
                "runtime_info": runtime_info
            }
            history.append(entry)
            
            # Save history to file
            with open(log_file, "w") as f:
                json.dump(history, f, indent=2)
                
            # Print beautiful live dashboard
            print("\033[H", end="") # Move cursor to top
            print("=========================================================")
            print("      DIFFERENTIAL KV LIVE MEMORY MONITOR & TRACKER      ")
            print("=========================================================")
            print(f"  PID:              {pid:<10}  |  Elapsed Time: {elapsed:.1f}s")
            print(f"  System RAM (RSS): {rss_gb:6.3f} GB    |  Virtual RAM (VMS): {vms_gb:6.3f} GB")
            print("---------------------------------------------------------")
            
            if runtime_info:
                device = runtime_info.get("device", "unknown")
                serving_mode = runtime_info.get("serving_mode", "unknown")
                model = runtime_info.get("model", "unknown")
                
                print(f"  Model Loaded:     {model}")
                print(f"  Device:           {device:<10}  |  Serving Mode:      {serving_mode}")
                print("---------------------------------------------------------")
                
                # Retrieve KV Cache metadata
                kv_sum = runtime_info.get("kv_summary", {})
                pager = kv_sum.get("pager", {}) if isinstance(kv_sum, dict) else {}
                kv_cache_resident_gb = 0.0
                if pager and isinstance(pager, dict):
                    kv_cache_resident_gb = pager.get("gpu_resident_mb", 0.0) / 1024.0

                # Determine model weights baseline size (Qwen 0.5B in FP16 is ~0.94 GB)
                est_weights_gb = 0.94 if "0.5B" in model else (3.1 if "1.5B" in model else 1.5)
                
                if device == "mps":
                    mps_allocated = runtime_info.get("mps_allocated_gb", 0.0)
                    mps_driver = runtime_info.get("mps_driver_gb", 0.0)
                    python_overhead = max(0.0, rss_gb - mps_allocated)
                    
                    # Estimate intermediate activation memory
                    est_activations_gb = max(0.0, mps_allocated - est_weights_gb - kv_cache_resident_gb)
                    
                    print(f"  [RAM Breakdown]")
                    print(f"    ├─ Python CPU Overhead: {python_overhead:6.3f} GB  (Interpreter, libraries)")
                    print(f"    ├─ MPS Allocated VRAM:  {mps_allocated:6.3f} GB  (Active GPU tensors)")
                    print(f"    │   ├─ Model Weights:   {min(est_weights_gb, mps_allocated):6.3f} GB  (Estimated weights)")
                    print(f"    │   ├─ KV Cache VRAM:   {kv_cache_resident_gb:6.3f} GB  (Hot block residency)")
                    print(f"    │   └─ Activations/Tmp: {est_activations_gb:6.3f} GB  (Intermediates, context)")
                    print(f"    └─ MPS Driver Total:    {mps_driver:6.3f} GB  (Unified memory reservation)")
                    
                elif device == "cuda":
                    cuda_allocated = runtime_info.get("cuda_allocated_gb", 0.0)
                    cuda_reserved = runtime_info.get("cuda_reserved_gb", 0.0)
                    python_overhead = max(0.0, rss_gb - cuda_allocated)
                    est_activations_gb = max(0.0, cuda_allocated - est_weights_gb - kv_cache_resident_gb)
                    
                    print(f"  [VRAM Breakdown]")
                    print(f"    ├─ Python CPU Overhead: {python_overhead:6.3f} GB")
                    print(f"    ├─ CUDA Allocated VRAM: {cuda_allocated:6.3f} GB")
                    print(f"    │   ├─ Model Weights:   {min(est_weights_gb, cuda_allocated):6.3f} GB")
                    print(f"    │   ├─ KV Cache VRAM:   {kv_cache_resident_gb:6.3f} GB")
                    print(f"    │   └─ Activations/Tmp: {est_activations_gb:6.3f} GB")
                    print(f"    └─ CUDA Reserved VRAM:  {cuda_reserved:6.3f} GB")
                
                print("---------------------------------------------------------")
                if kv_sum:
                    sessions = kv_sum.get("sessions", 0)
                    vram_saved = kv_sum.get("vram_saved_mb", 0.0)
                    active_blocks = pager.get("gpu_resident_mb", 0.0) / 2.0 if pager else 0 # approximate block count
                    total_blocks = pager.get("tracked_blocks", 0) if pager else 0
                    
                    print(f"  Active Sessions:  {sessions:<10} |  VRAM Saved by SVD: {vram_saved:.1f} MB")
                    print(f"  Pager Blocks:     {total_blocks} tracked blocks ({kv_cache_resident_gb * 1024:.1f} MB resident)")
            else:
                print("  [Server API status: Offline / Starting up...]")
            
            print("=========================================================")
            print("Press Ctrl+C to stop monitoring. Logs are saved to:")
            print(f"  {log_file}")
            print("=========================================================")
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\nError in monitor loop: {e}")

if __name__ == '__main__':
    main()
