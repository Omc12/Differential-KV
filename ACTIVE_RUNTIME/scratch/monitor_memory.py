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
            
            if runtime_info:
                device = runtime_info.get("device", "unknown")
                serving_mode = runtime_info.get("serving_mode", "unknown")
                model = runtime_info.get("model", "unknown")
                
                print(f"  Device:           {device:<10}  |  Serving Mode:      {serving_mode}")
                print(f"  Model Loaded:     {model}")
                
                if device == "mps":
                    mps_allocated = runtime_info.get("mps_allocated_gb", 0.0)
                    mps_driver = runtime_info.get("mps_driver_gb", 0.0)
                    print(f"  MPS Allocated:    {mps_allocated:6.3f} GB |  MPS Driver (Total): {mps_driver:6.3f} GB")
                    # Python process overhead
                    python_overhead = max(0.0, rss_gb - mps_allocated)
                    print(f"  Python Overhead:  {python_overhead:6.3f} GB (RSS - MPS Allocated)")
                elif device == "cuda":
                    cuda_allocated = runtime_info.get("cuda_allocated_gb", 0.0)
                    cuda_reserved = runtime_info.get("cuda_reserved_gb", 0.0)
                    print(f"  CUDA Allocated:   {cuda_allocated:6.3f} GB |  CUDA Reserved:     {cuda_reserved:6.3f} GB")
                    python_overhead = max(0.0, rss_gb - cuda_allocated)
                    print(f"  Python Overhead:  {python_overhead:6.3f} GB (RSS - CUDA Allocated)")
                
                kv_sum = runtime_info.get("kv_summary", {})
                if kv_sum:
                    sessions = kv_sum.get("sessions", 0)
                    vram_saved = kv_sum.get("vram_saved_mb", 0.0)
                    pager = kv_sum.get("pager", {})
                    active_blocks = 0
                    total_blocks = 0
                    if pager and isinstance(pager, dict):
                        # Pager summary usually contains list of active blocks or sizes
                        # Let's count keys or read values if present
                        active_blocks = pager.get("active_blocks", 0)
                        total_blocks = pager.get("total_blocks", 0)
                    
                    print(f"  Active Sessions:  {sessions:<10} |  VRAM Saved by SVD: {vram_saved:.1f} MB")
                    print(f"  Pager Blocks:     {active_blocks}/{total_blocks} (Active/Total)")
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
