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
    print("Memory monitor started.")
    proc = find_gateway_process()
    if proc is None:
        print("Gateway process not found!")
        sys.exit(1)
    
    pid = proc.pid
    print(f"Monitoring process PID: {pid}")
    
    log_file = "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/scratch/memory_log.json"
    history = []
    
    # Run for 20 minutes max or until killed
    start_time = time.time()
    while time.time() - start_time < 1200:
        try:
            if not proc.is_running():
                print("Process terminated.")
                break
            
            mem_info = proc.memory_info()
            rss_gb = mem_info.rss / (1024 ** 3)
            vms_gb = mem_info.vms / (1024 ** 3)
            
            # Query runtime info if available
            runtime_info = {}
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/v1/runtime_info")
                with urllib.request.urlopen(req, timeout=0.2) as response:
                    runtime_info = json.loads(response.read().decode())
            except Exception:
                pass
                
            entry = {
                "timestamp": time.time() - start_time,
                "rss_gb": rss_gb,
                "vms_gb": vms_gb,
                "runtime_info": runtime_info
            }
            history.append(entry)
            
            # Save history to file
            with open(log_file, "w") as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Error in monitor: {e}")
            
        time.sleep(0.5)

if __name__ == '__main__':
    main()
