import sys
import os
import time
import subprocess
import psutil
import json
import argparse

def get_process_tree_mem(parent_pid):
    """
    Finds parent process and all its children recursively,
    returning a dict of total RSS, VMS (in MB), and CPU %.
    """
    total_rss = 0.0
    total_vms = 0.0
    total_cpu = 0.0
    proc_list = []
    
    try:
        parent = psutil.Process(parent_pid)
        if parent.status() == 'zombie':
            return None
        proc_list.append(parent)
        children = parent.children(recursive=True)
        proc_list.extend(children)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
        
    for p in proc_list:
        try:
            # We use oneshot() to fetch all info in one syscall for efficiency
            with p.oneshot():
                mem = p.memory_info()
                total_rss += mem.rss / (1024 * 1024)
                total_vms += mem.vms / (1024 * 1024)
                total_cpu += p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    return {
        "rss_mb": total_rss,
        "vms_mb": total_vms,
        "cpu_percent": total_cpu,
        "num_processes": len(proc_list)
    }

def monitor_pid(pid, log_file, interval=0.2):
    print(f"[*] Monitoring process tree for PID {pid}...")
    print(f"[*] Saving memory logs to {log_file}")
    print(f"{'Time (s)':<10} | {'RSS (MB)':<12} | {'VMS (MB)':<12} | {'CPU %':<10} | {'Procs':<8}")
    print("-" * 60)
    
    history = []
    start_time = time.time()
    
    try:
        # Prime the CPU measurement
        try:
            p = psutil.Process(pid)
            p.cpu_percent(interval=None)
            for child in p.children(recursive=True):
                child.cpu_percent(interval=None)
        except Exception:
            pass
            
        peak_rss = 0.0
        peak_vms = 0.0
        peak_cpu = 0.0
        
        while True:
            elapsed = time.time() - start_time
            stats = get_process_tree_mem(pid)
            if stats is None:
                print("\n[!] Process tree terminated.")
                break
                
            peak_rss = max(peak_rss, stats["rss_mb"])
            peak_vms = max(peak_vms, stats["vms_mb"])
            peak_cpu = max(peak_cpu, stats["cpu_percent"])
            
            entry = {
                "elapsed_s": round(elapsed, 2),
                "rss_mb": round(stats["rss_mb"], 2),
                "vms_mb": round(stats["vms_mb"], 2),
                "cpu_percent": round(stats["cpu_percent"], 1),
                "num_processes": stats["num_processes"]
            }
            history.append(entry)
            
            # Print update
            print(f"{elapsed:10.1f} | {stats['rss_mb']:12.2f} | {stats['vms_mb']:12.2f} | {stats['cpu_percent']:10.1f} | {stats['num_processes']:8d}", end="\r")
            
            # Periodically write to file
            if len(history) % 5 == 0:
                with open(log_file, "w") as f:
                    json.dump({"peak_rss_mb": peak_rss, "peak_vms_mb": peak_vms, "peak_cpu": peak_cpu, "history": history}, f, indent=2)
                    
            time.sleep(interval)
            
        print("\n" + "=" * 60)
        print("                 MONITORING SUMMARY")
        print("=" * 60)
        print(f"Peak RAM (RSS):   {peak_rss:10.2f} MB")
        print(f"Peak Virtual:     {peak_vms:10.2f} MB")
        print(f"Peak CPU Usage:   {peak_cpu:10.1f} %")
        print(f"Total Duration:   {time.time() - start_time:10.2f} seconds")
        print("=" * 60)
        
        # Save final history
        with open(log_file, "w") as f:
            json.dump({"peak_rss_mb": peak_rss, "peak_vms_mb": peak_vms, "peak_cpu": peak_cpu, "history": history}, f, indent=2)
            
    except KeyboardInterrupt:
        print("\n[!] Monitoring stopped by user.")

def run_and_monitor(cmd, log_file, interval=0.2):
    print(f"[*] Launching command: {' '.join(cmd)}")
    
    # Start the process
    # We pipe stdout/stderr to stderr so that the user's terminal sees the outputs in real-time,
    # or let the process inherit stdout/stderr if we don't need to capture them.
    proc = subprocess.Popen(cmd)
    pid = proc.pid
    
    # Run the monitor loop
    monitor_pid(pid, log_file, interval)
    
    # Wait for the process to finish
    proc.wait()
    print(f"[*] Process exited with code {proc.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Profile memory/CPU of C++ DiffKV natively")
    parser.add_argument("--pid", type=int, help="PID of existing process to monitor")
    parser.add_argument("--cmd", type=str, help="Command to launch and monitor")
    parser.add_argument("--log", type=str, default="memory_profile.json", help="Log output path")
    parser.add_argument("--interval", type=float, default=0.2, help="Sample interval in seconds")
    
    args = parser.parse_args()
    
    if args.pid:
        monitor_pid(args.pid, args.log, args.interval)
    elif args.cmd:
        # Split command string into list
        import shlex
        cmd_args = shlex.split(args.cmd)
        run_and_monitor(cmd_args, args.log, args.interval)
    else:
        print("[!] Error: Must specify either --pid or --cmd.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
