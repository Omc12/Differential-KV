import json
import sys
import os

def main():
    log_path = "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/scratch/memory_log.json"
    if not os.path.exists(log_path):
        print(f"Error: log file {log_path} not found!")
        sys.exit(1)
        
    with open(log_path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON: {e}")
            sys.exit(1)
            
    print(f"Loaded {len(data)} log entries.")
    
    # Filter entries that have valid runtime_info
    valid_entries = [e for e in data if e.get("runtime_info") and "device" in e["runtime_info"]]
    if not valid_entries:
        print("No valid runtime entries found (server was offline during monitoring).")
        sys.exit(1)
        
    print(f"Found {len(valid_entries)} active runtime entries.")
    
    # Initial baseline
    baseline = valid_entries[0]
    b_rss = baseline["rss_gb"]
    b_vms = baseline["vms_gb"]
    b_mps_alloc = baseline["runtime_info"].get("mps_allocated_gb", 0.0)
    b_mps_driver = baseline["runtime_info"].get("mps_driver_gb", 0.0)
    
    print("\n--- BASELINE STATE (Server Started, Idle) ---")
    print(f"  System RSS (Process RAM):    {b_rss:.3f} GB")
    print(f"  PyTorch MPS Allocated:       {b_mps_alloc:.3f} GB")
    print(f"  MPS Driver (Total OS):       {b_mps_driver:.3f} GB")
    print(f"  Python Overhead (RSS - MPS): {max(0.0, b_rss - b_mps_alloc):.3f} GB")
    
    # Peak stats
    peak_rss = 0.0
    peak_mps_alloc = 0.0
    peak_mps_driver = 0.0
    peak_vram_saved = 0.0
    peak_active_blocks = 0
    peak_total_blocks = 0
    active_session_found = False
    
    peak_rss_entry = None
    peak_mps_alloc_entry = None
    peak_mps_driver_entry = None
    
    for entry in valid_entries:
        ri = entry["runtime_info"]
        rss = entry.get("rss_gb", 0.0)
        mps_alloc = ri.get("mps_allocated_gb", 0.0)
        mps_driver = ri.get("mps_driver_gb", 0.0)
        kv_sum = ri.get("kv_summary", {})
        vram_saved = kv_sum.get("vram_saved_mb", 0.0)
        pager = kv_sum.get("pager", {})
        active_blocks = pager.get("active_blocks", 0) if isinstance(pager, dict) else 0
        total_blocks = pager.get("total_blocks", 0) if isinstance(pager, dict) else 0
        sessions = kv_sum.get("sessions", 0)
        
        if sessions > 0 or active_blocks > 0:
            active_session_found = True
            
        if rss > peak_rss:
            peak_rss = rss
            peak_rss_entry = entry
        if mps_alloc > peak_mps_alloc:
            peak_mps_alloc = mps_alloc
            peak_mps_alloc_entry = entry
        if mps_driver > peak_mps_driver:
            peak_mps_driver = mps_driver
            peak_mps_driver_entry = entry
        if vram_saved > peak_vram_saved:
            peak_vram_saved = vram_saved
        if active_blocks > peak_active_blocks:
            peak_active_blocks = active_blocks
        if total_blocks > peak_total_blocks:
            peak_total_blocks = total_blocks

    print("\n--- PEAK STATE (During Prefill/Generation) ---")
    print(f"  Peak RSS (Process RAM):      {peak_rss:.3f} GB")
    if peak_rss_entry:
        r_ri = peak_rss_entry["runtime_info"]
        print(f"    (at peak RSS: MPS Alloc={r_ri.get('mps_allocated_gb'):.3f} GB, MPS Driver={r_ri.get('mps_driver_gb'):.3f} GB)")
    print(f"  Peak PyTorch MPS Allocated:  {peak_mps_alloc:.3f} GB")
    if peak_mps_alloc_entry:
        a_ri = peak_mps_alloc_entry["runtime_info"]
        print(f"    (at peak MPS: RSS={peak_mps_alloc_entry.get('rss_gb'):.3f} GB, MPS Driver={a_ri.get('mps_driver_gb'):.3f} GB)")
    print(f"  Peak MPS Driver (Total OS):  {peak_mps_driver:.3f} GB")
    if peak_mps_driver_entry:
        d_ri = peak_mps_driver_entry["runtime_info"]
        print(f"    (at peak Driver: RSS={peak_mps_driver_entry.get('rss_gb'):.3f} GB, MPS Alloc={d_ri.get('mps_allocated_gb'):.3f} GB)")
    
    print("\n--- DIFFKV SUMMARY STATISTICS ---")
    print(f"  Active Session Detected:     {active_session_found}")
    print(f"  Peak SVD VRAM Saved:         {peak_vram_saved:.2f} MB")
    print(f"  Peak Pager Blocks (Act/Tot): {peak_active_blocks}/{peak_total_blocks}")
    
    # Let's inspect memory timeline at key transitions
    # Let's print out transitions where sessions change, or memory increases significantly
    print("\n--- SIGNIFICANT MEMORY EVENTS TIMELINE ---")
    last_sess = -1
    last_alloc = -1.0
    for entry in valid_entries:
        ri = entry["runtime_info"]
        sess = ri.get("kv_summary", {}).get("sessions", 0)
        mps_alloc = ri.get("mps_allocated_gb", 0.0)
        elapsed = entry["timestamp"]
        
        # Print when session starts, stops, or memory changes by > 200MB
        if sess != last_sess or abs(mps_alloc - last_alloc) > 0.2:
            print(f"  [{elapsed:5.1f}s] Sessions: {last_sess} -> {sess} | MPS Allocated: {last_alloc:.3f} GB -> {mps_alloc:.3f} GB | RSS: {entry['rss_gb']:.3f} GB | Driver: {ri.get('mps_driver_gb'):.3f} GB")
            last_sess = sess
            last_alloc = mps_alloc

    # Final state
    final = valid_entries[-1]
    f_rss = final["rss_gb"]
    f_mps_alloc = final["runtime_info"].get("mps_allocated_gb", 0.0)
    f_mps_driver = final["runtime_info"].get("mps_driver_gb", 0.0)
    print("\n--- FINAL STATE (End of Run) ---")
    print(f"  System RSS:                  {f_rss:.3f} GB")
    print(f"  PyTorch MPS Allocated:       {f_mps_alloc:.3f} GB")
    print(f"  MPS Driver (Total OS):       {f_mps_driver:.3f} GB")

if __name__ == "__main__":
    main()
