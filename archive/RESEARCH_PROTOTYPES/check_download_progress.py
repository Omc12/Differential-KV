import os
import glob

def check_progress():
    user_profile = os.environ.get("USERPROFILE")
    blob_dir = os.path.join(user_profile, ".cache", "huggingface", "hub", "models--Qwen--Qwen2.5-7B-Instruct", "blobs")
    
    if not os.path.exists(blob_dir):
        print("Download Directory not found.")
        return

    # Sum all files (complete and .incomplete)
    total_bytes = sum(os.path.getsize(os.path.join(blob_dir, f)) for f in os.listdir(blob_dir))
    total_gb = total_bytes / (1024**3)
    
    # Target size for Qwen2.5-7B-Instruct is ~15.3 GB
    target_gb = 15.3 
    percentage = (total_gb / target_gb) * 100
    
    print("="*40)
    print("PHASE 18.1 DOWNLOAD PROGRESS [MEASURED]")
    print("="*40)
    print(f"Current Size: {total_gb:.2f} GB")
    print(f"Target Size:  {target_gb:.2f} GB")
    print(f"Progress:     {percentage:.1f}%")
    print("="*40)
    
    incomplete = [f for f in os.listdir(blob_dir) if f.endswith(".incomplete")]
    print(f"Active Shards: {len(incomplete)}")
    if percentage >= 99.9:
        print("\n[SUCCESS] Download near completion. Run verify_phase_18_1_readiness.py next.")

if __name__ == "__main__":
    check_progress()
