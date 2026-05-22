import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def plot_fragmentation_growth(log_paths: List[str], output_path: str = "results/reconstruction_6_5/fragmentation_growth.png"):
    plt.figure(figsize=(10, 6))
    
    for path in log_paths:
        if not os.path.exists(path): continue
        
        name = os.path.basename(os.path.dirname(path))
        data = []
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("category") == "memory_fragmentation":
                    data.append({
                        "time": entry["timestamp"], 
                        "frag": entry.get("gpu_fragmentation_ratio", 0)
                    })
        
        if not data: continue
        
        df = pd.DataFrame(data)
        df["time"] = (df["time"] - df["time"].min()) / 3600 # Convert to hours
        plt.plot(df["time"], df["frag"] * 100, label=name)

    plt.xlabel("Execution Time (Hours)")
    plt.ylabel("VRAM Fragmentation (%)")
    plt.title("Empirical Fragmentation Growth (Long-Horizon)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_fragmentation_growth(["results/reconstruction_6_5/short_val_run/raw_telemetry.json"])
