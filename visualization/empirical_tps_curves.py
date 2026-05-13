import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def plot_empirical_tps(log_paths: List[str], output_path: str = "results/reconstruction_6_5/tps_curves.png"):
    plt.figure(figsize=(10, 6))
    
    for path in log_paths:
        if not os.path.exists(path): continue
        
        name = os.path.basename(os.path.dirname(path))
        data = []
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("category") == "tps":
                    data.append({"time": entry["timestamp"], "tps": entry["value"]})
        
        if not data: continue
        
        df = pd.DataFrame(data)
        df["time"] = df["time"] - df["time"].min()
        plt.plot(df["time"], df["tps"], label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Tokens Per Second (TPS)")
    plt.title("Empirical TPS Scaling (Hardware Truth)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Test plot
    plot_empirical_tps([
        "results/reconstruction_6_5/short_val_run/raw_telemetry.json",
        "results/reconstruction_6_5/concurrency_val_run/raw_telemetry.json"
    ])
