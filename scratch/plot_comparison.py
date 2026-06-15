import os
import json
import matplotlib.pyplot as plt

def generate_comparative_plots():
    artifact_dir = "/Users/omchimurkar1/.gemini/antigravity/brain/ada31170-301d-45cf-bbdf-321c6b861dbc"
    results_path = "/Users/omchimurkar1/Desktop/Differential-KV/benchmark_results_ollama.json"
    
    if not os.path.exists(results_path):
        print("Error: benchmark_results_ollama.json not found!")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    contexts = [1024, 2048, 4096, 8192, 16384]
    modes = ["dense_pytorch", "diffkv_mlx", "ollama_fp16", "ollama_quant"]
    labels = {
        "dense_pytorch": "Dense (PyTorch)",
        "diffkv_mlx": "DiffKV (MLX)",
        "ollama_fp16": "Ollama FP16",
        "ollama_quant": "Ollama Quantized (4-bit)"
    }
    colors = {
        "dense_pytorch": "#ff7675",
        "diffkv_mlx": "#0984e3",
        "ollama_fp16": "#e056fd",
        "ollama_quant": "#10ac84"
    }
    markers = {
        "dense_pytorch": "o",
        "diffkv_mlx": "s",
        "ollama_fp16": "^",
        "ollama_quant": "d"
    }

    # Extract metrics
    prefill = {m: [] for m in modes}
    tps = {m: [] for m in modes}
    acc = {m: [] for m in modes}
    
    for c in contexts:
        for m in modes:
            m_res = data.get(m, {}).get(str(c), {})
            if "error" in m_res or not m_res:
                prefill[m].append(None)
                tps[m].append(None)
                acc[m].append(None)
            else:
                prefill[m].append(m_res.get("prefill_s"))
                tps[m].append(m_res.get("decode_tps"))
                acc[m].append(m_res.get("accuracy"))

    plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'

    # 1. Prefill Latency
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, p) for c, p in zip(contexts, prefill[m]) if p is not None]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] for x in valid], marker=markers[m], label=labels[m], color=colors[m], linewidth=2.0, markersize=8)
    plt.title('Prefill Latency (TTFT) vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Prefill Latency (seconds)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'compare_prefill.png'), dpi=200)
    plt.close()

    # 2. Decode TPS
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, t) for c, t in zip(contexts, tps[m]) if t is not None]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] for x in valid], marker=markers[m], label=labels[m], color=colors[m], linewidth=2.0, markersize=8)
    plt.title('Decode Throughput (TPS) vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Throughput (tokens/second)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'compare_tps.png'), dpi=200)
    plt.close()

    # 3. Accuracy comparison
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, a) for c, a in zip(contexts, acc[m]) if a is not None]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] for x in valid], marker=markers[m], label=labels[m], color=colors[m], linewidth=2.0, markersize=8)
    plt.title('NIAH Retrieval Accuracy vs. Context Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Accuracy (0.0 to 1.0)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.ylim(-0.1, 1.1)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'compare_accuracy.png'), dpi=200)
    plt.close()

    print("Consolidated comparative plots generated successfully.")

if __name__ == "__main__":
    generate_comparative_plots()
