import json
import matplotlib.pyplot as plt
import numpy as np

def detect_scaling_breakpoint(dense_results_path: str, sparse_results_path: str):
    """
    PHASE 11C: LONG-CONTEXT SPARSE ADVANTAGE VALIDATION
    
    Identifies the context length where sparse TPS exceeds dense TPS.
    """
    with open(dense_results_path, "r") as f:
        dense_data = json.load(f)
    with open(sparse_results_path, "r") as f:
        sparse_data = json.load(f)
        
    ctx_lengths = [d["context_length"] for d in dense_data if "error" not in d]
    dense_tps = [d["tps"] for d in dense_data if "error" not in d]
    sparse_tps = [d["tps"] for d in sparse_data if "error" not in d]
    
    # Filter to matching context lengths
    common_ctx = sorted(list(set(ctx_lengths) & set([d["context_length"] for d in sparse_data if "error" not in d])))
    dense_tps_f = [dense_tps[ctx_lengths.index(c)] for c in common_ctx]
    sparse_tps_f = [sparse_tps[[d["context_length"] for d in sparse_data].index(c)] for c in common_ctx]
    
    breakpoint_ctx = None
    for i in range(len(common_ctx)):
        if sparse_tps_f[i] > dense_tps_f[i]:
            breakpoint_ctx = common_ctx[i]
            break
            
    print(f"Detected sparse advantage breakpoint at context length: {breakpoint_ctx if breakpoint_ctx else 'Not found'}")
    
    # Generate plot (simulated)
    # plt.plot(common_ctx, dense_tps_f, label="Dense")
    # plt.plot(common_ctx, sparse_tps_f, label="Sparse")
    # plt.xlabel("Context Length")
    # plt.ylabel("TPS")
    # plt.legend()
    # plt.savefig("results/reconstruction_11/tps_scaling_comparison.png")
    
    return breakpoint_ctx

if __name__ == "__main__":
    # detect_scaling_breakpoint("results/reconstruction_11/extreme_context_dense.json", "results/reconstruction_11/extreme_context_sparse.json")
    pass
