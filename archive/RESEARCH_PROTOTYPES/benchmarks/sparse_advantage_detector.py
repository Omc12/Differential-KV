import json

def detect_sparse_advantage(results_path: str):
    """
    PHASE 11C: LONG-CONTEXT SPARSE ADVANTAGE VALIDATION
    
    Quantifies the efficiency gain of the sparse runtime compared to a base.
    """
    with open(results_path, "r") as f:
        data = json.load(f)
        
    # Example: Calculate relative improvement over the smallest context
    base_tps = data[0]["tps"]
    advantages = []
    for d in data:
        if "tps" in d:
            efficiency = d["tps"] / base_tps
            advantages.append({
                "context_length": d["context_length"],
                "efficiency_relative_to_base": efficiency
            })
            
    print(f"Sparse advantage analysis complete for {results_path}")
    return advantages
