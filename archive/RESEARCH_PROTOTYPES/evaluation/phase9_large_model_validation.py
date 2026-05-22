import sys
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.phase9_semantic_mapping import SemanticImportanceAnalyzer
from evaluation.perplexity_eval import Phase8PerplexityEvaluator

def run_large_model_validation(model_id="Qwen/Qwen2-1.5B"):
    print(f"\n==================================================")
    print(f"PHASE 9: LARGE MODEL VALIDATION — {model_id}")
    print(f"==================================================\n")
    
    try:
        analyzer = SemanticImportanceAnalyzer(model_id=model_id)
    except Exception as e:
        print(f"[Error] Could not load model {model_id}: {e}")
        return
        
    # Task 5: Focus ONLY on layer sensitivity and KL stability
    print("\n>>> Running Layer-wise Sensitivity Analysis (Qwen2-1.5B)...")
    layer_results = []
    # We do a subset of layers if it's too slow, but 1.5B should be fine
    for l in tqdm(range(analyzer.num_layers), desc="Layers"):
        # Use Rank 8 for sensitivity
        metrics = analyzer._get_quick_metrics("Layer-Shared Rank8", layer_mask={l}, samples=1, context_len=1024)
        metrics["layer"] = l
        layer_results.append(metrics)
        
    # Save results
    os.makedirs("results/phase9", exist_ok=True)
    output_path = f"results/phase9/large_model_validation_{model_id.split('/')[-1]}.json"
    with open(output_path, "w") as f:
        json.dump(layer_results, f, indent=2)
        
    print(f"\n[OK] Large model validation complete: {output_path}")

if __name__ == "__main__":
    # We prioritize 1.5B as it's more likely to fit in memory
    run_large_model_validation("Qwen/Qwen2-1.5B")
