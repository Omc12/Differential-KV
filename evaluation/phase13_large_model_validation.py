"""
evaluation/phase13_large_model_validation.py
Phase 13 Task 7: Mechanistic Validation on Large Models (Qwen2-1.5B)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from analysis.manifold_mechanics import SemanticManifoldAnalyzer
from analysis.influence_analyzer import AnchorInfluenceAnalyzer
import json

def run_validation(model_id="Qwen/Qwen2-1.5B"):
    print(f"\n==================================================")
    print(f"PHASE 13: LARGE MODEL VALIDATION — {model_id}")
    print(f"==================================================\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We'll use the existing analyzers but with the larger model
    manifold_analyzer = SemanticManifoldAnalyzer(model_id=model_id, device=device)
    
    text = "The scalability of transformer models is one of the most significant developments in artificial intelligence. As models grow, the semantic manifolds they form become more structured and potentially easier to stabilize with sparse anchors."
    
    # 1. Trajectory Analysis
    print("\n>>> Running Trajectory Analysis (1.5B)...")
    res_drift = manifold_analyzer.analyze_mechanics(text, anchor_interval=64)
    manifold_analyzer.visualize_drift(res_drift, f"results/phase13/plots/manifold_drift_{model_id.split('/')[-1]}.png")
    
    # 2. Influence Propagation
    print("\n>>> Running Influence Propagation (1.5B)...")
    influence_analyzer = AnchorInfluenceAnalyzer(model_id=model_id, device=device)
    res_influence = influence_analyzer.measure_propagation(text, anchor_pos=30)
    influence_analyzer.visualize_influence(res_influence, f"results/phase13/plots/influence_propagation_{model_id.split('/')[-1]}.png")
    
    # Save results
    output_path = f"results/phase13/large_model_validation_{model_id.split('/')[-1]}.json"
    with open(output_path, "w") as f:
        json.dump({
            "drift": res_drift,
            "influence": res_influence
        }, f, indent=4)
        
    print(f"\n[OK] Large model validation complete: {output_path}")

if __name__ == "__main__":
    run_validation("Qwen/Qwen2-1.5B")
