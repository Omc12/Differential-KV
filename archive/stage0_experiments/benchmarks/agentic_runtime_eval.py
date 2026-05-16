"""
benchmarks/agentic_runtime_eval.py

Evaluates the Unified Cognitive Runtime (UCR) on long-horizon agentic workflows.
Metrics: reasoning survival, memory stability, repair efficiency.
"""

import torch
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime

class AgenticRuntimeEval:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.runtime = UnifiedCognitiveRuntime(config)
        self.results = []

    def simulate_agent_workflow(self, workflow_name: str, steps: int = 100):
        """
        Simulates a long-horizon reasoning workflow (e.g., coding, planning).
        Uses synthetic hidden states and perturbations to simulate collapse.
        """
        print(f"Starting Agentic Workflow: {workflow_name} ({steps} steps)")
        self.runtime.initialize_runtime()
        
        workflow_history = []
        
        # Base hidden state
        hidden_dim = self.config.get("hidden_dim", 768)
        layers = self.config.get("layers", 12)
        base_hiddens = [torch.randn(1, 1, hidden_dim) for _ in range(layers)]
        
        for step in range(steps):
            # Simulate natural drift and occasional "cognitive cliffs"
            drift_factor = 0.01 * (step / 10)
            noise = [torch.randn(1, 1, hidden_dim) * drift_factor for _ in range(layers)]
            
            # Every 30 steps, simulate a "reasoning pivot" (higher curvature/acceleration)
            if step % 30 == 0:
                noise = [n * 10.0 for n in noise]
            
            current_hidden = [base + n for base, n in zip(base_hiddens, noise)]
            # Target hidden is the "clean" trajectory
            target_hidden = base_hiddens
            
            # Dummy KV and Attention
            dummy_kv = [(torch.randn(1, 8, 1, 64), torch.randn(1, 8, 1, 64)) for _ in range(layers)]
            
            # Focused attention (not fragmented)
            dummy_attn = []
            for _ in range(layers):
                # Put most weight on the last token and some on the first
                logits = torch.ones(1, 8, 1, step + 1) * -10.0
                logits[..., -1] = 10.0
                logits[..., 0] = 5.0
                dummy_attn.append(torch.softmax(logits, dim=-1))
            
            # Process step with UCR
            result = self.runtime.process_step(
                current_hidden, 
                dummy_kv, 
                attentions=dummy_attn, 
                target_hidden=target_hidden
            )
            
            workflow_history.append({
                "step": step,
                "health": result["health"].cognitive_health_score,
                "collapse_prob": result["health"].collapse_probability,
                "repaired": result["intervention"]["repaired"],
                "target_rank": result["resources"]["target_rank"]
            })
            
            if step % 20 == 0:
                print(f"Step {step}: Health={result['health'].cognitive_health_score:.2f}, State={result['runtime_state']}")

        summary = self.runtime.runtime_summary()
        summary["workflow"] = workflow_name
        summary["history"] = workflow_history
        self.results.append(summary)
        return summary

    def run_suite(self):
        scenarios = ["recursive_planning", "tool_use_chain", "long_context_retrieval"]
        for scenario in scenarios:
            self.simulate_agent_workflow(scenario, steps=150)
            
    def plot_results(self, output_path: str = "results/phase21/agentic_survival.png"):
        plt.figure(figsize=(12, 6))
        for res in self.results:
            steps = [h["step"] for h in res["history"]]
            health = [h["health"] for h in res["history"]]
            plt.plot(steps, health, label=f"{res['workflow']} (Avg Health: {res['avg_health']:.2f})")
            
            # Mark repairs
            repair_steps = [h["step"] for h in res["history"] if h["repaired"]]
            repair_health = [h["health"] for h in res["history"] if h["repaired"]]
            plt.scatter(repair_steps, repair_health, marker='x', color='red', alpha=0.5)

        plt.title("Agentic Reasoning Survival under UCR")
        plt.xlabel("Inference Steps")
        plt.ylabel("Cognitive Health Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    def save_results(self, path: str = "results/phase21/agentic_eval.json"):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("results/phase21", exist_ok=True)
    
    config = {
        "vram_limit_gb": 8,
        "base_rank": 16,
        "max_rank": 64,
        "hidden_dim": 768,
        "layers": 12,
        "device": "cpu" # Using CPU for simulation
    }
    evaluator = AgenticRuntimeEval(config)
    evaluator.run_suite()
    evaluator.save_results()
    evaluator.plot_results()
