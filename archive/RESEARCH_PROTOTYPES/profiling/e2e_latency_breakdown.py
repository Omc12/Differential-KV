"""
profiling/e2e_latency_breakdown.py

End-to-end breakdown of inference latency in UnifiedCognitiveRuntime.
Measures: kernel time, routing overhead, memory migration latency, retrieval orchestration cost.
"""

import torch
import time
from typing import Dict, Any, List
import numpy as np

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

class LatencyBreakdownProfiler:
    def __init__(self, runtime: UnifiedCognitiveRuntime):
        self.runtime = runtime
        self.stats = {
            "health_eval": [],
            "resource_alloc": [],
            "intervention": [],
            "anchor_mgmt": [],
            "total": []
        }

    def profile_step(self, hidden_states, kv_states):
        start_total = time.perf_counter()
        
        # 1. Health Eval
        s = time.perf_counter()
        health_state = self.runtime.state_engine.process_step(hidden_states)
        self.stats["health_eval"].append(time.perf_counter() - s)
        
        # 2. Resource Alloc
        s = time.perf_counter()
        resources = self.runtime.memory_optimizer.allocate_resources(
            cognitive_state=health_state.__dict__,
            context_depth=self.runtime.current_step
        )
        self.stats["resource_alloc"].append(time.perf_counter() - s)
        
        # 3. Intervention
        s = time.perf_counter()
        # Simulate some intervention probability
        if health_state.collapse_probability > 0.3:
            self.runtime.actr.evaluate_and_repair(
                self.runtime.current_step,
                health_state.__dict__,
                {"collapse_probability": health_state.collapse_probability},
                hidden_states,
                kv_states
            )
        self.stats["intervention"].append(time.perf_counter() - s)
        
        # 4. Anchor Mgmt
        s = time.perf_counter()
        if self.runtime.current_step % 10 == 0:
            p = self.runtime.priority_manager.calculate_token_priority(
                token_id=0,
                hidden_state=hidden_states[-1][:, -1, :],
                attention_weights=torch.ones(1, 1)
            )
            if p > 0.7:
                self.runtime.update_anchor_state(self.runtime.current_step, hidden_states, kv_states, p)
        self.stats["anchor_mgmt"].append(time.perf_counter() - s)
        
        self.stats["total"].append(time.perf_counter() - start_total)
        self.runtime.current_step += 1

    def print_summary(self):
        print("\n--- E2E LATENCY BREAKDOWN SUMMARY ---")
        for key, vals in self.stats.items():
            avg = np.mean(vals) * 1000
            std = np.std(vals) * 1000
            print(f"{key:15}: {avg:8.3f} ms (±{std:6.3f})")

def run_e2e_breakdown(config: Dict[str, Any], num_steps: int = 100):
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    profiler = LatencyBreakdownProfiler(runtime)
    
    print(f"Profiling {num_steps} steps...")
    for i in range(num_steps):
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        profiler.profile_step(hidden, kv)
        
    profiler.print_summary()

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3,
        "use_lcg": True
    }
    run_e2e_breakdown(config)
