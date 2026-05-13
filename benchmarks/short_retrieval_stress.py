"""
benchmarks/short_retrieval_stress.py

Stress tests Differential KV retrieval under rapid topic switching.
Focus: retrieval retention, sparse degradation, retrieval collapse probability.
"""

import torch
import time
import random
from typing import List, Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment
from validation.replay_attack_detector import ReplayAttackDetector
from validation.hidden_state_auditor import HiddenStateAuditor
from validation.memory_contamination_guard import MemoryContaminationGuard
from validation.sparse_failure_mapper import SparseFailureMapper

def run_short_retrieval_stress(config: Dict[str, Any], num_topics: int = 5, steps_per_topic: int = 20):
    print(f"--- STARTING SHORT RETRIEVAL STRESS TEST ({num_topics} topics) ---")
    reset_environment()
    
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    replay_detector = ReplayAttackDetector()
    auditor = HiddenStateAuditor()
    guard = MemoryContaminationGuard()
    failure_mapper = SparseFailureMapper()
    
    run_id = f"run_{int(time.time())}"
    results = {
        "retention_scores": [],
        "collapse_probs": [],
        "latencies": []
    }
    
    # Simulate topics
    for topic_idx in range(num_topics):
        print(f"Processing Topic {topic_idx + 1}/{num_topics}...")
        
        # Simulate a key "anchor" information at the start of each topic
        # In a real scenario, this would be specific facts.
        
        for step in range(steps_per_topic):
            global_step = topic_idx * steps_per_topic + step
            
            # Generate fake hidden states and KV
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            
            # Adversarial audit
            if not auditor.audit(hidden[-1], run_id, global_step):
                print("ABORTING: Hidden state collision detected.")
                return
            
            start_time = time.perf_counter()
            output = runtime.process_step(hidden, kv)
            end_time = time.perf_counter()
            
            results["latencies"].append(end_time - start_time)
            results["collapse_probs"].append(output["health"].collapse_probability)
            
            # Check for replay
            if replay_detector.check_for_replay(hidden[-1]):
                 print("WARNING: Replay detected!")
            
            # Periodically test retrieval of previous topics
            if step == steps_per_topic - 1:
                # Simulate a retrieval check
                # Here we check if the anchors for previous topics are still in SAM
                anchors_found = 0
                for anchor in runtime.sam.anchors.values():
                    if anchor.importance_score > 0.8: # High priority anchors
                        anchors_found += 1
                
                retention = anchors_found / (topic_idx + 1)
                results["retention_scores"].append(retention)
                print(f"Topic {topic_idx + 1} complete. Retention: {retention:.2f}")

    print("--- SHORT RETRIEVAL STRESS TEST COMPLETE ---")
    
    # Final Metrics
    avg_retention = sum(results["retention_scores"]) / len(results["retention_scores"])
    max_collapse = max(results["collapse_probs"])
    avg_latency = sum(results["latencies"]) / len(results["latencies"])
    
    print(f"Average Retention: {avg_retention:.2%}")
    print(f"Max Collapse Probability: {max_collapse:.4f}")
    print(f"Average Latency per Step: {avg_latency*1000:.2f} ms")
    
    return results

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
    run_short_retrieval_stress(config)
