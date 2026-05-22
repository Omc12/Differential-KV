"""
experiments/phase15_actr_validation.py
Phase 15: ACTR Validation Experiment
Integrates monitoring, prediction, and repair to validate cognitive recovery.
"""

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from analysis.trajectory_monitor import CognitiveTrajectoryMonitor
from analysis.divergence_predictor import DivergencePredictor
from analysis.pivot_detector import ReasoningPivotDetector
from anchor_logic.active_repair_controller import ActiveRepairController
from anchor_logic.dynamic_rank_scheduler import DynamicRankScheduler
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory, SemanticReinjector
from anchor_logic.anchor_graph import DynamicAnchorGraph
from analysis.attractor_mapper import AttractorMapper
from transformers import AutoTokenizer

class ACTRExperiment:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.monitor = CognitiveTrajectoryMonitor(model_id, device)
        self.tokenizer = self.monitor.tokenizer
        self.predictor = DivergencePredictor()
        self.pivot_detector = ReasoningPivotDetector()
        self.memory = SemanticAnchorMemory()
        self.repair_controller = ActiveRepairController(self.memory)
        self.rank_scheduler = DynamicRankScheduler()
        self.mapper = AttractorMapper()
        self.reinjector = SemanticReinjector(self.memory)
        
    def run_experiment(self, prompt, noise_std=0.1, use_actr=True):
        print(f"\n--- Running ACTR Experiment (Enabled: {use_actr}) ---")
        print(f"Prompt: {prompt[:100]}...")
        
        # We simulate the generation loop to apply active control
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.monitor.device)
        generated_ids = input_ids.clone()
        
        # 1. First, we need a baseline (perfect) trajectory to calculate drift
        # If we are doing real-time, we might not have a target, but for research we do.
        print("Capturing baseline trajectory...")
        _, traj_base = self.monitor.run_generation(prompt, max_new_tokens=50)
        
        # 2. Run test generation (with noise + ACTR)
        print("Running test generation...")
        test_tokens = []
        metrics_history = []
        past_key_values = None
        
        # Reset monitor for test run
        self.monitor.prev_hidden_states = None
        self.monitor.prev_velocity = None
        
        for i in range(50):
            # Simulation of noise injection (compression artifacts)
            def actr_kv_mod(l_idx, k, v):
                # Inject noise
                kn = k + torch.randn_like(k) * noise_std
                vn = v + torch.randn_like(v) * noise_std
                
                if use_actr:
                    # Apply repairs from memory
                    # Reinjector handles substituting exact anchors back
                    kn = self.reinjector.apply_kv_substitution(kn, [i]) # Simplification
                    vn = self.reinjector.apply_kv_substitution(vn, [i])
                
                return kn, vn

            # We'll use the monitor's hook-based generation but step-by-step
            self.monitor._attach_hooks()
            
            # Rank scheduling (Task 4)
            context = self.rank_scheduler.classify_context(generated_ids[0].tolist(), self.tokenizer)
            stability = self.monitor.history[-1]["cognitive_stability_score"] if self.monitor.history else 1.0
            rank = self.rank_scheduler.determine_rank(context, stability)
            
            # Forward pass
            outputs = self.monitor.model(
                input_ids=generated_ids[:, -1:] if i > 0 else generated_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True
            )
            self.monitor._remove_hooks()
            
            # Monitor (Task 1)
            target_h = traj_base[i]["hidden"] if i < len(traj_base) else None
            metrics = self.monitor.monitor_step(
                self.monitor.captured_hidden_states, 
                target_h,
                self.monitor.captured_attentions
            )
            metrics_history.append(metrics)
            
            # Predict (Task 2)
            self.predictor.update(metrics)
            prediction = self.predictor.predict_collapse()
            
            # Pivot Detection (Task 5)
            pivot = self.pivot_detector.detect_pivot(generated_ids[0].tolist(), self.tokenizer, metrics)
            
            # Repair (Task 3)
            if use_actr:
                repair_action = self.repair_controller.evaluate_and_repair(
                    i, metrics, prediction, self.monitor.captured_hidden_states, outputs.past_key_values
                )
            
            # Mapping (Task 7)
            h_flat = self.monitor.captured_hidden_states[-1][:, -1, :].cpu().numpy()
            v_flat = np.zeros_like(h_flat) # Placeholder for velocity projection
            self.mapper.record_state(h_flat, v_flat, metrics["cognitive_stability_score"])
            
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1:].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
        return {
            "text": self.tokenizer.decode(generated_ids[0]),
            "metrics": metrics_history,
            "repairs": self.repair_controller.active_repairs
        }

if __name__ == "__main__":
    exp = ACTRExperiment()
    prompt = "Question: If a train travels at 60 mph for 2 hours and then at 80 mph for 3 hours, what is the total distance traveled? Let's think step by step."
    
    # 1. Test baseline collapse
    res_no_actr = exp.run_experiment(prompt, noise_std=0.15, use_actr=False)
    
    # 2. Test ACTR recovery
    exp_actr = ACTRExperiment() # New instance to reset
    res_actr = exp_actr.run_experiment(prompt, noise_std=0.15, use_actr=True)
    
    os.makedirs("results/phase15", exist_ok=True)
    with open("results/phase15/actr_experiment_results.json", "w") as f:
        json.dump({
            "no_actr": {"text": res_no_actr["text"], "repairs": res_no_actr["repairs"]},
            "with_actr": {"text": res_actr["text"], "repairs": res_actr["repairs"]}
        }, f, indent=4)
        
    print("\n--- EXPERIMENT COMPLETE ---")
    print(f"No ACTR Output: {res_no_actr['text'].encode('ascii', 'ignore').decode('ascii')}")
    print(f"With ACTR Output: {res_actr['text'].encode('ascii', 'ignore').decode('ascii')}")
    print(f"Repairs Applied: {res_actr['repairs']}")
