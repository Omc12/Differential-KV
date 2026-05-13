import uuid
import random
import json
import os
from validation.clean_baseline_runner import run_baseline
from validation.reset_environment import reset_environment

class BlindExperimentRunner:
    def __init__(self, model_id, prompts):
        self.model_id = model_id
        self.prompts = prompts
        self.experiments = []
        self.mappings = {}

    def add_variant(self, name, config):
        run_id = f"RUN_{uuid.uuid4().hex[:8]}"
        self.experiments.append((run_id, config))
        self.mappings[run_id] = name
        print(f"Registered variant under Run ID: {run_id}")

    def run_all(self):
        # Shuffle to ensure blindness
        random.shuffle(self.experiments)
        
        all_results = {}
        
        for run_id, config in self.experiments:
            print(f"\n--- EXECUTING {run_id} ---")
            # In a real setup, we'd apply the config to the model
            # For this simulation, we'll just record the intent
            
            # Reset environment before each run
            reset_environment()
            
            # Execute (simplified for this script)
            # In practice, this would call a modified model runner
            results = run_baseline(self.model_id, self.prompts)
            
            all_results[run_id] = {
                "config": config,
                "metrics": results
            }
            
        return all_results

    def reveal_and_report(self, all_results):
        print("\n=== FINAL RESULTS (DE-MASKED) ===")
        final_report = {}
        for run_id, results in all_results.items():
            real_name = self.mappings[run_id]
            final_report[real_name] = results
            print(f"Variant: {real_name} ({run_id})")
            # Print average metrics
            avg_tps = sum(r['tokens_per_sec'] for r in results['metrics']) / len(results['metrics'])
            print(f"  Avg TPS: {avg_tps:.2f}")
            
        return final_report

if __name__ == "__main__":
    prompts = ["Tell me a long story about a robot.", "What is the capital of France?"]
    runner = BlindExperimentRunner("microsoft/phi-3-mini-4k-instruct", prompts)
    
    runner.add_variant("Baseline", {"type": "vanilla"})
    runner.add_variant("KV_Pruning_50", {"type": "pruning", "ratio": 0.5})
    
    results = runner.run_all()
    final = runner.reveal_and_report(results)
    
    with open("results/reality_reset/blind_eval_results.json", "w") as f:
        json.dump(final, f, indent=4)
