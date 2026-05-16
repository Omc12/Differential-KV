import torch
import json
import os
import time
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.multidomain_symbolic_suite import MultidomainSymbolicSuite
from runtime.attention_steering_resolver import AttentionSteeringResolver
from transformers import AutoTokenizer, DynamicCache

class SteeringSensitivitySweeper:
    def __init__(self, results_dir="results/reconstruction_20_4"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.log_file = os.path.join(self.results_dir, "steering_sensitivity_sweep.jsonl")
        
        print("[PHASE 18.1A] Loading REAL Checkpoint for Sweep...")
        loader = Qwen7BRealLoader()
        self.model = loader.load(attn_implementation="eager")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.suite = MultidomainSymbolicSuite(self.tokenizer)

    def execute_run(self, bias_strength, ctx_len=4096, domain="activation_code"):
        # We'll use a fixed domain to isolate bias sensitivity
        test_case = self.suite.create_domain_test_case(domain, ctx_len)
        input_ids = torch.tensor([test_case['tokens']]).to("cuda")
        needle = test_case['needle']
        
        resolver = AttentionSteeringResolver(self.tokenizer, anchor_budget=2048, fidelity_budget=1024)
        resolver.logit_bias_strength = bias_strength
            
        past_key_values = DynamicCache()
        
        # Prefill
        chunk_size = 512
        for i in range(0, input_ids.shape[1], chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            with torch.no_grad():
                outputs = self.model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                legacy = past_key_values.to_legacy_cache()
                pruned, meta = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                past_key_values = DynamicCache.from_legacy_cache(pruned)
        
        # Generation
        curr_input = input_ids[:, -1:]
        generated_tokens = []
        steering_factors = []
        
        for _ in range(48):
            with torch.no_grad():
                outputs = self.model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True, output_attentions=True)
                logits = outputs.logits[:, -1, :]
                
                # Manual capture of steering factor for telemetry
                # (We could expose this in the resolver, but I'll do it here for the sweep)
                locked_token_ids = resolver.booster.get_locked_token_ids()
                factor = 0.0
                if locked_token_ids:
                    attentions = torch.stack(outputs.attentions)
                    mass = attentions[-1, 0].mean(dim=0)[-1]
                    steering_mask = resolver.booster.get_steering_bias(
                        torch.arange(mass.shape[0], device=mass.device), mass.device
                    )
                    span_mass = mass[steering_mask > 1.0].sum().item()
                    factor = max(0.0, 1.0 - (span_mass / 0.25))
                
                steering_factors.append(factor)
                
                # Apply resolver guidance
                attentions = torch.stack(outputs.attentions)
                logits = resolver.guide_decoder(logits, attentions)
                
                token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_tokens.append(token.item())
                curr_input = token
                if token.item() == self.tokenizer.eos_token_id: break
        
        output_text = self.tokenizer.decode(generated_tokens)
        success = needle.lower() in output_text.lower()
        avg_steering = sum(steering_factors) / len(steering_factors) if steering_factors else 0.0
        
        result = {
            "bias_strength": bias_strength,
            "success": success,
            "avg_steering_factor": avg_steering,
            "output": output_text[:100],
            "needle": needle
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        return result

    def run_sweep(self):
        if os.path.exists(self.log_file): os.remove(self.log_file)
        
        # Sweep logit_bias_strength
        strengths = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
        
        for s in strengths:
            print(f"[SWEEP] Testing Bias Strength: {s}")
            res = self.execute_run(s)
            print(f"[RESULT] Bias {s}: Success={res['success']}, AvgSteering={res['avg_steering_factor']:.3f}")

if __name__ == "__main__":
    sweeper = SteeringSensitivitySweeper()
    sweeper.run_sweep()
