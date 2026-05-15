import torch
import json
import time
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.multidomain_symbolic_suite import MultidomainSymbolicSuite
from validation.noisy_context_injector import NoisyContextInjector
from runtime.attention_steering_resolver import AttentionSteeringResolver
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner20_4:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.symbolic_suite = MultidomainSymbolicSuite(tokenizer)
        self.noise_injector = NoisyContextInjector()
        self.results_dir = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_4"
        os.makedirs(self.results_dir, exist_ok=True)
        self.log_file = os.path.join(self.results_dir, "msspsr_8k_fidelity.jsonl")

    def execute_single_run(self, mode, ctx_len, domain, use_noise=False, run_id=0):
        print(f"\n[RUNNING] Mode: {mode}, Context: {ctx_len}, Domain: {domain}, Noise: {use_noise}")
        
        # Setup Resolver
        if mode == "dense":
            resolver = None
        elif mode == "msspsr_20_4":
            # PHASE 20.4: MSSPSR - Minimal steering stability
            resolver = AttentionSteeringResolver(self.tokenizer, anchor_budget=ctx_len//2, fidelity_budget=1024)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create Test Case
        test_case = self.symbolic_suite.create_domain_test_case(domain, ctx_len)
        needle = test_case['needle']
        
        if use_noise:
            full_prompt = self.noise_injector.inject_noise(test_case['full_prompt'], intensity=0.1)
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
        else:
            input_ids = torch.tensor([test_case['tokens']]).to("cuda")

        past_key_values = DynamicCache()
        
        # Prefill
        chunk_size = 512
        for i in range(0, input_ids.shape[1], chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            pos_ids = torch.arange(i, i + chunk.shape[1], device="cuda").unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=chunk, 
                    position_ids=pos_ids,
                    past_key_values=past_key_values, 
                    use_cache=True, 
                    output_hidden_states=True
                )
                if resolver:
                    legacy = past_key_values.to_legacy_cache()
                    pruned, meta = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                    past_key_values = DynamicCache.from_legacy_cache(pruned)
        
        vram = torch.cuda.memory_allocated() / 1024**3

        # Generation
        print(f"Generating for {mode}...")
        curr_input = input_ids[:, -1:]
        curr_pos = input_ids.shape[1]
        generated_tokens = []
        start_gen = time.perf_counter()
        
        for _ in range(64):
            pos_ids = torch.tensor([[curr_pos]], device="cuda")
            with torch.no_grad():
                # Enable attentions for steering analysis
                outputs = self.model(
                    input_ids=curr_input, 
                    position_ids=pos_ids,
                    past_key_values=past_key_values, 
                    use_cache=True, 
                    output_attentions=True
                )
                logits = outputs.logits[:, -1, :] / 0.7
                attentions = torch.stack(outputs.attentions) if outputs.attentions else None
                
                if resolver and hasattr(resolver, 'guide_decoder'):
                    logits = resolver.guide_decoder(logits, attentions)
                
                token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_tokens.append(token.item())
                curr_input = token
                curr_pos += 1
                if token.item() == self.tokenizer.eos_token_id: break
        
        duration = time.perf_counter() - start_gen
        tps = len(generated_tokens) / duration if duration > 0 else 0
        output_text = self.tokenizer.decode(generated_tokens)
        
        # Validation
        def normalize(text):
            return "".join(text.lower().split())
            
        success = normalize(needle) in normalize(output_text)
        
        result = {
            "mode": mode,
            "ctx": ctx_len,
            "domain": domain,
            "noise": use_noise,
            "success": success,
            "tps": tps,
            "vram_gb": vram,
            "output": output_text,
            "expected": needle
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
            
        print(f"[RESULT] Success: {success}, TPS: {tps:.2f}, VRAM: {vram:.2f}GB")
        
        # Monitor stability
        if resolver:
             print(f"[STATUS] Booster Spans: {len(resolver.booster.active_spans)}")
             
        return result

    def run_suite(self):
        contexts = [8192]
        domains = ["activation_code", "api_key", "json_snippet"]
        modes = ["dense", "msspsr_20_4"]
        
        for ctx in contexts:
            for domain in domains:
                for mode in modes:
                    # Run once per condition to verify fix
                    self.execute_single_run(mode, ctx, domain, use_noise=False)

if __name__ == "__main__":
    print("[PHASE 20.4] Initializing Hardware Validation...")
    loader = Qwen7BRealLoader()
    model = loader.load(attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    runner = ValidationRunner20_4(model, tokenizer)
    if os.path.exists(runner.log_file): os.remove(runner.log_file)
    runner.run_suite()
