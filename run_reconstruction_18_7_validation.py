import torch
import os
import time
import json
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.long_context_workload_suite import LongContextWorkloadSuite
from runtime.hierarchical_memory_resolver import HierarchicalMemoryResolver
from runtime.hybrid_memory_resolver import HybridMemoryResolver # Phase 18.6
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from analysis.precision_cost_tracker import PrecisionCostTracker
from analysis.compute_balance_auditor import ComputeBalanceAuditor
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner:
    def __init__(self, model, tokenizer, results_dir="results/reconstruction_18_7/"):
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.suite = LongContextWorkloadSuite(tokenizer)
        self.overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)
        self.cost_tracker = PrecisionCostTracker()
        self.auditor = ComputeBalanceAuditor(min_tps=1.0) # Absolute minimum floor

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        modes = ["dense", "sparse_baseline", "continuity_aware", "capsule_linked"]
        
        for ctx_len in contexts:
            for mode in modes:
                print(f"\n[RUN] Mode: {mode} | Context: {ctx_len}")
                result = self.execute_single_run(mode, ctx_len)
                self.log_result(mode, ctx_len, result)

    def execute_single_run(self, mode, ctx_len):
        # Reset state
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Select Resolver based on mode
        resolver = None
        if mode == "sparse_baseline":
            # Just use the geometry manager with standard pruning
            resolver = HybridMemoryResolver(anchor_budget=ctx_len // 2)
            # Disable symbolic detection for baseline by setting threshold very high
            resolver.fidelity.threshold = 100.0 
        elif mode == "continuity_aware":
            resolver = HybridMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "capsule_linked":
            resolver = HierarchicalMemoryResolver(anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        
        # Test Case: NIAH with Symbolic IDs
        test_case = self.suite.create_needle_in_haystack(ctx_len, needle_pos_ratio=0.3)
        input_ids = torch.tensor([test_case['tokens']]).to("cuda")
        
        past_key_values = DynamicCache()
        chunk_size = 512
        chunks = self.overlap_scheduler.get_chunks(input_ids, chunk_size)
        
        start_time = time.perf_counter()
        
        # Prefill Phase
        for i, (chunk, _, _) in enumerate(chunks):
            with torch.no_grad():
                outputs = self.model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            if mode != "dense":
                legacy = past_key_values.to_legacy_cache()
                pruned, _ = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                past_key_values = DynamicCache.from_legacy_cache(pruned)
        
        prefill_end = time.perf_counter()
        ttft = prefill_end - start_time
        
        # Generation Phase
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        
        gen_start = time.perf_counter()
        for _ in range(64):
            with torch.no_grad():
                outputs = self.model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7 # Standard temp
                token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
                response_tokens.append(token.item())
                curr_input = token
                new_text = self.tokenizer.decode([token.item()])
                generated_text += new_text
                if self.tokenizer.eos_token in new_text: break
        
        gen_end = time.perf_counter()
        tps = len(response_tokens) / (gen_end - gen_start) if (gen_end - gen_start) > 0 else 0
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        success = test_case['answer'] in generated_text
        
        # Record Raw Artifacts
        if mode == "capsule_linked":
            with open(os.path.join(self.results_dir, "raw_capsule_registry.jsonl"), "a") as f:
                for cap in resolver.registry.capsules.values():
                    f.write(json.dumps({"ctx": ctx_len, "id": cap.capsule_id, "start": cap.start_idx, "end": cap.end_idx, "tier": cap.precision_tier, "entropy": cap.entropy_score}) + "\n")
            
            with open(os.path.join(self.results_dir, "raw_precision_allocations.jsonl"), "a") as f:
                f.write(json.dumps({"ctx": ctx_len, "budget": resolver.budget_controller.max_tokens, "utilization": resolver.budget_controller.get_utilization(resolver.registry)}) + "\n")

        with open(os.path.join(self.results_dir, "raw_compute_overheads.jsonl"), "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx_len, "tps": tps, "ttft": ttft, "vram": vram}) + "\n")

        return {
            "success": success,
            "tps": tps,
            "ttft": ttft,
            "vram_gb": vram,
            "output": generated_text,
            "expected": test_case['answer']
        }

    def log_result(self, mode, ctx, res):
        log_file = os.path.join(self.results_dir, "raw_validation_matrix.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx, **res}) + "\n")
        print(f"  [RESULT] Success: {res['success']} | TPS: {res['tps']:.2f} | VRAM: {res['vram_gb']:.2f} GB")

def main():
    print("="*60)
    print("PHASE 18.7 — SCIENTIFIC VALIDATION MATRIX")
    print("="*60)
    
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner(model, tokenizer)
    runner.run_matrix()

if __name__ == "__main__":
    start_wallclock = time.perf_counter()
    main()
    end_wallclock = time.perf_counter()
    
    with open("results/reconstruction_18_7/raw_wallclock_trace.log", "w") as f:
        f.write(f"START: {start_wallclock}\nEND: {end_wallclock}\nDURATION: {end_wallclock - start_wallclock}\n")
