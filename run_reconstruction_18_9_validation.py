import torch
import os
import time
import json
import numpy as np
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.long_context_workload_suite import LongContextWorkloadSuite
from runtime.hierarchical_memory_resolver import HierarchicalMemoryResolver
from runtime.persistent_relevance_resolver import PersistentRelevanceResolver
from runtime.anchor_reinforcement_resolver import AnchorReinforcementResolver
from runtime.hybrid_memory_resolver import HybridMemoryResolver
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner18_9:
    def __init__(self, model, tokenizer, results_dir="results/reconstruction_18_9/"):
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.suite = LongContextWorkloadSuite(tokenizer)
        self.overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        modes = ["dense", "sparse_baseline", "hmc_18_7", "prmrs_18_8", "arrsbs_18_9"]
        
        # Standardized Test Identifier for Phase 18.9
        test_id = "ALPHA-9-STABLE-RECONSTRUCTION-18.9"

        for ctx_len in contexts:
            for mode in modes:
                print(f"\n[RUN] Mode: {mode} | Context: {ctx_len} | ID: {test_id}")
                result = self.execute_single_run(mode, ctx_len, test_id)
                self.log_result(mode, ctx_len, result)

    def execute_single_run(self, mode, ctx_len, needle_str):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        resolver = None
        if mode == "sparse_baseline":
            resolver = HybridMemoryResolver(anchor_budget=ctx_len // 2)
            resolver.fidelity.threshold = 100.0 
        elif mode == "hmc_18_7":
            resolver = HierarchicalMemoryResolver(anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        elif mode == "prmrs_18_8":
            resolver = PersistentRelevanceResolver(anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        elif mode == "arrsbs_18_9":
            resolver = AnchorReinforcementResolver(self.tokenizer, anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        
        test_case = self.suite.create_needle_in_haystack(
            ctx_len, 
            needle=f"The secret activation code for the Differential KV project is: {needle_str}. This is stored in the section marked 'Context:'.", 
            answer=needle_str,
            needle_pos_ratio=0.4
        )
        input_ids = torch.tensor([test_case['tokens']]).to("cuda")
        
        past_key_values = DynamicCache()
        chunk_size = 512
        chunks = self.overlap_scheduler.get_chunks(input_ids, chunk_size)
        
        start_time = time.perf_counter()
        
        # Prefill Phase
        for chunk, _, _ in chunks:
            with torch.no_grad():
                outputs = self.model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            if mode != "dense":
                legacy = past_key_values.to_legacy_cache()
                pruned, meta = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                past_key_values = DynamicCache.from_legacy_cache(pruned)
                
                if mode == "arrsbs_18_9":
                    self._log_raw_arrsbs(ctx_len, meta)
        
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
                logits = outputs.logits[:, -1, :] / 0.7
                token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
                response_tokens.append(token.item())
                curr_input = token
                new_text = self.tokenizer.decode([token.item()])
                generated_text += new_text
                if self.tokenizer.eos_token in new_text: break
        
        gen_end = time.perf_counter()
        tps = len(response_tokens) / (gen_end - gen_start) if (gen_end - gen_start) > 0 else 0
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Success Metrics
        exact_match = needle_str in generated_text
        # Partial match: middle part of the identifier survives
        middle_part = needle_str[len(needle_str)//4 : -len(needle_str)//4]
        partial_match = middle_part in generated_text if len(middle_part) > 2 else False
        # Prefix: first 4-8 chars
        prefix_str = needle_str[:8]
        prefix_match = prefix_str in generated_text
        # Suffix: last 4-8 chars
        suffix_str = needle_str[-8:]
        suffix_match = suffix_str in generated_text
        
        return {
            "success": exact_match,
            "partial": partial_match,
            "prefix": prefix_match,
            "suffix": suffix_match,
            "tps": tps,
            "ttft": ttft,
            "vram_gb": vram,
            "output": generated_text,
            "expected": needle_str
        }

    def _log_raw_arrsbs(self, ctx_len, meta):
        with open(os.path.join(self.results_dir, "raw_anchor_registry.jsonl"), "a") as f:
            f.write(json.dumps({"ctx": ctx_len, "anchors": meta.get('anchors_detected', 0)}) + "\n")
        with open(os.path.join(self.results_dir, "raw_boundary_reinforcement.jsonl"), "a") as f:
            f.write(json.dumps({"ctx": ctx_len, "reinforced": meta.get('reinforced_spans', []), "budget": meta.get('budget_utilization', 0)}) + "\n")

    def log_result(self, mode, ctx, res):
        log_file = os.path.join(self.results_dir, "raw_validation_matrix.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx, **res}) + "\n")
        print(f"  [RESULT] EM: {res['success']} | Prefix: {res['prefix']} | Suffix: {res['suffix']} | TPS: {res['tps']:.2f}")

def main():
    print("="*60)
    print("PHASE 18.9 — ANCHOR-RELATIVE REINFORCEMENT (ARRSBS)")
    print("="*60)
    
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner18_9(model, tokenizer)
    runner.run_matrix()

if __name__ == "__main__":
    start_wallclock = time.perf_counter()
    main()
    end_wallclock = time.perf_counter()
    
    if not os.path.exists("results/reconstruction_18_9/"):
        os.makedirs("results/reconstruction_18_9/")
        
    with open("results/reconstruction_18_9/raw_wallclock_trace.log", "w") as f:
        f.write(f"START: {start_wallclock}\nEND: {end_wallclock}\nDURATION: {end_wallclock - start_wallclock}\n")
