import torch
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.long_context_workload_suite import LongContextWorkloadSuite
from analysis.retrieval_survival_mapper import RetrievalSurvivalMapper
from runtime.hybrid_memory_resolver import HybridMemoryResolver
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

def run_symbolic_benchmark():
    print("="*60)
    print("PHASE 18.6 — SYMBOLIC FIDELITY & EXACT RECOVERY")
    print("="*60)

    # 1. Initialization
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Clear previous results
    res_path = "results/reconstruction_18_6/raw_symbolic_retrieval.jsonl"
    if os.path.exists("results/reconstruction_18_6/"):
        for f in os.listdir("results/reconstruction_18_6/"):
            os.remove(os.path.join("results/reconstruction_18_6/", f))
    else:
        os.makedirs("results/reconstruction_18_6/")
    
    suite = LongContextWorkloadSuite(tokenizer)
    mapper = RetrievalSurvivalMapper(export_dir="results/reconstruction_18_6/")
    resolver = HybridMemoryResolver(anchor_budget=8192, fidelity_budget=1024)
    overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)
    
    contexts = [8192, 16384]
    
    for ctx_len in contexts:
        print(f"\n[RUN] Symbolic Fidelity Check - Context: {ctx_len}")
        resolver.geometry.reset_accumulation()
        resolver.global_offset = 0
        
        # --- TEST 1: Exact Identifier Recall (NIAH) ---
        niah_test = suite.create_needle_in_haystack(ctx_len, needle_pos_ratio=0.1)
        input_ids = torch.tensor([niah_test['tokens']]).to("cuda")
        
        past_key_values = DynamicCache()
        chunk_size = 512
        
        chunks = overlap_scheduler.get_chunks(input_ids, chunk_size)
        print(f"  [EXEC] Processing {len(chunks)} hybrid chunks...")
        
        for chunk, _, _ in chunks:
            with torch.no_grad():
                outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            # Hybrid Resolve and Prune
            legacy = past_key_values.to_legacy_cache()
            pruned, _ = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
            past_key_values = DynamicCache.from_legacy_cache(pruned)

        # Generate Answer with Repetition Penalty
        print(f"  [EXEC] Generating exact retrieval for {ctx_len} tokens...")
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        
        # Use slightly lower temperature (0.5) for symbolic precision
        for _ in range(64):
            with torch.no_grad():
                outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.5
                
                # Manual Repetition Penalty
                for token_id in response_tokens[-20:]:
                    logits[:, token_id] /= 1.3
                
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                
                response_tokens.append(token.item())
                curr_input = token
                new_text = tokenizer.decode([token.item()])
                generated_text += new_text
                if tokenizer.eos_token in new_text: break
        
        response_text = generated_text
        # EXACT MATCH CHECK
        success = niah_test['answer'] in response_text
        
        mapper.record_test(ctx_len, "SYMBOLIC_NIAH", success, response_text, niah_test['answer'])
        print(f"  [RESULT] Symbolic Success: {success} | Response: {repr(response_text.strip())}")

    print("\n[COMPLETE] Phase 18.6 Symbolic Validation Finished.")

if __name__ == "__main__":
    run_symbolic_benchmark()
