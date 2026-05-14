import torch
import json
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.long_context_workload_suite import LongContextWorkloadSuite
from analysis.retrieval_survival_mapper import RetrievalSurvivalMapper
from memory.real_sparse_kv_manager import RealSparseKVManager
from transformers import AutoTokenizer, DynamicCache

def run_integrity_benchmark():
    print("="*60)
    print("PHASE 18.4 — CONTEXT INTEGRITY & RETRIEVAL SURVIVAL")
    print("="*60)

    # 1. Initialization
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    suite = LongContextWorkloadSuite(tokenizer)
    mapper = RetrievalSurvivalMapper()
    kv_manager = RealSparseKVManager(anchor_budget=4096)
    
    contexts = [4096, 8192, 16384]
    
    for ctx_len in contexts:
        print(f"\n[RUN] Integrity Check - Context: {ctx_len}")
        kv_manager.reset_accumulation()
        
        # --- TEST 1: Needle In A Haystack ---
        niah_test = suite.create_needle_in_haystack(ctx_len, needle_pos_ratio=0.1)
        input_ids = torch.tensor([niah_test['tokens']]).to("cuda")
        
        past_key_values = DynamicCache()
        chunk_size = 512
        
        # Chunked Prefill with Pruning
        for i in range(0, ctx_len, chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            with torch.no_grad():
                outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            # Synchronize importance and prune if necessary
            legacy = past_key_values.to_legacy_cache()
            pruned, _ = kv_manager.prune_kv(legacy, hidden_states=outputs.hidden_states[-1])
            past_key_values = DynamicCache.from_legacy_cache(pruned)

        # Generate Answer with Production Parameters (Manual Loop)
        print(f"  [EXEC] Generating retrieval response for {ctx_len} tokens...")
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        
        for _ in range(32):
            with torch.no_grad():
                outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7 # Temperature
                
                # Apply Repetition Penalty (Simple)
                for token_id in response_tokens:
                    logits[:, token_id] /= 1.1
                
                # Top-p (Nucleus) Sampling
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > 0.9
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                
                response_tokens.append(token.item())
                curr_input = token
                
                new_text = tokenizer.decode([token.item()])
                generated_text += new_text
                if tokenizer.eos_token in new_text:
                    break
        
        response_text = generated_text
        success = niah_test['answer'].lower() in response_text.lower()
        
        mapper.record_test(ctx_len, "NIAH", success, response_text, niah_test['answer'])
        print(f"  [RESULT] NIAH Success: {success} | Response: {response_text.strip()}")

        # --- TEST 2: Instruction Persistence ---
        instr_test = suite.create_instruction_persistence_test(ctx_len)
        input_ids = torch.tensor([instr_test['tokens']]).to("cuda")
        kv_manager.reset_accumulation()
        
        past_key_values = DynamicCache() # Reset cache
        for i in range(0, ctx_len, chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            with torch.no_grad():
                outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            legacy = past_key_values.to_legacy_cache()
            pruned, _ = kv_manager.prune_kv(legacy, hidden_states=outputs.hidden_states[-1])
            past_key_values = DynamicCache.from_legacy_cache(pruned)

        print(f"  [EXEC] Generating instruction check for {ctx_len} tokens...")
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        
        for _ in range(32):
            with torch.no_grad():
                outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7
                
                for token_id in response_tokens:
                    logits[:, token_id] /= 1.1
                
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                
                response_tokens.append(token.item())
                curr_input = token
                
                new_text = tokenizer.decode([token.item()])
                generated_text += new_text
                if tokenizer.eos_token in new_text:
                    break
        
        response_text = generated_text
        success = all(req.lower() in response_text.lower() for req in instr_test['requirements'])
        
        mapper.record_test(ctx_len, "Instruction", success, response_text, str(instr_test['requirements']))
        print(f"  [RESULT] Instruction Persistence: {success} | Response: {response_text.strip()}")

    # 2. Final Report Generation
    generate_final_integrity_report(mapper.results)

def generate_final_integrity_report(results):
    path = "results/reconstruction_18_4/reconstruction_18_4_context_integrity.md"
    with open(path, 'w') as f:
        f.write("# PHASE 18.4 — CONTEXT INTEGRITY & RETRIEVAL SURVIVAL\n\n")
        f.write("## [MEASURED] Retrieval Continuity Baseline\n\n")
        f.write("| Context | Test Type | Status | Expected | Response |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            status = "PASSED" if r['success'] else "FAILED"
            f.write(f"| {r['context_len']} | {r['test_type']} | {status} | {r['ground_truth']} | {r['response'][:50]}... |\n")
        
        f.write("\n## Scientific Conclusion\n")
        f.write("This table documents the semantic survival of long-context information under DiffKV sparse execution.\n")

if __name__ == "__main__":
    run_integrity_benchmark()
