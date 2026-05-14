import torch
import json
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.long_context_workload_suite import LongContextWorkloadSuite
from analysis.retrieval_survival_mapper import RetrievalSurvivalMapper
from memory.semantic_geometry_tracker import SemanticGeometryKVManager
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

def run_continuity_benchmark():
    print("="*60)
    print("PHASE 18.5 — SEMANTIC GEOMETRY & CONTINUITY")
    print("="*60)

    # 1. Initialization
    loader = Qwen7BRealLoader()
    model = loader.load(attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Clear previous results for scientific integrity
    res_path = "results/reconstruction_18_5/raw_retrieval_accuracy.jsonl"
    if os.path.exists(res_path): os.remove(res_path)
    
    suite = LongContextWorkloadSuite(tokenizer)
    mapper = RetrievalSurvivalMapper(export_dir="results/reconstruction_18_5/")
    kv_manager = SemanticGeometryKVManager(anchor_budget=6144, neighborhood_size=16)
    overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)
    
    contexts = [8192, 16384] # Focus on the failure zone
    
    for ctx_len in contexts:
        print(f"\n[RUN] Continuity Check - Context: {ctx_len}")
        kv_manager.reset_accumulation()
        
        # --- TEST 1: Needle In A Haystack (Grounded) ---
        niah_test = suite.create_needle_in_haystack(ctx_len, needle_pos_ratio=0.1)
        input_ids = torch.tensor([niah_test['tokens']]).to("cuda")
        
        past_key_values = DynamicCache()
        chunk_size = 512
        
        # Overlapping Prefill with Neighborhood Pruning
        chunks = overlap_scheduler.get_chunks(input_ids, chunk_size)
        print(f"  [EXEC] Processing {len(chunks)} overlapping chunks...")
        
        for chunk, stride_start, stride_len in chunks:
            with torch.no_grad():
                # We process the chunk, then prune the KV cache
                outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            # Prune/Sync
            legacy = past_key_values.to_legacy_cache()
            pruned, _ = kv_manager.prune_kv(legacy, hidden_states=outputs.hidden_states[-1])
            past_key_values = DynamicCache.from_legacy_cache(pruned)

        # Generate Answer with Nucleus Sampling
        print(f"  [EXEC] Generating retrieval response for {ctx_len} tokens...")
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        
        for _ in range(64): # Longer response for multi-hop
            with torch.no_grad():
                outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7
                
                # Top-p Sampling
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
                if tokenizer.eos_token in new_text: break
        
        response_text = generated_text
        success = niah_test['answer'].lower() in response_text.lower()
        
        mapper.record_test(ctx_len, "NIAH", success, response_text, niah_test['answer'])
        print(f"  [RESULT] NIAH Success: {success} | Response: {response_text.strip()}")

        # --- TEST 2: Instruction Persistence ---
        instr_test = suite.create_instruction_persistence_test(ctx_len)
        input_ids = torch.tensor([instr_test['tokens']]).to("cuda")
        kv_manager.reset_accumulation()
        past_key_values = DynamicCache()
        
        chunks = overlap_scheduler.get_chunks(input_ids, chunk_size)
        for chunk, _, _ in chunks:
            with torch.no_grad():
                outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            legacy = past_key_values.to_legacy_cache()
            pruned, _ = kv_manager.prune_kv(legacy, hidden_states=outputs.hidden_states[-1])
            past_key_values = DynamicCache.from_legacy_cache(pruned)

        print(f"  [EXEC] Generating instruction check for {ctx_len} tokens...")
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        for _ in range(64):
            with torch.no_grad():
                outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                response_tokens.append(token.item())
                curr_input = token
                new_text = tokenizer.decode([token.item()])
                generated_text += new_text
                if tokenizer.eos_token in new_text: break
        
        response_text = generated_text
        success = all(req.lower() in response_text.lower() for req in instr_test['requirements'])
        
        mapper.record_test(ctx_len, "Instruction", success, response_text, str(instr_test['requirements']))
        print(f"  [RESULT] Instruction Persistence: {success} | Response: {response_text.strip()}")

    print("\n[COMPLETE] Phase 18.5 Continuity Validation Finished.")

if __name__ == "__main__":
    run_continuity_benchmark()
