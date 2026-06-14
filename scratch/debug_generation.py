import os
import sys
import torch

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
_runtime_dir = os.path.join(_project_dir, "ACTIVE_RUNTIME")
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"

from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper
from native_core.mac_utils import get_best_device

def main():
    device = get_best_device()
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    
    wrapper = PyTorchDiffKVHFWrapper(
        MODEL,
        config={
            "rank": 16,
            "micro_block_size": 32,
            "serving_mode": "balanced",
            "local_files_only": True
        },
        device=device
    )
    
    # Configure overrides to match run_graph_analysis.py
    if wrapper.manager._streaming_mgr is not None:
        wrapper.manager._streaming_mgr.recency_window = 64
        wrapper.manager._streaming_mgr.short_context_threshold = 0
        wrapper.manager._streaming_mgr.protect_block_zero = True
        wrapper.manager._streaming_mgr._should_skip_compression = lambda *args, **kwargs: True
        
        from native_core.streaming_sparse_ingest import StreamingKVBlock
        StreamingKVBlock.short_context_threshold = 0
        StreamingKVBlock.protect_block_zero = True

    session_id = "debug-generation-session"
    wrapper.active_session = session_id
    
    content = "Degeneracies occur when two eigenvalues become equal, but not all degeneracies are alike. In Hermitian systems, a degeneracy does not destroy the independence of eigenvectors: although the eigenvalues coincide, the eigenvectors remain distinct and can still be chosen orthogonal. Such degeneracies are often represented geometrically by conical intersections, also called diabolical points, where two eigenvalue sheets touch without merging. The number of independent parameters that must be tuned to create a degeneracy is called its codimension. For real symmetric matrices this codimension is two, while for complex Hermitian matrices it is three, reflecting the additional constraint required to force the eigenvalues to coincide.\n\nNon-Hermitian systems exhibit a qualitatively different phenomenon known as an exceptional point. Here the degeneracy is stronger: not only do the eigenvalues become equal, but the eigenvectors themselves coalesce into a single state, causing the matrix to become defective and lose a complete eigenbasis. Exceptional points typically have codimension two and possess a square-root branch-point topology. The eigenvalue surfaces therefore form two connected Riemann sheets rather than a simple double cone. Encircling an exceptional point once exchanges the eigenvalues and associated states, so two loops are required to return to the original branch. Because non-Hermitian operators are not generally equal to their adjoints, left and right eigenvectors must be treated separately; near an exceptional point they exhibit biorthogonal behavior and can become self-orthogonal."
    
    question = "Using only the information above, answer the following:\n\n1. Define codimension and state the codimensions of:\n\n   * real symmetric degeneracies,\n   * complex Hermitian degeneracies,\n   * exceptional points."
    
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    prompt1 = wrapper.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    
    print("\n--- Running Prompt 1 ---")
    response1 = wrapper.generate(prompt1, max_new_tokens=10, temperature=0.0)
    print(f"Response 1: {response1}")
    
    assistant_reply = response1.split("assistant")[-1].strip()
    print(f"Extracted assistant reply: {repr(assistant_reply)}")
    
    messages2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": question}
    ]
    prompt2 = wrapper.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    
    print("\n--- Decoding Step-by-Step for Prompt 2 ---")
    # Instead of generate(), we do manual stepping
    inputs = wrapper.tokenizer(prompt2, return_tensors='pt').to(wrapper.device)
    prompt_ids = inputs.input_ids[0].tolist()
    
    cached_len = 0
    seq_len = wrapper.manager.get_session_sequence_length(session_id)
    if seq_len > 0 and seq_len < len(prompt_ids):
        stored_ids = getattr(wrapper, "_session_token_ids", {}).setdefault(session_id, [])
        if len(stored_ids) >= seq_len and prompt_ids[:seq_len] == stored_ids[:seq_len]:
            cached_len = seq_len
            print(f"Reusing KV cache! Length {cached_len}")
            
    if cached_len == 0:
        wrapper.manager.clear_session(session_id)
        wrapper._session_token_ids = {session_id: []}
        new_prompt_ids = prompt_ids
    else:
        new_prompt_ids = prompt_ids[cached_len:]
        
    input_ids = torch.tensor([new_prompt_ids], device=wrapper.device)
    prefill_len = input_ids.shape[1]
    generated = prompt_ids.copy()
    
    wrapper.manager.init_session(session_id, prefill_len=cached_len + prefill_len)
    if hasattr(wrapper.manager, "register_prefill_tokens"):
        wrapper.manager.register_prefill_tokens(session_id, torch.tensor(new_prompt_ids, dtype=torch.long))
    wrapper.model._diffkv_session_ids = [session_id]
    
    # Process prefill chunks
    PREFILL_CHUNK = 512
    for i in range(0, len(new_prompt_ids), PREFILL_CHUNK):
        chunk = new_prompt_ids[i:i + PREFILL_CHUNK]
        pos_ids = torch.arange(cached_len + i, cached_len + i + len(chunk), dtype=torch.long, device=wrapper.device).unsqueeze(0)
        chunk_tensor = torch.tensor([chunk], dtype=torch.long, device=wrapper.device)
        with torch.no_grad():
            outputs = wrapper.model(input_ids=chunk_tensor, position_ids=pos_ids, use_cache=True)
            
    if hasattr(wrapper.manager, "finalize_compressed_blocks"):
        wrapper.manager.finalize_compressed_blocks()
    if hasattr(wrapper.manager, "finalize_srl_index"):
        wrapper.manager.finalize_srl_index(session_id, cached_len=cached_len)
        
    srl_state = getattr(wrapper.manager, "_session_srl", {}).get(session_id)
    if srl_state is not None:
        srl_state.vsl_active_candidates = []
        srl_state.vsl_consecutive_helpers = 0
        srl_state.factual_anchor_q = None
        srl_state.current_entity_id = -1
        
    logits = outputs.logits[:, -1, :]
    past_kv = outputs.past_key_values
    cur_pos = cached_len + prefill_len
    
    print(f"Prefill done. Starting generation loop from pos {cur_pos}...")
    
    for step in range(200):
        # Apply factual biases
        sfa_active = (
            srl_state is not None
            and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.3
            and bool(getattr(srl_state, "current_step_factual_sequences", None))
        )
        
        sim = getattr(srl_state, "current_step_max_similarity", 0.0) if srl_state else 0.0
        print(f"\nStep {step}: pos={cur_pos} | sfa_active={sfa_active} | max_similarity={sim:.4f}")
        
        if sfa_active:
            from native_core.srl.factual_alignment import get_allowed_tokens_vsl, get_structural_helper_token_ids, get_helper_token_ids
            helper_ids = get_helper_token_ids(wrapper.tokenizer)
            structural_helper_ids = get_structural_helper_token_ids(wrapper.tokenizer)
            allowed_ids = get_allowed_tokens_vsl(
                srl_state, helper_ids,
                structural_helper_ids=structural_helper_ids,
                sfa_active=True
            )
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[list(allowed_ids)] = False
            
            if sim >= 0.70:
                logits[0, mask] = -65000.0
                print(f"  Hard masking applied: allowed_count={len(allowed_ids)}")
            else:
                logits[0, mask] -= 7.0
                print(f"  Soft penalty applied: allowed_count={len(allowed_ids)}")
                
        # Sample
        effective_temperature = 0.0
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.argmax(probs, dim=-1).unsqueeze(0)
        next_id_val = next_id.item()
        next_text = wrapper.tokenizer.decode([next_id_val])
        print(f"  Sampled token ID: {next_id_val} -> {repr(next_text)}")
        
        # Stop check
        stop_generation = False
        if srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.5:
            if getattr(srl_state, "current_step_factual_sequences", None):
                for seq in srl_state.current_step_factual_sequences:
                    if len(seq) >= 5 and next_id_val == seq[-1]:
                        stop_generation = True
                        print(f"  Triggered factual stop on sequence: {wrapper.tokenizer.decode(seq)}")
                        break
        
        if stop_generation:
            print("  Halting due to factual stop condition.")
            break
            
        if next_id_val in wrapper.stop_token_ids:
            print("  Halting due to stop token.")
            break
            
        generated.append(next_id_val)
        if hasattr(wrapper.manager, "register_prefill_tokens"):
            wrapper.manager.register_prefill_tokens(session_id, torch.tensor([next_id_val], dtype=torch.long))
            
        if sfa_active and srl_state is not None:
            from native_core.srl.factual_alignment import update_vsl_state, get_helper_token_ids
            helper_ids = get_helper_token_ids(wrapper.tokenizer)
            update_vsl_state(next_id_val, srl_state, helper_ids)
            print(f"  Updated VSL state: consecutive_helpers={srl_state.vsl_consecutive_helpers} | active_locks={len(srl_state.vsl_active_candidates)}")
            if srl_state.vsl_consecutive_helpers >= 16:
                print("  Halting due to 16 consecutive helpers.")
                break
                
        pos_tensor = torch.tensor([[cur_pos]], device=wrapper.device)
        input_ids = next_id
        with torch.no_grad():
            outputs = wrapper.model(input_ids=input_ids, position_ids=pos_tensor, use_cache=True)
        logits = outputs.logits[:, -1, :]
        cur_pos += 1

if __name__ == "__main__":
    main()
