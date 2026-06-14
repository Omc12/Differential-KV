import os
import sys
import torch

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_runtime_dir = os.path.dirname(_script_dir)
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "1"

from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper
from native_core.mac_utils import get_best_device

def main():
    device = get_best_device()
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading PyTorch wrapper for {MODEL} on {device}...")
    
    wrapper = PyTorchDiffKVHFWrapper(
        MODEL,
        config={
            "rank": 16,
            "micro_block_size": 32,
            "block_size": 32,
            "serving_mode": "balanced",
            "local_files_only": True
        },
        device=device
    )
    
    if wrapper.manager._streaming_mgr is not None:
        wrapper.manager._streaming_mgr.recency_window = 32
        wrapper.manager._streaming_mgr.short_context_threshold = 0
        wrapper.manager._streaming_mgr.protect_block_zero = True
        wrapper.manager._streaming_mgr._should_skip_compression = lambda *args, **kwargs: False
        
        from native_core.streaming_sparse_ingest import StreamingKVBlock
        StreamingKVBlock.short_context_threshold = 0
        StreamingKVBlock.protect_block_zero = True

    session_id = "diagnostic-sfa-session"
    wrapper.active_session = session_id
    
    content = """Degeneracies occur when two eigenvalues become equal, but not all degeneracies are alike. In Hermitian systems, a degeneracy does not destroy the independence of eigenvectors: although the eigenvalues coincide, the eigenvectors remain distinct and can still be chosen orthogonal. Such degeneracies are often represented geometrically by conical intersections, also called diabolical points, where two eigenvalue sheets touch without merging. The number of independent parameters that must be tuned to create a degeneracy is called its codimension. For real symmetric matrices this codimension is two, while for complex Hermitian matrices it is three, reflecting the additional constraint required to force the eigenvalues to coincide.

Non-Hermitian systems exhibit a qualitatively different phenomenon known as an exceptional point. Here the degeneracy is stronger: not only do the eigenvalues become equal, but the eigenvectors themselves coalesce into a single state, causing the matrix to become defective and lose a complete eigenbasis. Exceptional points typically have codimension two and possess a square-root branch-point topology. The eigenvalue surfaces therefore form two connected Riemann sheets rather than a simple double cone. Encircling an exceptional point once exchanges the eigenvalues and associated states, so two loops are required to return to the original branch. Because non-Hermitian operators are not generally equal to their adjoints, left and right eigenvectors must be treated separately; near an exceptional point they exhibit biorthogonal behavior and can become self-orthogonal."""

    question = """Using only the information above, answer the following:

1. Define codimension and state the codimensions of:

   * real symmetric degeneracies,
   * complex Hermitian degeneracies,
   * exceptional points.

2. Compare the topology of eigenvalue surfaces near:

   * a diabolical point,
   * an exceptional point.

3. Explain the difference between:

   * eigenvalue degeneracy,
   * eigenvector coalescence.

4. Why are exceptional points associated with Riemann sheets whereas diabolical points are associated with conical intersections?

5. What happens after one closed loop around:

   * a Hermitian degeneracy,
   * an exceptional point?

6. Why are left and right eigenvectors important in non-Hermitian systems but not in Hermitian ones?"""

    prompt1 = wrapper.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n--- Running Prompt 1 ---")
    response1 = wrapper.generate(prompt1, max_new_tokens=10, temperature=0.0)
    print(f"Response 1: {response1}")
    
    assistant_reply = response1.split("assistant")[-1].strip()
    prompt2 = wrapper.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_reply},
            {"role": "user", "content": question}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n--- Running Prompt 2 with custom step-by-step logging ---")
    # We do a step-by-step custom generation loop here
    input_ids = wrapper.tokenizer(prompt2, return_tensors="pt")["input_ids"].to(device)
    
    # Run prefill
    wrapper.manager.init_session(session_id, input_ids.shape[1])
    with torch.no_grad():
        out = wrapper.model(input_ids)
    
    # We now generate step-by-step
    generated = []
    logits = out.logits[:, -1, :]
    cur_pos = input_ids.shape[1]
    
    from native_core.srl.factual_alignment import get_helper_token_ids, get_allowed_tokens_vsl, get_structural_helper_token_ids
    
    for step in range(100):
        # Apply temperature scaling
        srl_state = wrapper.manager.get_srl_state(session_id)
        max_sim = getattr(srl_state, "current_step_max_similarity", 0.0) if srl_state else 0.0
        
        sfa_active = (
            srl_state is not None
            and max_sim >= 0.55
            and bool(getattr(srl_state, "current_step_factual_sequences", None))
        )
        
        # Apply VSL logit masking
        if sfa_active:
            helper_ids = get_helper_token_ids(wrapper.tokenizer)
            structural_helper_ids = get_structural_helper_token_ids(wrapper.tokenizer)
            allowed_ids = get_allowed_tokens_vsl(
                srl_state, helper_ids,
                structural_helper_ids=structural_helper_ids,
                sfa_active=True
            )
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[list(allowed_ids)] = False
            
            if max_sim >= 0.70:
                print(f"[Step {step}] Hard masking active! max_sim = {max_sim:.3f}")
                logits[0, mask] = -65000.0
            else:
                print(f"[Step {step}] Soft masking active! max_sim = {max_sim:.3f}")
                logits[0, mask] -= 7.0
                
        # Sample next token
        next_id = logits.argmax(dim=-1).item()
        next_tok_text = wrapper.tokenizer.decode([next_id])
        print(f"[Step {step}] Generated token ID: {next_id} -> {repr(next_tok_text)}")
        
        # Check factual early stopping
        stop_generation = False
        if srl_state is not None and max_sim >= 0.5:
            factual_seqs = getattr(srl_state, "current_step_factual_sequences", [])
            for seq in factual_seqs:
                if len(seq) >= 5 and len(generated) + 1 >= len(seq):
                    test_gen = generated + [next_id]
                    if test_gen[-len(seq):] == list(seq):
                        stop_generation = True
                        print(f"[Step {step}] Early stopping triggered on sequence: {wrapper.tokenizer.decode(seq)}")
                        break
        
        generated.append(next_id)
        if stop_generation:
            print(f"[Step {step}] Stopping because stop_generation = True")
            break
            
        if next_id in wrapper.stop_token_ids:
            print(f"[Step {step}] Stopping because EOT/EOS token")
            break
            
        # Update SRL state
        if sfa_active and srl_state is not None:
            from native_core.srl.factual_alignment import update_vsl_state
            helper_ids = get_helper_token_ids(wrapper.tokenizer)
            update_vsl_state(next_id, srl_state, helper_ids)
            
        # Forward next token
        next_id_tensor = torch.tensor([[next_id]], device=device)
        pos_tensor = torch.tensor([[cur_pos]], device=device)
        cur_pos += 1
        
        with torch.no_grad():
            out = wrapper.model(next_id_tensor, position_ids=pos_tensor)
        logits = out.logits[:, -1, :]
        
    print(f"\nFinal Generated Text: {wrapper.tokenizer.decode(generated)}")
    wrapper.close()

if __name__ == "__main__":
    main()
