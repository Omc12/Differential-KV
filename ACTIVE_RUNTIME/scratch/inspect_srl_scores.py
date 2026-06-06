import asyncio
import sys
import os
import torch
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def inspect_scores():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from native_core.srl.chunk_descriptor import compute_query_descriptor
    from native_core.srl.query_router import adaptive_k

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    session_id = "test_inspect"
    wrapper.manager.init_session(session_id, prefill_len=3000)
    
    tokenizer = wrapper.tokenizer
    
    secret_info = "The secret code word is: ALBATROSS. Remember this secret word.\n\n"
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. "
    )
    prompt = secret_info + (filler * 80)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    
    print(f"Prompt length: {len(input_ids)} tokens")
    
    wrapper.model._diffkv_session_ids = [session_id]
    wrapper.manager.register_prefill_tokens(session_id, input_ids.cpu())
    
    with torch.no_grad():
        _ = wrapper.model(
            input_ids=input_ids.unsqueeze(0).to(device),
            position_ids=torch.arange(len(input_ids), device=device).unsqueeze(0),
            use_cache=True
        )
    
    print("Waiting for SVD background compression threads...")
    t_start = time.time()
    while True:
        wrapper.manager.finalize_compressed_blocks()
        pending = getattr(wrapper.manager, "_pending_cpu_blocks", 0)
        if pending <= 0:
            break
        if time.time() - t_start > 30:
            print("Timeout waiting for compression!")
            break
        await asyncio.sleep(0.1)
        
    print(f"Compression finished. Finalizing SRL index...")
    wrapper.manager.finalize_srl_index(session_id)
    
    srl_state = wrapper.manager.get_srl_state(session_id)
    if srl_state is None:
        print("Error: srl_state is None!")
        return
        
    print(f"SRL Index built: active_blocks={srl_state.n_active_blocks()}")
    
    q1_text = "What is the secret code word?"
    q1_ids = tokenizer(q1_text, return_tensors="pt").input_ids[0].to(device)
    
    q2_text = "hi"
    q2_ids = tokenizer(q2_text, return_tensors="pt").input_ids[0].to(device)
    
    embed_tokens = wrapper.model.model.embed_tokens
    q_proj = wrapper.model.model.layers[0].self_attn.q_proj
    
    def get_query_states(token_ids):
        with torch.no_grad():
            hidden = embed_tokens(token_ids.unsqueeze(0))
            q_states = q_proj(hidden)
            num_heads = wrapper.model.config.num_attention_heads
            head_dim = wrapper.model.config.hidden_size // num_heads
            q_states = q_states.view(1, len(token_ids), num_heads, head_dim).transpose(1, 2).squeeze(0)
            return q_states
            
    q1_states = get_query_states(q1_ids)
    q2_states = get_query_states(q2_ids)
    
    # Let's inspect raw dot products with anchors_K
    # pool.anchors_K has shape [max_blocks, kv_heads, head_dim]
    pool = wrapper.manager.native_pool
    slot_tensor = torch.tensor(srl_state.ordered_slot_ids, dtype=torch.long, device=device)
    anc_K = pool.anchors_K[slot_tensor].float()  # [N, kv_heads, head_dim]
    anc_flat = anc_K.mean(dim=1)  # [N, head_dim]
    
    scale = 1.0 / math.sqrt(wrapper.model.config.hidden_size // wrapper.model.config.num_attention_heads)
    
    def print_anchor_scores(q_states, label):
        q_mean = q_states[:, -1, :].float().mean(dim=0)  # [head_dim]
        raw_scores = (anc_flat @ q_mean) * scale  # [N]
        
        print(f"\n--- Anchor scores for {label} ---")
        print(f"Max: {raw_scores.max().item():.4f}")
        print(f"Min: {raw_scores.min().item():.4f}")
        print(f"Mean: {raw_scores.mean().item():.4f}")
        print(f"Std: {raw_scores.std().item():.4f}")
        print(f"Range: {(raw_scores.max() - raw_scores.min()).item():.4f}")
        
        # Let's see how softmax behaves on raw anchor scores
        probs = torch.softmax(raw_scores, dim=0)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
        max_ent = math.log(len(raw_scores))
        complexity = min(entropy / max(max_ent, 1e-8), 1.0)
        print(f"Softmax entropy: {entropy:.4f} / max_ent: {max_ent:.4f} -> complexity: {complexity:.4f}")
        
    print_anchor_scores(q1_states, "Query 1 ('What is the secret code word?')")
    print_anchor_scores(q2_states, "Query 2 ('hi')")

if __name__ == "__main__":
    asyncio.run(inspect_scores())
