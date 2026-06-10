import os
import sys
import torch
import math

os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from native_core.compression.lowrank import compress_lowrank

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 16, "micro_block_size": 16},
        device=device,
    )

    prompt = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. "
    ) * 10
    
    encoded = wrapper.tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        wrapper.model(**encoded, use_cache=True)
        cap_dict = getattr(wrapper.manager, "_prefill_kv_capture", {})
        session_id = list(cap_dict.keys())[0]
        K_raw, V_raw = cap_dict[session_id][5]
        
    # Block details
    anchor_idx = 100
    block_size = 32
    k_block = K_raw[0, :, anchor_idx : anchor_idx + block_size].float()
    v_block = V_raw[0, :, anchor_idx : anchor_idx + block_size].float()
    
    anchor_k = k_block[:, 0]
    anchor_v = v_block[:, 0]
    k_active = k_block[:, 1:]
    v_active = v_block[:, 1:]
    
    heads = k_block.shape[0]
    head_dim = k_block.shape[2]
    feat_dim = 2 * heads * head_dim
    
    # original deltas
    k_deltas = k_active.transpose(0, 1) - anchor_k.unsqueeze(0)
    v_deltas = v_active.transpose(0, 1) - anchor_v.unsqueeze(0)
    
    stacked = torch.stack([k_active.transpose(0, 1), v_active.transpose(0, 1)], dim=1)
    flat_tokens = stacked.reshape(block_size - 1, feat_dim)
    anchor_flat = torch.stack([anchor_k, anchor_v], dim=0).reshape(-1)
    deltas = flat_tokens - anchor_flat.unsqueeze(0)
    
    rank = 16
    
    # 1. Stacked SVD without normalization
    lr_none = compress_lowrank(deltas, rank)
    recon_none = (lr_none.U @ lr_none.V) * lr_none.scale
    recon_none_kv = recon_none.reshape(block_size-1, 2, heads, head_dim)
    recon_k_none = recon_none_kv[:, 0]
    recon_v_none = recon_none_kv[:, 1]
    err_k_none = (recon_k_none - k_deltas).norm() / k_deltas.norm()
    err_v_none = (recon_v_none - v_deltas).norm() / v_deltas.norm()
    
    # 2. Token-wise Norm-Normalization
    token_norms = deltas.norm(dim=1)
    token_norms = torch.clamp(token_norms, min=1e-5)
    normalized_deltas_tok = deltas / token_norms.unsqueeze(1)
    lr_tok = compress_lowrank(normalized_deltas_tok, rank)
    recon_tok = (lr_tok.U @ lr_tok.V) * lr_tok.scale * token_norms.unsqueeze(1)
    recon_tok_kv = recon_tok.reshape(block_size-1, 2, heads, head_dim)
    recon_k_tok = recon_tok_kv[:, 0]
    recon_v_tok = recon_tok_kv[:, 1]
    err_k_tok = (recon_k_tok - k_deltas).norm() / k_deltas.norm()
    err_v_tok = (recon_v_tok - v_deltas).norm() / v_deltas.norm()
    
    # 3. Channel-wise Norm-Normalization
    channel_norms = deltas.norm(dim=0)
    channel_norms = torch.clamp(channel_norms, min=1e-5)
    normalized_deltas_chan = deltas / channel_norms.unsqueeze(0)
    lr_chan = compress_lowrank(normalized_deltas_chan, rank)
    recon_chan = (lr_chan.U @ lr_chan.V) * lr_chan.scale * channel_norms.unsqueeze(0)
    recon_chan_kv = recon_chan.reshape(block_size-1, 2, heads, head_dim)
    recon_k_chan = recon_chan_kv[:, 0]
    recon_v_chan = recon_chan_kv[:, 1]
    err_k_chan = (recon_k_chan - k_deltas).norm() / k_deltas.norm()
    err_v_chan = (recon_v_chan - v_deltas).norm() / v_deltas.norm()
    
    print("Delta Reconstruction Errors (Rank 16, Block Size 32):")
    print("1. None Normalization:")
    print(f"   K error: {err_k_none.item():.4f}, V error: {err_v_none.item():.4f}")
    print("2. Token-wise Normalization:")
    print(f"   K error: {err_k_tok.item():.4f}, V error: {err_v_tok.item():.4f}")
    print("3. Channel-wise Normalization:")
    print(f"   K error: {err_k_chan.item():.4f}, V error: {err_v_chan.item():.4f}")
    
    wrapper.stop()

if __name__ == "__main__":
    main()
