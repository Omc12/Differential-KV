import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )
    
    # Check rotary embedding module
    rotary_emb = wrapper.model.model.rotary_emb
    print("Rotary embedding module:", rotary_emb)
    
    # Dry run rotary embedding
    v = torch.randn(1, 2, 2, 64, device=device) # [bsz, seq_len, heads, dim]
    pos = torch.arange(10, device=device).unsqueeze(0)
    cos, sin = rotary_emb(v, pos)
    print("cos shape:", cos.shape)
    print("sin shape:", sin.shape)
    print("cos dim:", cos.dim())
    
    wrapper.stop()

if __name__ == "__main__":
    main()
