import os
import sys
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True
    )
    
    prompt_file = "/Users/omchimurkar1/Desktop/Differential-KV/benchmarks/results/prompt_4096.txt"
    with open(prompt_file, "r") as f:
        prompt_text = f.read()
        
    ids = tok.encode(prompt_text)[:512]
    ct = torch.tensor([ids], dtype=torch.long)
    
    captured = {}
    def hook_q(module, input, output):
        captured["q"] = output
    def hook_k(module, input, output):
        captured["k"] = output
        
    h_q = model.model.layers[0].self_attn.q_proj.register_forward_hook(hook_q)
    h_k = model.model.layers[0].self_attn.k_proj.register_forward_hook(hook_k)
    
    with torch.no_grad():
        model(ct)
        
    h_q.remove()
    h_k.remove()
    
    q = captured["q"]
    k = captured["k"]
    self_attn = model.model.layers[0].self_attn
    
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    
    bsz, q_len, _ = q.shape
    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Check rotary_emb on model or model.model
    rotary_emb = getattr(self_attn, "rotary_emb", getattr(model.model, "rotary_emb", None))
    cos, sin = rotary_emb(k, position_ids=torch.arange(q_len).unsqueeze(0))
    
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    def apply_rope(x, cos, sin):
        return x * cos + rotate_half(x) * sin
        
    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)
    
    g = num_heads // num_kv_heads
    k_rot = k_rot.repeat_interleave(g, dim=1)
    
    scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) / math.sqrt(head_dim)
    mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=q.device), diagonal=1)
    scores = scores + mask.unsqueeze(0).unsqueeze(0)
    
    lse = torch.logsumexp(scores.float(), dim=-1)
    
    scores = scores.float()
    lse = lse.float()
    
    print("Standard HF Layer 0 Attention Scores min/max:", scores[torch.isfinite(scores)].min().item(), scores[torch.isfinite(scores)].max().item())
    print("Standard HF Layer 0 LogSumExp min/max:", lse.min().item(), lse.max().item())

if __name__ == "__main__":
    main()
