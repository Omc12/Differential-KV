import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""
    long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])
    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    
    print("1. Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
    )
    model_baseline.eval()
    
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Running baseline first step...")
    with torch.no_grad():
        outputs_baseline = model_baseline(**encoded, use_cache=True)
    logits_baseline = outputs_baseline.logits[0, -1, :].cpu().float()
    
    # Clean up baseline model to free memory
    del model_baseline
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    
    print("\n2. Loading patched model...")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )
    
    # We patch _compress_block_sync to use Channel Norm-Normalized SVD
    manager = wrapper.manager
    
    def patched_compress_block_sync(self, block, k, v):
        input_device = k.device
        if input_device.type == "cpu":
            anchor_kv_local = getattr(block, "anchor_kv_cpu", None)
            if anchor_kv_local is None:
                anchor_kv_local = block.anchor_kv.cpu()
        else:
            anchor_kv_local = block.anchor_kv
        anchor_flat = anchor_kv_local.reshape(-1).float().to(input_device)
        seq_len  = k.shape[2]
        heads    = k.shape[1]
        head_dim = k.shape[3]
        feat_dim = 2 * heads * head_dim

        stacked     = torch.stack([k[0].transpose(0, 1), v[0].transpose(0, 1)], dim=1)
        flat_tokens = stacked.reshape(seq_len, feat_dim).float()
        deltas      = flat_tokens - anchor_flat.unsqueeze(0)

        # Norm-Normalization
        channel_norms = deltas.norm(dim=0)
        channel_norms = torch.clamp(channel_norms, min=1e-5)
        normalized_deltas = deltas / channel_norms.unsqueeze(0)

        rank = self.rank
        lr_delta = compress_lowrank(normalized_deltas, rank)
        
        V_scaled = lr_delta.V.float() * channel_norms.unsqueeze(0)
        V_scaled = V_scaled.to(torch.float16)

        gpu_device = block.anchor_kv.device
        block.U          = lr_delta.U.to(gpu_device)
        block.V          = V_scaled.to(gpu_device)
        block.scale      = lr_delta.scale
        block.cosine_sim = lr_delta.cosine_sim
        block.norm_drift = lr_delta.norm_drift
        block.dynamic_rank = getattr(lr_delta, "dynamic_rank", self.rank)

        block.active_k = None
        block.active_v = None
        block.active_k_cpu = None
        block.active_v_cpu = None
        block.dirty    = True
        block.state = "COMPRESSED"

        session_id = getattr(block, 'session_id', None)
        session_active = True
        if session_id is not None:
            if self._streaming_mgr is not None:
                session_active = session_id in self._streaming_mgr.session_blocks
            else:
                session_active = session_id in self.session_blocks

        if session_active:
            if hasattr(self, 'native_pool') and self.native_pool is not None:
                try:
                    if getattr(block, 'pool_idx', None) is None:
                        block.pool_idx = self.native_pool.allocate_block()
                    self.native_pool.write_block(
                        pool_idx=block.pool_idx,
                        U=block.U,
                        V=block.V,
                        anchor_K=block.anchor_kv[0, 0],
                        anchor_V=block.anchor_kv[0, 1],
                        scale=block.scale,
                        seq_len=block.U.shape[0]
                    )
                except Exception as e:
                    print(f"[DiffKV] WARNING: Failed to write block to NativeBlockPool: {e}")

            if self._streaming_mgr is not None and getattr(block, 'session_id', None) is not None and getattr(block, 'layer_idx', None) is not None:
                self._streaming_mgr.update_metadata_state(block.session_id, block.layer_idx, block)

    import types
    manager._compress_block_sync = types.MethodType(patched_compress_block_sync, manager)
    
    print("Running patched model first step...")
    with torch.no_grad():
        # Dry run wrapper generate to trigger first step logits capture
        session_id = "default"
        wrapper.manager.clear_session(session_id)
        wrapper.manager.init_session(session_id, prefill_len=encoded.input_ids.shape[1])
        wrapper.model._diffkv_session_ids = [session_id]
        
        outputs_patched = wrapper.model(**encoded, use_cache=True)
        wrapper.manager.compress_prefill_kv(session_id)
        
    logits_patched = outputs_patched.logits[0, -1, :].cpu().float()
    
    # Print comparison metrics
    diff = (logits_patched - logits_baseline).abs()
    print("\nLogit Comparison at First Decode Step:")
    print(f"  Max absolute difference: {diff.max().item():.6f}")
    print(f"  Mean absolute difference: {diff.mean().item():.6f}")
    print(f"  Top-1 token ID - Baseline: {logits_baseline.argmax().item()}, Patched: {logits_patched.argmax().item()}")
    
    # Print top 5 token IDs for both
    val_b, idx_b = torch.topk(logits_baseline, k=5)
    val_p, idx_p = torch.topk(logits_patched, k=5)
    print("\nTop 5 Tokens (Baseline):")
    for i in range(5):
        print(f"  {tokenizer.decode([idx_b[i].item()]):<15} (id={idx_b[i].item()}): logit={val_b[i].item():.4f}")
    print("Top 5 Tokens (Patched):")
    for i in range(5):
        print(f"  {tokenizer.decode([idx_p[i].item()]):<15} (id={idx_p[i].item()}): logit={val_p[i].item():.4f}")
        
    wrapper.stop()

if __name__ == "__main__":
    main()
