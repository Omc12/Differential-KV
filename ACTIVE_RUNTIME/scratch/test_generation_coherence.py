import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.hf_diffkv_wrapper import DiffKVHFWrapper

def diagnose_generation():
    print("=" * 60)
    print("  DiffKV Generation Coherence Diagnostics")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # We load Qwen2.5-0.5B-Instruct
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32, "serving_mode": "balanced"},
        device=device,
    )

    prompt = (
        "Abstract\n"
        "While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) "
        "it is time to move beyond that and to address the sustainability of developing and using AI systems. "
        "In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in "
        "the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, "
        "and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two "
        "categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI "
        "(sustainable development, training, and use of AI). The focus of this paper is on the latter.\n"
    ) * 6  # Replicate 6 times for ~1500 tokens (enough to trigger SVD on several blocks)

    prompt += "\nQuestion: What are the two categories of Sustainable AI? Answer in exactly one short sentence."
    
    print(f"Prompt length: {len(prompt)} characters")
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    num_tokens = encoded.input_ids.shape[1]
    print(f"Prompt tokens: {num_tokens} tokens")

    # Generate response
    print("\nGenerating response...")
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.0,
    )
    print(f"\nGenerated Response: {repr(response)}")

    # Print summary metrics of blocks
    print("\nLayer-by-Layer Block Analysis:")
    manager = wrapper.manager
    for layer_idx in range(manager.num_layers):
        blocks = manager.get_streaming_blocks("default", layer_idx)
        if not blocks:
            continue
        compressed_blocks = [b for b in blocks if getattr(b, "state", None) == "COMPRESSED"]
        print(f"  Layer {layer_idx}: {len(blocks)} total blocks, {len(compressed_blocks)} compressed blocks")
        for i, b in enumerate(compressed_blocks):
            # Print block info
            print(f"    Block {i} (anchor={b.anchor_idx}): state={b.state}, scale={b.scale:.4f}, rank={b.dynamic_rank}, cos_sim={b.cosine_sim:.4f}, drift={b.norm_drift:.4f}")

    wrapper.stop()

if __name__ == "__main__":
    diagnose_generation()
