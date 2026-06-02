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
    
    print("\n2. Loading patched model with current active runtime configuration...")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )
    
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
