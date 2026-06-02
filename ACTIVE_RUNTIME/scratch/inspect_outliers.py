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
    
    LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""
    long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])
    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    
    print("Ingesting prompt...")
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    
    with torch.no_grad():
        _ = wrapper.model(input_ids=input_ids, use_cache=True)
        
    sid = "default"
    cap_dict = getattr(wrapper.manager, "_prefill_kv_capture", {})
    print("\nKey Activations Analysis (All vs. Normal range 100-400):")
    for layer in range(len(cap_dict[sid])):
        k_cap, _ = cap_dict[sid][layer]
        k_cap = k_cap[0].cpu().float()  # [heads, seq_len, dim]
        total_max = k_cap.abs().max().item()
        normal_max = k_cap[:, 100:400, :].abs().max().item()
        print(f"  Layer {layer:2d}: total_max={total_max:8.4f}  |  normal_max={normal_max:8.4f}")
        
    wrapper.stop()

if __name__ == "__main__":
    main()
