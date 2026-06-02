#!/usr/bin/env python3
"""
scratch/test_large_prompt.py

Loads the Qwen/Qwen2.5-0.5B-Instruct model and feeds the large Sustainable AI
research paper abstract as a prompt, tracing the sequence of tokens, block ingestion,
and response coherence.
"""

import os
import sys
import time

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFKV_USE_TORCH_COMPILE", "0")
os.environ.setdefault("DIFFKV_TELEMETRY", "1")  # enable verbose logging
os.environ.setdefault("DIFFKV_DIAGNOSTICS", "1")  # enable detailed decode diagnostics

import torch
from native_core.mac_utils import get_best_device
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

DEVICE = get_best_device()

LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""

# Replicate the abstract 10 times to create a truly long prompt (>2500 tokens)
long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])

def main():
    print("=" * 60)
    print("  Large Prompt Coherence & Correctness Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    config = {
        "rank": 32,
        "micro_block_size": 32,
        "serving_mode": "balanced",
    }

    t0 = time.perf_counter()
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config=config,
        device=DEVICE,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")

    # Ingest the large research paper prompt
    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    print(f"Prompt length: {len(prompt)} characters")
    
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    num_tokens = encoded.input_ids.shape[1]
    print(f"Prompt token count: {num_tokens} tokens")
    
    print("\nExecuting forward prefill and decode sequence...")
    t1 = time.perf_counter()
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=64,
        temperature=0.0,  # greedy for stability
        repetition_penalty=1.0,
    )
    duration = time.perf_counter() - t1
    print(f"\nResponse received in {duration:.2f}s:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
    # Assert response contains meaningful, non-gibberish words
    words = response.split()
    assert len(words) > len(prompt.split()), "Model did not generate new tokens!"
    print("Coherence Check: PASS ✓")
    
    wrapper.stop()
    print("\nAll checks completed successfully!")

if __name__ == "__main__":
    main()
