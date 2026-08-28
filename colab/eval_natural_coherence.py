"""Coherence and Natural Language Quality Evaluation for DKV (FP16 vs INT4).

Evaluates whether model outputs are grammatically natural, semantically coherent,
and factually faithful with no phantom tokens, repetition loops, or degradations.
"""
import os
import sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
from serving.decode_config import BEST_DECODE_DEFAULTS

PROMPTS = [
    {
        "category": "Conceptual Reasoning",
        "system": "You are an expert AI systems engineer. Give clear, insightful, and natural explanations.",
        "user": "Explain in two clear paragraphs why low-rank key-value cache compression works well in large language models, and why residual token preservation is crucial for avoiding attention degradation."
    },
    {
        "category": "Structured Synthesis & Analysis",
        "system": "You are an analytical assistant. Summarize key technical insights into clear bullet points.",
        "user": (
            "Context: Project Differential KV replaces standard uniform KV caching with adaptive low-rank factorization. "
            "Instead of storing raw FP16 keys and values for every past token, each block is decomposed into a shared anchor "
            "and low-rank projection matrices (U and V). High-norm residual tokens that deviate from the low-rank subspace "
            "are preserved explicitly. Recent improvements introduce group-quantized 4-bit residual storage and rarity-aware "
            "token selection based on inverse document frequency.\n\n"
            "Question: Based on the context, what are the three core architectural pillars of Differential KV, and how do they balance memory efficiency with retrieval fidelity?"
        )
    },
    {
        "category": "Multi-Step Problem Solving",
        "system": "You are a helpful and precise assistant. Solve the problem step-by-step.",
        "user": "A GPU with 16 GB VRAM allocates 8 GB for model weights and 2 GB for activations. A standard FP16 KV cache consumes 256 bytes per token. If Differential KV compresses the KV cache by 3.5x, calculate how many tokens of context can fit in the remaining memory for both standard and Differential KV. Show your calculations clearly."
    }
]


def test_coherence(model_id="Qwen/Qwen3.5-2B", quant_modes=("none", "int4")):
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)

    for quant in quant_modes:
        print("\n" + "=" * 90)
        print(f"RUNNING COHERENCE & NATURALNESS EVALUATION: MODE = {quant.upper()}")
        print("=" * 90)

        os.environ["DKV_RESIDUAL_QUANT"] = quant
        os.environ["DKV_MAX_RESIDUAL_TOKENS"] = "80"

        wrapper = PyTorchDKVHFWrapper(
            model_id=model_id,
            config={"mode": "fp16", "residual_quant": quant, "max_residual_tokens": 80},
            device="cuda"
        )
        wrapper.ensure_loaded()

        for idx, p in enumerate(PROMPTS, 1):
            print(f"\n--- [Test {idx}: {p['category']} | Quant: {quant.upper()}] ---")
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]}
            ]
            prompt = wrapper.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Generate response
            with torch.inference_mode():
                out = wrapper.generate(prompt, max_new_tokens=220, temperature=0.0)

            # Extract assistant response
            if "<|im_start|>assistant" in out:
                reply = out.split("<|im_start|>assistant")[-1].strip()
            elif "</think>" in out:
                reply = out.split("</think>")[-1].strip()
            else:
                reply = out[len(prompt):].strip()

            print(reply)
            print("-" * 90)

        del wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_coherence()
