#!/usr/bin/env python3
import sys
import os
import time
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

import mlx.core as mx
import mlx.nn as nn
from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
import torch

SAMPLE_TEXT = """
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. 
The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. 
This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. 
This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. 

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA, in the summer of 1956. 
Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and millions of dollars were given to make this vision come true. 
Eventually, it became obvious that commercial researchers had grossly underestimated the difficulty of the project. 

In 1973, in response to the criticism from Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. 
Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again. 
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. 
Learning can be supervised, semi-supervised or unsupervised. 
Deep-learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.
"""

def generate_eval_corpus(tokenizer, target_tokens: int):
    toks = tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)
    reps = (target_tokens // len(toks)) + 1
    full_toks = (toks * reps)[:target_tokens]
    return full_toks

def compute_ppl_for_mode(wrapper, token_ids, compressed_decode=True):
    wrapper.ensure_loaded()
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if compressed_decode else "0"
    
    sid = "ppl_eval"
    wrapper.manager.clear_session(sid)
    if hasattr(wrapper, "_session_token_ids"):
        wrapper._session_token_ids[sid] = []
    wrapper.active_session = sid

    input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)
    target_ids = mx.array(token_ids[1:])

    wrapper.model._diffkv_session_ids = [sid]
    wrapper.manager.init_session(sid, prefill_len=len(token_ids)-1)

    # Simple sample decoding / logits cross entropy evaluation
    logits_list = []
    CH = 512
    N = len(token_ids) - 1
    for i in range(0, N, CH):
        chunk = input_ids[:, i:i+CH]
        pos = torch.tensor([list(range(i, i + chunk.shape[1]))], dtype=torch.long)
        out = wrapper.model(chunk, pos)
        
        # Extracted logits for chunk tokens
        lg = mx.array(out.logits.detach().cpu().numpy()) # shape [1, 1, V] or [1, len, V]
        if lg.ndim == 3:
            lg = lg[0]
        logits_list.append(lg)
        wrapper.manager.compress_deferred_prefill_blocks(sid)

    # Compute loss over generated steps
    all_logits = mx.concatenate(logits_list, axis=0)
    loss = nn.losses.cross_entropy(all_logits, target_ids[:all_logits.shape[0]], reduction="mean")
    mx.eval(loss)
    loss_val = float(loss.item())
    ppl = math.exp(loss_val)
    return loss_val, ppl

def run_ppl_eval(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print("--- Running Perplexity Evaluation (Dense vs DiffKV) ---", flush=True)
    wrapper = MLXDiffKVWrapper(
        model_id=model_id,
        config={"rank": 32, "block_size": 256},
    )
    wrapper.ensure_loaded()

    contexts = [4000, 8000, 16000]
    results = []

    for ctx in contexts:
        token_ids = generate_eval_corpus(wrapper.tokenizer, ctx)

        # Dense PPL
        loss_dense, ppl_dense = compute_ppl_for_mode(wrapper, token_ids, compressed_decode=False)

        # DiffKV PPL
        loss_diffkv, ppl_diffkv = compute_ppl_for_mode(wrapper, token_ids, compressed_decode=True)

        res = {
            "context_tokens": ctx,
            "loss_dense": round(loss_dense, 4),
            "ppl_dense": round(ppl_dense, 4),
            "loss_diffkv": round(loss_diffkv, 4),
            "ppl_diffkv": round(ppl_diffkv, 4),
            "ppl_delta_pct": round(((ppl_diffkv - ppl_dense) / ppl_dense) * 100.0, 2),
        }
        results.append(res)
        print(f"Context {ctx:>5} tokens | Dense PPL: {ppl_dense:.4f} (Loss: {loss_dense:.4f}) | DiffKV PPL: {ppl_diffkv:.4f} (Loss: {loss_diffkv:.4f}) | Delta: {res['ppl_delta_pct']:+.2f}%", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test3_perplexity.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved perplexity results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_ppl_eval()
