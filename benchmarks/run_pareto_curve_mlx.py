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

def compute_ppl_for_residual(wrapper, token_ids, max_residual: int):
    wrapper.ensure_loaded()
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
    os.environ["DIFFKV_MAX_RESIDUAL"] = str(max_residual)

    sid = f"pareto_res_{max_residual}"
    wrapper.manager.clear_session(sid)
    if hasattr(wrapper, "_session_token_ids"):
        wrapper._session_token_ids[sid] = []
    wrapper.active_session = sid

    input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)
    target_ids = mx.array(token_ids[1:])

    wrapper.model._diffkv_session_ids = [sid]
    wrapper.manager.init_session(sid, prefill_len=len(token_ids)-1)

    logits_list = []
    CH = 512
    N = len(token_ids) - 1
    for i in range(0, N, CH):
        chunk = input_ids[:, i:i+CH]
        pos = torch.tensor([list(range(i, i + chunk.shape[1]))], dtype=torch.long)
        out = wrapper.model(chunk, pos)
        
        lg = mx.array(out.logits.detach().cpu().numpy())
        if lg.ndim == 3:
            lg = lg[0]
        logits_list.append(lg)
        wrapper.manager.compress_deferred_prefill_blocks(sid)

    all_logits = mx.concatenate(logits_list, axis=0)
    loss = nn.losses.cross_entropy(all_logits, target_ids[:all_logits.shape[0]], reduction="mean")
    mx.eval(loss)
    loss_val = float(loss.item())
    ppl = math.exp(loss_val)
    return loss_val, ppl

def calculate_compression_ratio(rank=32, block_size=256, max_residual=128, dim=2048):
    dense_per_block = 2 * block_size * dim * 2
    lowrank_per_block = (block_size * rank + rank * dim + dim * rank) * 2 * 2
    residual_per_block = max_residual * dim * 2
    comp_per_block = lowrank_per_block + residual_per_block
    ratio = dense_per_block / comp_per_block if comp_per_block > 0 else 1.0
    return round(ratio, 2)

def run_pareto_sweep(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print("--- Running Compression Ratio vs Quality Pareto Curve Benchmark ---", flush=True)
    wrapper = MLXDiffKVWrapper(
        model_id=model_id,
        config={"rank": 32, "block_size": 256},
    )
    wrapper.ensure_loaded()

    token_ids = generate_eval_corpus(wrapper.tokenizer, 8192)
    max_residuals = [0, 16, 32, 64, 128, 256]
    results = []

    for r in max_residuals:
        t0 = time.perf_counter()
        loss, ppl = compute_ppl_for_residual(wrapper, token_ids, max_residual=r)
        dt = time.perf_counter() - t0

        comp_ratio = calculate_compression_ratio(rank=32, block_size=256, max_residual=r)
        vram_saving_pct = round((1.0 - (1.0 / comp_ratio)) * 100.0, 1)

        entry = {
            "max_residual": r,
            "perplexity": round(ppl, 4),
            "loss": round(loss, 4),
            "compression_ratio": comp_ratio,
            "vram_saving_pct": vram_saving_pct,
            "eval_time_sec": round(dt, 2),
        }
        results.append(entry)
        print(f"Max Residual: {r:>3} | PPL: {ppl:.4f} | Loss: {loss:.4f} | Compression Ratio: {comp_ratio}x ({vram_saving_pct}% VRAM saved)", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test10_pareto_curve.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Pareto curve results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_pareto_sweep()
