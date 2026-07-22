#!/usr/bin/env python3
import sys
import os
import time
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

os.environ["DKV_COMPRESSED_DECODE"] = "1"
os.environ.setdefault("DKV_MAX_RESIDUAL", "128")

from serving.mlx_dkv_wrapper import MLXDKVWrapper
import mlx.core as mx

NEEDLES = [
    ("OMEGA-7741-DELTA", "The first secret passcode is OMEGA-7741-DELTA."),
    ("SIGMA-9923-BETA", "The second secret passcode is SIGMA-9923-BETA."),
    ("THETA-1105-ALPHA", "The third secret passcode is THETA-1105-ALPHA."),
    ("KAPPA-4419-GAMMA", "The fourth secret passcode is KAPPA-4419-GAMMA."),
]

QUESTION = "What are the four secret passcodes? List all of them clearly."

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

def make_multi_needle_prompt(tokenizer, target_tokens: int, depths=[0.2, 0.4, 0.6, 0.8]):
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_tok_list = [tokenizer.encode(sent + "\n", add_special_tokens=False) for _, sent in NEEDLES]
    question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)
    overhead = sum(len(n) for n in needle_tok_list) + len(question_toks) + 80

    budget = target_tokens - overhead
    if budget < 100:
        budget = 100

    repeats = (budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:budget]

    # Insert needles at specified depth points
    indices = [int(len(all_filler) * d) for d in depths]
    indices = sorted(indices)

    parts = []
    prev = 0
    for i, idx in enumerate(indices):
        parts.append(tokenizer.decode(all_filler[prev:idx]))
        parts.append("\n" + NEEDLES[i][1] + "\n")
        prev = idx
    parts.append(tokenizer.decode(all_filler[prev:]))

    body = "".join(parts)

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + body + "\n\n"
        + QUESTION + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def run_eval(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print(f"--- Running Multi-Needle NIAH Benchmark ---", flush=True)
    print(f"Model: {model_id}", flush=True)
    wrapper = MLXDKVWrapper(
        model_id=model_id,
        config={"rank": 32, "block_size": 256},
    )

    contexts = [4000, 16000, 32000]
    results = []

    for ctx in contexts:
        prompt = make_multi_needle_prompt(wrapper.tokenizer, ctx)
        prompt_toks = len(wrapper.tokenizer.encode(prompt))

        sid = f"multi_needle_{ctx}"
        wrapper.manager.clear_session(sid)
        if hasattr(wrapper, "_session_token_ids"):
            wrapper._session_token_ids[sid] = []
        wrapper.active_session = sid

        t0 = time.perf_counter()
        response = wrapper.generate(
            prompt=prompt,
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        dt = time.perf_counter() - t0

        gen_text = response.strip()

        # Check each needle
        found_flags = [code in gen_text for code, _ in NEEDLES]
        num_found = sum(found_flags)
        recall_pct = (num_found / len(NEEDLES)) * 100.0

        gen_toks = len(wrapper.tokenizer.encode(gen_text))
        tps = gen_toks / dt if dt > 0 else 0.0

        res_entry = {
            "context_tokens": ctx,
            "prompt_tokens": prompt_toks,
            "gen_tokens": gen_toks,
            "needles_total": len(NEEDLES),
            "needles_found": num_found,
            "recall_pct": recall_pct,
            "needle_details": {NEEDLES[i][0]: found_flags[i] for i in range(len(NEEDLES))},
            "time_sec": round(dt, 2),
            "tps": round(tps, 2),
            "response_sample": gen_text[:200],
        }
        results.append(res_entry)

        print(f"Context: {ctx:>5} tokens | Needles: {num_found}/{len(NEEDLES)} ({recall_pct:.1f}%) | Time: {dt:.2f}s | TPS: {tps:.1f}", flush=True)
        print(f"  Response: {gen_text!r}", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test1_multi_needle.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_eval()
