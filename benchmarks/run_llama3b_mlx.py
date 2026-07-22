#!/usr/bin/env python3
import sys
import os
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

os.environ["DKV_COMPRESSED_DECODE"] = "1"
os.environ.setdefault("DKV_MAX_RESIDUAL", "128")

from serving.mlx_dkv_wrapper import MLXDKVWrapper

NEEDLE = "OMEGA-7741-DELTA"
NEEDLE_SENT = f"The secret passcode is {NEEDLE}."
QUESTION = "What is the secret passcode? Repeat it exactly."

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

def make_llama_prompt(tokenizer, target_tokens: int, depth: float):
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_toks = tokenizer.encode(NEEDLE_SENT + "\n", add_special_tokens=False)
    question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)

    overhead = len(needle_toks) + len(question_toks) + 80
    budget = max(100, target_tokens - overhead)

    repeats = (budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:budget]

    at = int(len(all_filler) * depth)
    p1 = tokenizer.decode(all_filler[:at])
    p2 = tokenizer.decode(all_filler[at:])

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + p1 + "\n" + NEEDLE_SENT + "\n" + p2 + "\n\n"
        + QUESTION + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def run_eval(model_id="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    print(f"--- Running Second Model Generalization Benchmark (Llama-3.2-3B) ---", flush=True)
    print(f"Model: {model_id}", flush=True)
    
    wrapper = MLXDKVWrapper(
        model_id=model_id,
        config={"rank": 32, "block_size": 256},
    )

    contexts = [4000, 8000, 16000]
    depths = [0.1, 0.5, 0.9]
    results = []

    for ctx in contexts:
        for depth in depths:
            prompt = make_llama_prompt(wrapper.tokenizer, ctx, depth)
            prompt_toks = len(wrapper.tokenizer.encode(prompt))

            sid = f"llama3b_{ctx}_{depth}"
            wrapper.manager.clear_session(sid)
            if hasattr(wrapper, "_session_token_ids"):
                wrapper._session_token_ids[sid] = []
            wrapper.active_session = sid

            t0 = time.perf_counter()
            response = wrapper.generate(
                prompt=prompt,
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
            dt = time.perf_counter() - t0

            gen_text = response.strip()
            success = NEEDLE in gen_text
            gen_toks = len(wrapper.tokenizer.encode(gen_text))
            tps = gen_toks / dt if dt > 0 else 0.0

            res_entry = {
                "context_tokens": ctx,
                "depth": depth,
                "prompt_tokens": prompt_toks,
                "gen_tokens": gen_toks,
                "success": success,
                "time_sec": round(dt, 2),
                "tps": round(tps, 2),
                "response": gen_text[:120],
            }
            results.append(res_entry)

            status = "PASS" if success else "FAIL"
            print(f"Context: {ctx:>5} | Depth: {depth:.1f} | Result: {status} | Time: {dt:.2f}s | TPS: {tps:.1f}", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test4_llama3b_niah.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_eval()
