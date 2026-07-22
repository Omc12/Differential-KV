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

NEEDLE_A = ("CYAN-EAGLE-88", "The security classification key for Project Apex is CYAN-EAGLE-88.")
NEEDLE_B = ("ZETA-9912-OMEGA", "The passcode for Project Apex is ZETA-9912-OMEGA.")

QUESTION = "What is the passcode for the project whose security classification key is CYAN-EAGLE-88? State the passcode clearly."

FILLER = (
    "The facility maintenance log records routine calibration of the cooling loops, "
    "periodic inspection of the conduit seals, and scheduled rotation of the backup generators. "
    "Technicians note ambient humidity, verify the airlock interlocks, and confirm telemetry uplink. "
)

def make_multihop_prompt(tokenizer, target_tokens: int):
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    nA_toks = tokenizer.encode(NEEDLE_A[1] + "\n", add_special_tokens=False)
    nB_toks = tokenizer.encode(NEEDLE_B[1] + "\n", add_special_tokens=False)
    q_toks = tokenizer.encode(QUESTION, add_special_tokens=False)

    overhead = len(nA_toks) + len(nB_toks) + len(q_toks) + 80
    budget = max(100, target_tokens - overhead)

    repeats = (budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:budget]

    at1 = int(len(all_filler) * 0.25)
    at2 = int(len(all_filler) * 0.75)

    p1 = tokenizer.decode(all_filler[:at1])
    p2 = tokenizer.decode(all_filler[at1:at2])
    p3 = tokenizer.decode(all_filler[at2:])

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + p1 + "\n" + NEEDLE_A[1] + "\n"
        + p2 + "\n" + NEEDLE_B[1] + "\n"
        + p3 + "\n\n"
        + QUESTION + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def run_eval(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print(f"--- Running Multi-Hop NIAH Benchmark ---", flush=True)
    print(f"Model: {model_id}", flush=True)
    wrapper = MLXDKVWrapper(
        model_id=model_id,
        config={"rank": 32, "block_size": 256},
    )

    contexts = [4000, 16000, 32000]
    results = []

    for ctx in contexts:
        prompt = make_multihop_prompt(wrapper.tokenizer, ctx)
        prompt_toks = len(wrapper.tokenizer.encode(prompt))

        sid = f"multihop_{ctx}"
        wrapper.manager.clear_session(sid)
        if hasattr(wrapper, "_session_token_ids"):
            wrapper._session_token_ids[sid] = []
        wrapper.active_session = sid

        t0 = time.perf_counter()
        response = wrapper.generate(
            prompt=prompt,
            max_new_tokens=48,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        dt = time.perf_counter() - t0

        gen_text = response.strip()
        success = NEEDLE_B[0] in gen_text
        gen_toks = len(wrapper.tokenizer.encode(gen_text))
        tps = gen_toks / dt if dt > 0 else 0.0

        res_entry = {
            "context_tokens": ctx,
            "prompt_tokens": prompt_toks,
            "gen_tokens": gen_toks,
            "target_passcode": NEEDLE_B[0],
            "success": success,
            "time_sec": round(dt, 2),
            "tps": round(tps, 2),
            "response": gen_text[:200],
        }
        results.append(res_entry)

        status = "PASS" if success else "FAIL"
        print(f"Context: {ctx:>5} tokens | Result: {status} | Target: {NEEDLE_B[0]} | Time: {dt:.2f}s | TPS: {tps:.1f}", flush=True)
        print(f"  Response: {gen_text!r}", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test2_multihop.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_eval()
