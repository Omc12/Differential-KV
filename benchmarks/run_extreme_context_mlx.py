#!/usr/bin/env python3
import sys
import os
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
os.environ.setdefault("DIFFKV_MAX_RESIDUAL", "128")
os.environ["DIFFKV_MAX_BLOCKS"] = "600"

from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper

NEEDLE = "PHOENIX-9942-ULTRA"
NEEDLE_SENT = f"The master key for emergency override is {NEEDLE}."
QUESTION = "What is the master key for emergency override? State it clearly."

FILLER = (
    "The facility maintenance log records routine calibration of the cooling loops, "
    "periodic inspection of the conduit seals, and scheduled rotation of the backup generators. "
    "Technicians note ambient humidity, verify the airlock interlocks, and confirm telemetry uplink. "
)

def make_extreme_prompt(tokenizer, target_tokens: int, depth=0.5):
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    sys_part = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    assist_part = "\n\n" + QUESTION + "<|im_end|>\n<|im_start|>assistant\n"

    sys_ids = tokenizer.encode(sys_part, add_special_tokens=False)
    assist_ids = tokenizer.encode(assist_part, add_special_tokens=False)
    needle_ids = tokenizer.encode("\n" + NEEDLE_SENT + "\n", add_special_tokens=False)

    overhead = len(sys_ids) + len(assist_ids) + len(needle_ids)
    budget = max(100, target_tokens - overhead)

    repeats = (budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:budget]

    at = int(len(all_filler) * depth)
    p1 = tokenizer.decode(all_filler[:at])
    p2 = tokenizer.decode(all_filler[at:])

    prompt = sys_part + p1 + "\n" + NEEDLE_SENT + "\n" + p2 + assist_part
    return prompt

def run_extreme_eval(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print("--- Running Extreme Context Stress Test (48k & 64k) ---", flush=True)
    wrapper = MLXDiffKVWrapper(
        model_id=model_id,
        config={"rank": 32, "block_size": 256},
    )

    contexts = [48000, 64000]
    results = []

    for ctx in contexts:
        try:
            prompt = make_extreme_prompt(wrapper.tokenizer, ctx, depth=0.5)

            sid = f"extreme_{ctx}"
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
                "success": success,
                "time_sec": round(dt, 2),
                "tps": round(tps, 2),
                "response": gen_text[:120],
                "error": None,
            }
            results.append(res_entry)

            status = "PASS" if success else "FAIL"
            print(f"Context: {ctx:>6} tokens | Result: {status} | Time: {dt:.2f}s | TPS: {tps:.1f}", flush=True)
            print(f"  Response: {gen_text!r}", flush=True)

        except Exception as e:
            print(f"Context: {ctx:>6} tokens | FAILED with error: {e}", flush=True)
            results.append({
                "context_tokens": ctx,
                "success": False,
                "error": str(e),
            })

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test8_extreme_context.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved extreme context results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_extreme_eval()
