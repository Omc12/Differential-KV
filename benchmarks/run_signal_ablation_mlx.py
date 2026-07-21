#!/usr/bin/env python3
import sys
import os
import time
import json
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

ENTITIES = [
    ("Meridian",  "4382"),
    ("Okazaki",   "7156"),
    ("Halvorsen", "2903"),
    ("Brancusi",  "8617"),
    ("Tarkovsky", "5248"),
    ("Ellsworth", "1794"),
]

FILLER_PARA = (
    "The history of artificial intelligence is long and complex. Early AI "
    "researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

QUESTION = ("Based on the facility reports above, list each facility and its "
            "daily sample count, one per line, in the format 'Name: number'.")

def build_prompt(tokenizer, ctx_len: int):
    sents = [f"The {n} facility processes {v} samples per day." for n, v in ENTITIES]
    sys_part = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    assist_part = "<|im_end|>\n<|im_start|>assistant\n"

    sys_ids = tokenizer.encode(sys_part, add_special_tokens=False)
    q_ids = tokenizer.encode("\n\n" + QUESTION + assist_part, add_special_tokens=False)
    sent_ids = [tokenizer.encode(" " + s, add_special_tokens=False) for s in sents]
    filler_ids = tokenizer.encode(FILLER_PARA, add_special_tokens=False)

    budget = ctx_len - len(sys_ids) - len(q_ids) - sum(len(x) for x in sent_ids)
    n_rep = max(1, budget // len(filler_ids) + 1)
    all_filler = (filler_ids * n_rep)[:budget]

    # Interleave entities inside filler
    n_ent = len(sents)
    step = len(all_filler) // (n_ent + 1)
    prompt_ids = sys_ids[:]
    for i in range(n_ent):
        prompt_ids += all_filler[i * step: (i + 1) * step]
        prompt_ids += sent_ids[i]
    prompt_ids += all_filler[n_ent * step:]
    prompt_ids += q_ids

    return tokenizer.decode(prompt_ids)

def score_response(text):
    lines = text.lower().splitlines()
    allvals = {v for _, v in ENTITIES}
    correct = 0
    swaps = 0
    misses = 0

    for name, val in ENTITIES:
        verdict = "miss"
        for ln in lines:
            if name.lower() in ln:
                nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", ln)}
                planted = nums & allvals
                if val in planted:
                    verdict = "correct"
                elif planted:
                    verdict = "swap"
                break
        if verdict == "correct":
            correct += 1
        elif verdict == "swap":
            swaps += 1
        else:
            misses += 1

    return {"correct": correct, "swaps": swaps, "misses": misses, "total": len(ENTITIES)}

def run_ablation_eval(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print("--- Running Residual Signal Ablation Benchmark ---", flush=True)

    arms = [
        ("Dense Baseline", {"DIFFKV_COMPRESSED_DECODE": "0"}),
        ("Full DiffKV", {"DIFFKV_COMPRESSED_DECODE": "1", "DIFFKV_RESIDUAL_OWNER_CAPTURE": "1", "DIFFKV_RESIDUAL_EDGE_CAPTURE": "1"}),
        ("No Owner Capture", {"DIFFKV_COMPRESSED_DECODE": "1", "DIFFKV_RESIDUAL_OWNER_CAPTURE": "0", "DIFFKV_RESIDUAL_EDGE_CAPTURE": "1"}),
        ("No Edge Capture", {"DIFFKV_COMPRESSED_DECODE": "1", "DIFFKV_RESIDUAL_OWNER_CAPTURE": "1", "DIFFKV_RESIDUAL_EDGE_CAPTURE": "0"}),
    ]

    results = []

    for arm_name, env_vars in arms:
        # Update environment variables
        for k, v in env_vars.items():
            os.environ[k] = v

        # Import/re-init wrapper
        from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
        wrapper = MLXDiffKVWrapper(
            model_id=model_id,
            config={"rank": 32, "block_size": 256},
        )

        prompt = build_prompt(wrapper.tokenizer, 8192)
        sid = f"ablation_{arm_name.replace(' ', '_')}"
        wrapper.manager.clear_session(sid)
        if hasattr(wrapper, "_session_token_ids"):
            wrapper._session_token_ids[sid] = []
        wrapper.active_session = sid

        t0 = time.perf_counter()
        response = wrapper.generate(
            prompt=prompt,
            max_new_tokens=96,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        dt = time.perf_counter() - t0

        scores = score_response(response)
        res_entry = {
            "arm": arm_name,
            "env": env_vars,
            "scores": scores,
            "accuracy_pct": round((scores["correct"] / scores["total"]) * 100.0, 1),
            "time_sec": round(dt, 2),
            "response": response[:200],
        }
        results.append(res_entry)
        print(f"Arm: {arm_name:<20} | Correct: {scores['correct']}/{scores['total']} ({res_entry['accuracy_pct']}%) | Swaps: {scores['swaps']} | Misses: {scores['misses']} | Time: {dt:.2f}s", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test5_signal_ablation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved signal ablation results to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_ablation_eval()
