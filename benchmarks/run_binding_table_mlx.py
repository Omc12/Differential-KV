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
    correct, swaps, misses = 0, 0, 0

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

def run_binding_table_eval(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    print("--- Running Formal Entity-Binding Probe Table Benchmark ---", flush=True)

    configs = [
        ("Dense Baseline", "0", "0"),
        ("DKV without Owner Capture", "1", "0"),
        ("DKV with Owner Capture", "1", "1"),
    ]

    results = []

    for name, compressed, owner_cap in configs:
        os.environ["DKV_COMPRESSED_DECODE"] = compressed
        os.environ["DKV_RESIDUAL_OWNER_CAPTURE"] = owner_cap
        os.environ["DKV_RESIDUAL_EDGE_CAPTURE"] = "1"

        from serving.mlx_dkv_wrapper import MLXDKVWrapper
        wrapper = MLXDKVWrapper(
            model_id=model_id,
            config={"rank": 32, "block_size": 256},
        )

        prompt = build_prompt(wrapper.tokenizer, 8192)
        sid = f"binding_{name.replace(' ', '_')}"
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

        sc = score_response(response)
        entry = {
            "configuration": name,
            "compressed_decode": bool(int(compressed)),
            "owner_capture_enabled": bool(int(owner_cap)),
            "correct": f"{sc['correct']}/{sc['total']}",
            "correct_pct": round((sc['correct'] / sc['total']) * 100.0, 1),
            "swaps": sc['swaps'],
            "misses": sc['misses'],
            "time_sec": round(dt, 2),
            "response": response[:200],
        }
        results.append(entry)
        print(f"{name:<30} | Accuracy: {entry['correct']} ({entry['correct_pct']}%) | Swaps: {sc['swaps']} | Misses: {sc['misses']} | Time: {dt:.2f}s", flush=True)

    out_dir = os.path.join(REPO, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test9_entity_binding_table.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved entity binding table to {out_file}\n", flush=True)
    return results

if __name__ == "__main__":
    run_binding_table_eval()
