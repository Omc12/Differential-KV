#!/usr/bin/env python3
"""
benchmarks/prose_fact_recall.py — Non-digit proper-noun fact recall benchmark.

Evaluates retrieval of 10 bare proper-noun entity facts (no digits, symbols, or hyphenated codes)
embedded in dense prose across sequence lengths (8k, 16k, 32k, 64k) in dense vs DKV compressed mode.
"""
import os
import sys
import time
import argparse
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

PROSE_FACTS = [
    {"entity": "Recife", "fact": "The capital city of the Brazilian state of Pernambuco is Recife.", "query": "What is the capital city of Pernambuco?", "target": "Recife"},
    {"entity": "Bogor", "fact": "The historic botanical gardens of West Java are located in Bogor.", "query": "In which city are the historic botanical gardens of West Java located?", "target": "Bogor"},
    {"entity": "Akureyri", "fact": "The northern capital and major port city of Iceland is Akureyri.", "query": "What is the northern port city of Iceland?", "target": "Akureyri"},
    {"entity": "Gaborone", "fact": "The capital and largest economic center of Botswana is Gaborone.", "query": "What is the capital city of Botswana?", "target": "Gaborone"},
    {"entity": "Tromso", "fact": "The primary Arctic research hub in northern Norway is Tromso.", "query": "What is the primary Arctic research hub in northern Norway?", "target": "Tromso"},
    {"entity": "Cuenca", "fact": "The colonial highland city famous for Panama hats in Ecuador is Cuenca.", "query": "Which Ecuadorian city is famous for colonial architecture and Panama hats?", "target": "Cuenca"},
    {"entity": "Mendoza", "fact": "The principal wine-producing region at the foot of the Andes in Argentina is Mendoza.", "query": "What is the principal wine-producing city in Argentina?", "target": "Mendoza"},
    {"entity": "Oamaru", "fact": "The limestone architecture coastal town on the South Island of New Zealand is Oamaru.", "query": "Which New Zealand coastal town is known for limestone architecture?", "target": "Oamaru"},
    {"entity": "Brest", "fact": "The naval port city located in Finistere in western Brittany is Brest.", "query": "What is the major naval port city in Brittany, France?", "target": "Brest"},
    {"entity": "Turku", "fact": "The oldest city and former medieval capital of Finland is Turku.", "query": "What is the oldest medieval city in Finland?", "target": "Turku"},
]

FILLER = (
    "Geography and history interweave across global urban development. "
    "Cities around the world have served as centers of commerce, trade, and culture for centuries. "
    "Urban planners study regional migration patterns, architectural heritage, and historical archives. "
    "Preserving municipal records and local documentation ensures future generations understand regional identity. "
)

def build_prose_prompt(tok, fact_dict, ctx_len, depth=0.5, is_llama=False):
    filler_toks = tok.encode(FILLER, add_special_tokens=False)
    fact_toks = tok.encode(fact_dict["fact"] + "\n", add_special_tokens=False)
    q_toks = tok.encode(fact_dict["query"] + " Answer with the exact name only.", add_special_tokens=False)
    
    budget = ctx_len - len(fact_toks) - len(q_toks) - 80
    if budget < 100:
        budget = 100
    reps = budget // len(filler_toks) + 1
    all_filler = (filler_toks * reps)[:budget]
    split_at = int(len(all_filler) * depth)
    
    p1 = tok.decode(all_filler[:split_at])
    p2 = tok.decode(all_filler[split_at:])
    
    if is_llama:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + p1 + "\n" + fact_dict["fact"] + "\n" + p2 + "\n\n"
            + fact_dict["query"] + " Answer with the exact name only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + p1 + "\n" + fact_dict["fact"] + "\n" + p2 + "\n\n"
        + fact_dict["query"] + " Answer with the exact name only.<|im_end|>\n<|im_start|>assistant\n"
    )

def run_prose_fact_eval(model_id, ctx_list, compressed=True):
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
    print(f"[Prose Fact Recall] Loading {model_id} (Compressed={compressed})...")
    
    env_backup = os.environ.get("DKV_COMPRESSED_DECODE")
    if not compressed:
        os.environ["DKV_COMPRESSED_DECODE"] = "0"
    else:
        os.environ["DKV_COMPRESSED_DECODE"] = "1"
        
    wrapper = MLXDKVWrapper(model_id, preset="mid")
    tok = wrapper.tokenizer
    is_llama = "llama" in model_id.lower()
    
    summary_results = {}
    
    for ctx in ctx_list:
        recalled = 0
        total = len(PROSE_FACTS)
        print(f"\n--- Context Length: {ctx} ---")
        
        for i, item in enumerate(PROSE_FACTS):
            sid = f"prose_{ctx}_{i}"
            prompt = build_prose_prompt(tok, item, ctx, depth=0.5, is_llama=is_llama)
            
            try:
                wrapper.clear_session(sid)
            except Exception:
                pass
                
            t0 = time.time()
            out = wrapper.generate(prompt, max_new_tokens=16, temperature=0.0, session_id=sid)
            elapsed = time.time() - t0
            
            found = item["target"].lower() in out.lower()
            if found:
                recalled += 1
                status = "PASS"
            else:
                status = "FAIL"
                
            print(f"[{i+1}/{total}] Fact: {item['target']} -> {status} ({elapsed:.1f}s) | Output: {out.strip()!r}")
            
        acc = (recalled / total) * 100.0
        summary_results[ctx] = {"recalled": recalled, "total": total, "accuracy": acc}
        print(f"Ctx {ctx}: {recalled}/{total} ({acc:.1f}%)")
        
    if env_backup is not None:
        os.environ["DKV_COMPRESSED_DECODE"] = env_backup
        
    return summary_results

def main():
    parser = argparse.ArgumentParser(description="Prose Proper Noun Fact Recall Benchmark")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--ctx", nargs="+", type=int, default=[8192, 16384, 32768, 65536])
    parser.add_argument("--mode", choices=["compressed", "dense", "both"], default="compressed")
    args = parser.parse_args()
    
    results = {}
    if args.mode in ("compressed", "both"):
        print("\n=== Running DKV Compressed Mode ===")
        results["compressed"] = run_prose_fact_eval(args.model, args.ctx, compressed=True)
        
    if args.mode in ("dense", "both"):
        print("\n=== Running Dense Baseline Mode ===")
        results["dense"] = run_prose_fact_eval(args.model, args.ctx, compressed=False)
        
    print("\n================== SUMMARY ==================")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
