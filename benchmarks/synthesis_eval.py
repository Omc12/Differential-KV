#!/usr/bin/env python3
import os
import sys
import time
import argparse
import re
import subprocess
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

FACTS = [
    "rahimi",
    "recht",
    "bochner",
    "fourier",
    "sinusoid",
    "hoeffding",
    "hessian",
    "laplacian",
    "binning",
    "kernel",
    "support vector machine",
    "least squares",
    "cvm",
    "forest",
    "randomly shifted"
]

LINKED_PAIRS = [
    ("rahimi", "recht"),
    ("bochner", "fourier"),
    ("hoeffding", "convergence"),
    ("randomly shifted", "binning"),
    ("least squares", "linear")
]

def compute_scores(text):
    text_lower = text.lower()
    facts_found = [f for f in FACTS if f in text_lower]
    n_facts = len(facts_found)
    
    # Split sentences
    sentences = [s.lower() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    
    linked_found = 0
    for x, y in LINKED_PAIRS:
        satisfied = False
        for i in range(len(sentences)):
            if x in sentences[i]:
                for j in [i-1, i, i+1]:
                    if 0 <= j < len(sentences) and y in sentences[j]:
                        satisfied = True
                        break
            if satisfied:
                break
        if satisfied:
            linked_found += 1
            
    fact_score = (n_facts / len(FACTS)) * 50.0
    link_score = (linked_found / len(LINKED_PAIRS)) * 50.0
    total_score = fact_score + link_score
    return n_facts, linked_found, total_score

def build_prompt(tokenizer, paper_text, filler_tokens, target_len):
    inst = "\n\nWrite a connected, narrative paragraph summarizing the key contributions and mathematical details of the text above."
    system_part = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    assistant_part = "<|im_end|>\n<|im_start|>assistant\n"
    
    sys_ids = tokenizer.encode(system_part, add_special_tokens=False)
    inst_ids = tokenizer.encode(inst + assistant_part, add_special_tokens=False)
    paper_ids = tokenizer.encode(paper_text, add_special_tokens=False)
    
    base_len = len(sys_ids) + len(inst_ids) + len(paper_ids)
    filler_budget = target_len - base_len
    if filler_budget < 0:
        filler_budget = 0
        
    slice_filler = filler_tokens[:filler_budget]
    filler_text = tokenizer.decode(slice_filler)
    
    prompt = system_part + paper_text + "\n\n[Background context / Filler]:\n" + filler_text + inst + assistant_part
    return prompt

def run_mlx(model_id, prompt_text, mode, max_tokens=250):
    import torch
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if mode == "compressed" else "0"
    
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DiffKVHFWrapper(model_id=model_id, config=cfg)
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model
    
    ids = tok.encode(prompt_text)
    sid = "synthesis_eval"
    mgr.clear_session(sid)
    wrapper._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    model._diffkv_session_ids = [sid]
    
    CH = 512
    output = None
    for cs in range(0, len(ids), CH):
        chunk = ids[cs:cs + CH]
        ct = torch.tensor([chunk], dtype=torch.long)
        pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
        output = model(ct, pt)
        mgr.compress_deferred_prefill_blocks(sid)
    logits = output.logits[0, -1].cpu().numpy()
    
    cur = len(ids)
    generated = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        nid = int(np.argmax(logits))
        generated.append(nid)
        if nid == tok.eos_token_id:
            break
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        output = model(torch.tensor([[nid]], dtype=torch.long),
                       torch.tensor([[cur]], dtype=torch.long))
        logits = output.logits[0, -1].cpu().numpy()
        cur += 1
    dt = time.perf_counter() - t0
    
    summary = tok.decode(generated)
    tps = len(generated) / dt if dt > 0 else 0.0
    return summary, tps

def run_native(model_path, prompt_text, mode, max_tokens=250):
    binary_path = os.path.join(REPO, "diffkv_native/build/diffkv_native")
    
    env = os.environ.copy()
    env["DIFFKV_MAX_TOKENS"] = str(max_tokens)
    env["DIFFKV_TEMPERATURE"] = "0"
    if mode == "compressed":
        env["DIFFKV_ENGAGE_THRESHOLD"] = "1024"
    else:
        env["DIFFKV_ENGAGE_THRESHOLD"] = "999999"
        
    cmd = [binary_path, model_path, prompt_text]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, errors='replace')
    dt = time.perf_counter() - t0
    
    output = res.stdout
    summary = output.strip()
    words = len(summary.split())
    tps = words / dt if dt > 0 else 0.0
    return summary, tps

def main():
    parser = argparse.ArgumentParser(description="Synthesis Evaluation Harness")
    parser.add_argument("--ctx", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--model-mlx", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--model-native", default=os.path.join(REPO, "diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"))
    parser.add_argument("--engine", choices=["mlx", "native", "both"], default="both")
    parser.add_argument("--mode", choices=["dense", "compressed", "both"], default="both")
    parser.add_argument("--gen", type=int, default=250)
    
    # Internal options for subprocess single-run execution
    parser.add_argument("--single-run", action="store_true", help="run a single isolated evaluation step")
    args = parser.parse_args()
    
    if args.single_run or "--single-run" in sys.argv:
        # Read Random Features Paper text
        paper_path = os.path.join(HERE, "random_features_paper.txt")
        if not os.path.exists(paper_path):
            print(f"Error: paper file not found at {paper_path}")
            sys.exit(1)
        with open(paper_path, "r") as f:
            paper_text = f.read()
            
        # Read Pride and Prejudice as filler source
        filler_path = os.path.join(REPO, "scratch/pride_and_prejudice.txt")
        if not os.path.exists(filler_path):
            print(f"Error: filler source file not found at {filler_path}")
            sys.exit(1)
        with open(filler_path, "r") as f:
            filler_text = f.read()
            
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_mlx, local_files_only=True)
        filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)

        single_engine = args.engine
        single_mode = args.mode
        single_ctx = args.ctx[0]
        
        prompt = build_prompt(tokenizer, paper_text, filler_tokens, single_ctx)
        
        if single_engine == "mlx":
            summary, tps = run_mlx(args.model_mlx, prompt, single_mode, max_tokens=args.gen)
        else:
            summary, tps = run_native(args.model_native, prompt, single_mode, max_tokens=args.gen)
            
        print(json.dumps({"summary": summary, "tps": tps}))
        sys.exit(0)
        
    # Main orchestrator mode
    engines = ["mlx", "native"] if args.engine == "both" else [args.engine]
    modes = ["dense", "compressed"] if args.mode == "both" else [args.mode]
    
    print("=" * 60)
    print("RUNNING LONG-FORM COHERENCE/SYNTHESIS EVALUATION (ISOLATED PROCESSES)")
    print("=" * 60)
    
    results = {}
    
    for engine in engines:
        for mode in modes:
            for ctx in args.ctx:
                print(f"\nRunning {engine.upper()} | {mode.upper()} | Context: {ctx}...")
                
                # Launch a new subprocess of this exact script to isolate memory/Unified Memory state
                cmd = [
                    sys.executable,
                    __file__,
                    "--single-run",
                    "--engine", engine,
                    "--mode", mode,
                    "--ctx", str(ctx),
                    "--model-mlx", args.model_mlx,
                    "--model-native", args.model_native,
                    "--gen", str(args.gen)
                ]
                
                t0 = time.perf_counter()
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors='replace')
                dt = time.perf_counter() - t0
                
                try:
                    stdout_str = res.stdout.strip()
                    json_start = stdout_str.find("{")
                    if json_start == -1:
                        raise ValueError("No JSON object found in stdout")
                    data = json.loads(stdout_str[json_start:])
                    summary = data["summary"]
                    tps = data["tps"]
                except Exception as e:
                    print(f"Subprocess run failed to return valid JSON: {e}")
                    print(f"Stdout was:\n{res.stdout}")
                    continue
                    
                n_facts, n_links, score = compute_scores(summary)
                print(f"TPS: {tps:.1f}")
                print(f"Score: {score:.1f}/100 (Facts: {n_facts}/15, Linkages: {n_links}/5)")
                print(f"Output: {summary[:200]}...")
                
                results[(engine, mode, ctx)] = {
                    "tps": tps,
                    "facts": n_facts,
                    "links": n_links,
                    "score": score,
                    "summary": summary
                }
                
    # Print the Markdown results table
    print("\n" + "=" * 60)
    print("FINAL RESULTS TABLE")
    print("=" * 60)
    
    print("\n| Engine | Mode | Context | Score | Facts | Linkages | TPS |")
    print("|---|---|---|---|---|---|---|")
    for (eng, md, ct), res in sorted(results.items()):
        print(f"| {eng.upper()} | {md.lower()} | {ct} | {res['score']:.1f} | {res['facts']}/15 | {res['links']}/5 | {res['tps']:.1f} |")

if __name__ == "__main__":
    main()
