#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

sys.path.insert(0, HERE)

# ------------------------------------------------------------------------------
# BASELINE EVALUATION LOGIC
# ------------------------------------------------------------------------------

# Needle definitions for multi-needle
NEEDLES_MULTI = ["OMEGA-7741-DELTA", "SIGMA-9923-BETA", "THETA-1105-ALPHA"]
NEEDLE_SENTS_MULTI = [
    "The first secret passcode is OMEGA-7741-DELTA.",
    "The second secret passcode is SIGMA-9923-BETA.",
    "The third secret passcode is THETA-1105-ALPHA."
]
QUESTION_MULTI = "What are the three secret passcodes? List them all in order."

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

# Relational entities
NATURAL = [
    ("Quillfeather", "4193"),
    ("Braxanible",   "8857"),
    ("Morrowind",    "2206"),
    ("Vantablack",   "6034"),
]
NAT_SENT = "Dr. {name} reported that the {name}-cluster survey catalogued precisely {val} variable stars."
NAT_Q = "How many variable stars did Dr. {name} report?"

# Synthesis facts
FACTS = [
    "rahimi", "recht", "bochner", "fourier", "sinusoid", "hoeffding",
    "hessian", "laplacian", "binning", "kernel", "support vector machine",
    "least squares", "cvm", "forest", "randomly shifted"
]
LINKED_PAIRS = [
    ("rahimi", "recht"),
    ("bochner", "fourier"),
    ("hoeffding", "convergence"),
    ("randomly shifted", "binning"),
    ("least squares", "linear")
]


# Prompt builders
def build_multi_needle_prompt(tok, ctx, is_llama=False):
    filler = tok.encode(FILLER, add_special_tokens=False)
    sents_tokens = []
    for s in NEEDLE_SENTS_MULTI:
        sents_tokens.extend(tok.encode(s + "\n", add_special_tokens=False))
    q_tokens = tok.encode(QUESTION_MULTI, add_special_tokens=False)
    
    budget = ctx - len(sents_tokens) - len(q_tokens) - 80
    if budget < 100:
        budget = 100
    reps = budget // len(filler) + 1
    allf = (filler * reps)[:budget]
    
    at1 = int(len(allf) * 0.25)
    at2 = int(len(allf) * 0.50)
    at3 = int(len(allf) * 0.75)
    
    p1 = tok.decode(allf[:at1])
    p2 = tok.decode(allf[at1:at2])
    p3 = tok.decode(allf[at2:at3])
    p4 = tok.decode(allf[at3:])
    
    if is_llama:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + p1 + "\n" + NEEDLE_SENTS_MULTI[0] + "\n"
            + p2 + "\n" + NEEDLE_SENTS_MULTI[1] + "\n"
            + p3 + "\n" + NEEDLE_SENTS_MULTI[2] + "\n"
            + p4 + "\n\n" + QUESTION_MULTI + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + p1 + "\n" + NEEDLE_SENTS_MULTI[0] + "\n"
        + p2 + "\n" + NEEDLE_SENTS_MULTI[1] + "\n"
        + p3 + "\n" + NEEDLE_SENTS_MULTI[2] + "\n"
        + p4 + "\n\n" + QUESTION_MULTI + "<|im_end|>\n<|im_start|>assistant\n"
    )

def build_synthesis_prompt(tokenizer, paper_text, filler_tokens, target_len, is_llama=False):
    inst = "\n\nWrite a connected, narrative paragraph summarizing the key contributions and mathematical details of the text above."
    
    if is_llama:
        system_part = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant_part = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
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

def build_relational_prompt(ask_name, tokenizer, target_tokens=3500, spread=True, is_llama=False):
    question = "\n\nQuestion: " + NAT_Q.format(name=ask_name) + " Answer with the number only."
    
    def n(t):
        return len(tokenizer.encode(t, add_special_tokens=False))

    def templ(content):
        if is_llama:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n" + content + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
        )

    fact_lines = [NAT_SENT.format(name=nm, val=v) + " " for nm, v in NATURAL]
    filler_tok = max(1, n(FILLER))
    if spread:
        per_gap = max(1, (target_tokens // (len(NATURAL) + 1)) // filler_tok)
        body = "Observatory bulletin.\n"
        for fl in fact_lines:
            body += (FILLER * per_gap) + fl + "\n"
        body += FILLER * per_gap
    else:
        facts = "Observatory bulletin.\n" + "".join(fl + "\n" for fl in fact_lines) + "\n"
        overhead = n(templ(facts + question))
        reps = max(1, (target_tokens - overhead) // filler_tok)
        body = facts + (FILLER * reps)
    return templ(body + question)


def compute_synthesis_scores(text):
    text_lower = text.lower()
    facts_found = [f for f in FACTS if re.search(r"\b" + re.escape(f) + r"\b", text_lower)]
    n_facts = len(facts_found)
    
    sentences = [s.lower() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    
    linked_found = 0
    for x, y in LINKED_PAIRS:
        satisfied = False
        for i in range(len(sentences)):
            if re.search(r"\b" + re.escape(x) + r"\b", sentences[i]):
                for j in [i-1, i, i+1]:
                    if 0 <= j < len(sentences) and re.search(r"\b" + re.escape(y) + r"\b", sentences[j]):
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

# ------------------------------------------------------------------------------
# TEST RUNNERS (SUBPROCESS ENTRY POINTS)
# ------------------------------------------------------------------------------

def _mx_peak_gb():
    try:
        import mlx.core as mx
    except ImportError:
        return 0.0
    for obj, name in ((mx, "get_peak_memory"),
                      (getattr(mx, "metal", None), "get_peak_memory")):
        if obj is not None and hasattr(obj, name):
            try:
                return float(getattr(obj, name)()) / 1e9
            except Exception:
                pass
    return 0.0

def _mx_reset_peak():
    try:
        import mlx.core as mx
    except ImportError:
        return
    for obj, name in ((mx, "reset_peak_memory"),
                      (getattr(mx, "metal", None), "reset_peak_memory")):
        if obj is not None and hasattr(obj, name):
            try:
                getattr(obj, name)()
                return
            except Exception:
                pass

def load_model_and_tokenizer(model_id, mode):
    import mlx.core as mx
    _mx_reset_peak()
    
    is_llama = "llama" in model_id.lower()
    quant = None if is_llama else "int4"
    
    if mode == "dense":
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(model_id)
        return model, tokenizer, None
    else:
        sys.path.insert(0, ACTIVE)
        from serving.hf_dkv_wrapper import DKVHFWrapper
        
        cfg = {"quantization": quant, "rank": 16, "block_size": 256,
               "micro_block_size": 256, "preset": "mid"}
        w = DKVHFWrapper(model_id=model_id, config=cfg)
        w.ensure_loaded()
        return w.model, w.tokenizer, w.manager


def run_multi_needle_sub(model_id, mode, ctx, gen=64):
    import mlx.core as mx
    import numpy as np
    import torch
    
    model, tokenizer, manager = load_model_and_tokenizer(model_id, mode)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    is_llama = "llama" in model_id.lower() or "llama" in str(type(tok)).lower()
    prompt = build_multi_needle_prompt(tok, ctx, is_llama=is_llama)
    
    ids = tok.encode(prompt)
    
    if mode == "dense":
        from mlx_lm.models.cache import make_prompt_cache
        
        prompt_cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        
        CH = 512
        logits = None
        for cs in range(0, len(ids), CH):
            chunk = mx.array(ids[cs:cs + CH])[None]
            logits = model(chunk, cache=prompt_cache)
            mx.eval(logits)
            
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        prefill_s = time.perf_counter() - t0
        
        generated = []
        t0 = time.perf_counter()
        steps = 0
        
        for _ in range(gen):
            nid = int(y.item())
            generated.append(nid)
            steps += 1
            text_so_far = tok.decode(generated)
            if all(n in text_so_far for n in NEEDLES_MULTI):
                break
            logits = model(y[None], cache=prompt_cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
        
        decode_s = time.perf_counter() - t0
        text = tok.decode(generated)
        ok = all(n in text for n in NEEDLES_MULTI)
        
        return {
            "ok": ok,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tps": steps / decode_s if decode_s > 0 else 0.0,
            "peak_mem_gb": _mx_peak_gb(),
            "output": text
        }
    else:
        sid = "multi_needle_eval"
        manager.clear_session(sid)
        manager._session_token_ids[sid] = []
        manager.init_session(sid, prefill_len=len(ids))
        manager.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
        model._dkv_session_ids = [sid]
        
        t0 = time.perf_counter()
        CH = 512
        output = None
        for cs in range(0, len(ids), CH):
            chunk = ids[cs:cs + CH]
            ct = torch.tensor([chunk], dtype=torch.long)
            pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
            output = model(ct, pt)
            manager.compress_deferred_prefill_blocks(sid)
        logits = output.logits[0, -1].cpu().numpy()
        prefill_s = time.perf_counter() - t0
        
        cur = len(ids)
        generated = []
        
        nid = int(np.argmax(logits))
        generated.append(nid)
        manager.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        output = model(torch.tensor([[nid]], dtype=torch.long),
                       torch.tensor([[cur]], dtype=torch.long))
        logits = output.logits[0, -1].cpu().numpy()
        cur += 1
        
        t0 = time.perf_counter()
        steps = 0
        for _ in range(gen - 1):
            nid = int(np.argmax(logits))
            generated.append(nid)
            steps += 1
            text_so_far = tok.decode(generated)
            if all(n in text_so_far for n in NEEDLES_MULTI):
                break
            manager.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
            output = model(torch.tensor([[nid]], dtype=torch.long),
                           torch.tensor([[cur]], dtype=torch.long))
            logits = output.logits[0, -1].cpu().numpy()
            cur += 1
        decode_s = time.perf_counter() - t0
        
        text = tok.decode(generated)
        ok = all(n in text for n in NEEDLES_MULTI)
        
        return {
            "ok": ok,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tps": steps / decode_s if decode_s > 0 else 0.0,
            "peak_mem_gb": _mx_peak_gb(),
            "output": text
        }


def run_synthesis_sub(model_id, mode, ctx, gen=250):
    import mlx.core as mx
    import numpy as np
    import torch
    
    paper_path = os.path.join(HERE, "random_features_paper.txt")
    with open(paper_path, "r") as f:
        paper_text = f.read()
        
    filler_path = os.path.join(REPO, "scratch/pride_and_prejudice.txt")
    with open(filler_path, "r") as f:
        filler_text = f.read()
        
    model, tokenizer, manager = load_model_and_tokenizer(model_id, mode)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    is_llama = "llama" in model_id.lower() or "llama" in str(type(tok)).lower()
    
    filler_tokens = tok.encode(filler_text, add_special_tokens=False)
    
    if mode == "active":
        os.environ["DKV_SPARSE_BIAS"] = "2.0"
        
    prompt = build_synthesis_prompt(tok, paper_text, filler_tokens, ctx, is_llama=is_llama)
    ids = tok.encode(prompt)
    
    if mode == "dense":
        from mlx_lm.models.cache import make_prompt_cache
        
        prompt_cache = make_prompt_cache(model)
        
        CH = 512
        logits = None
        for cs in range(0, len(ids), CH):
            chunk = mx.array(ids[cs:cs + CH])[None]
            logits = model(chunk, cache=prompt_cache)
            mx.eval(logits)
            
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        
        generated = []
        for _ in range(gen):
            nid = int(y.item())
            generated.append(nid)
            if nid == tok.eos_token_id:
                break
            logits = model(y[None], cache=prompt_cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
        
        text = tok.decode(generated)
        n_facts, n_links, score = compute_synthesis_scores(text)
        return {
            "score": score,
            "facts": n_facts,
            "links": n_links,
            "output": text,
            "peak_mem_gb": _mx_peak_gb()
        }
    else:
        sid = "synthesis_eval"
        manager.clear_session(sid)
        manager._session_token_ids[sid] = []
        manager.init_session(sid, prefill_len=len(ids))
        manager.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
        model._dkv_session_ids = [sid]
        
        CH = 512
        output = None
        for cs in range(0, len(ids), CH):
            chunk = ids[cs:cs + CH]
            ct = torch.tensor([chunk], dtype=torch.long)
            pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
            output = model(ct, pt)
            manager.compress_deferred_prefill_blocks(sid)
        logits = output.logits[0, -1].cpu().numpy()
        
        cur = len(ids)
        generated = []
        for _ in range(gen):
            nid = int(np.argmax(logits))
            generated.append(nid)
            if nid == tok.eos_token_id:
                break
            manager.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
            output = model(torch.tensor([[nid]], dtype=torch.long),
                           torch.tensor([[cur]], dtype=torch.long))
            logits = output.logits[0, -1].cpu().numpy()
            cur += 1
            
        text = tok.decode(generated)
        n_facts, n_links, score = compute_synthesis_scores(text)
        return {
            "score": score,
            "facts": n_facts,
            "links": n_links,
            "output": text,
            "peak_mem_gb": _mx_peak_gb()
        }


def run_relational_sub(model_id, mode, gen=24):
    import mlx.core as mx
    import numpy as np
    import torch
    
    model, tokenizer, manager = load_model_and_tokenizer(model_id, mode)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    is_llama = "llama" in model_id.lower() or "llama" in str(type(tok)).lower()
    
    results = []
    n_correct = n_misbound = 0
    
    for i, (name, key) in enumerate(NATURAL):
        sid = f"rel_{mode}_{i}"
        
        if mode == "dense":
            from mlx_lm.models.cache import make_prompt_cache
            
            prompt = build_relational_prompt(name, tok, target_tokens=3500, spread=True, is_llama=is_llama)
            prompt_ids = tok.encode(prompt)
            
            prompt_cache = make_prompt_cache(model)
            CH = 512
            logits = None
            for cs in range(0, len(prompt_ids), CH):
                chunk = mx.array(prompt_ids[cs:cs + CH])[None]
                logits = model(chunk, cache=prompt_cache)
                mx.eval(logits)
                
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
            
            generated = []
            for _ in range(gen):
                nid = int(y.item())
                generated.append(nid)
                if nid == tok.eos_token_id:
                    break
                logits = model(y[None], cache=prompt_cache)
                y = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(y)
            
            out = tok.decode(generated, skip_special_tokens=True)
        else:
            manager.clear_session(sid)
            manager._session_token_ids[sid] = []
            prompt = build_relational_prompt(name, tok, target_tokens=3500, spread=True, is_llama=is_llama)
            prompt_ids = tok.encode(prompt)
            
            manager.init_session(sid, prefill_len=len(prompt_ids))
            manager.register_prefill_tokens(sid, torch.tensor(prompt_ids, dtype=torch.long))
            model._dkv_session_ids = [sid]
            
            CH = 512
            output = None
            for cs in range(0, len(prompt_ids), CH):
                chunk = prompt_ids[cs:cs + CH]
                ct = torch.tensor([chunk], dtype=torch.long)
                pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
                output = model(ct, pt)
                manager.compress_deferred_prefill_blocks(sid)
            logits = output.logits[0, -1].cpu().numpy()
            
            cur = len(prompt_ids)
            generated = []
            for _ in range(gen):
                nid = int(np.argmax(logits))
                generated.append(nid)
                if nid == tok.eos_token_id:
                    break
                manager.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
                output = model(torch.tensor([[nid]], dtype=torch.long),
                               torch.tensor([[cur]], dtype=torch.long))
                logits = output.logits[0, -1].cpu().numpy()
                cur += 1
                
            out = tok.decode(generated, skip_special_tokens=True)
            
        correct = key in out
        num = key.split("-")[-1]
        num_correct = num in out
        others = [k for nm, k in NATURAL if k != key]
        other_nums = [k.split("-")[-1] for k in others]
        misbound = (not num_correct) and any(o in out for o in other_nums)
        n_correct += int(correct)
        n_misbound += int(misbound)
        results.append({
            "module": name,
            "want": key,
            "correct": correct,
            "num_correct": num_correct,
            "misbound": misbound,
            "out": out.strip()[:80]
        })
        
    n_num_correct = sum(int(r["num_correct"]) for r in results)
    
    return {
        "n_total": len(NATURAL),
        "n_correct": n_correct,
        "n_num_correct": n_num_correct,
        "n_misbound": n_misbound,
        "peak_mem_gb": _mx_peak_gb(),
        "results": results
    }

# ------------------------------------------------------------------------------
# MAIN EXECUTION ORCHESTRATOR
# ------------------------------------------------------------------------------

def run_isolated_test(model_id, mode, test_name, ctx):
    cmd = [
        sys.executable,
        __file__,
        "--run-sub",
        "--model", model_id,
        "--mode", mode,
        "--test", test_name,
        "--ctx", str(ctx)
    ]
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    if res.returncode != 0:
        return {"status": "error", "error": f"Exit code {res.returncode}", "stderr": res.stderr}
        
    try:
        stdout_str = res.stdout.strip()
        json_start = stdout_str.find("{")
        if json_start == -1:
            raise ValueError(f"No JSON found. stdout: {stdout_str}")
        return json.loads(stdout_str[json_start:])
    except Exception as e:
        return {"status": "error", "error": str(e), "stdout": res.stdout, "stderr": res.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sub", action="store_true", help="run as isolated sub-worker")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--mode", choices=["dense", "active"], default="active")
    parser.add_argument("--test", choices=["multi_needle", "synthesis", "relational"], default="multi_needle")
    parser.add_argument("--ctx", type=int, default=4096)
    args = parser.parse_args()
    
    if args.run_sub:
        # Set environment variables BEFORE importing/loading anything
        if args.mode == "active":
            os.environ["DKV_COMPRESSED_DECODE"] = "1"
            os.environ["DKV_ENGAGE_THRESHOLD"] = "1024"
        else:
            os.environ["DKV_COMPRESSED_DECODE"] = "0"
            
        if args.test == "synthesis" and args.mode == "active":
            os.environ["DKV_SPARSE_BIAS"] = "2.0"

        if args.test == "multi_needle":
            result = run_multi_needle_sub(args.model, args.mode, args.ctx)
        elif args.test == "synthesis":
            result = run_synthesis_sub(args.model, args.mode, args.ctx)
        elif args.test == "relational":
            result = run_relational_sub(args.model, args.mode)
        else:
            result = {"status": "unknown test"}
        print(json.dumps(result))
        sys.exit(0)
        
    print("=" * 70)
    print("STARTING RIGOROUS EVALUATION RUNNER (ISOLATED PROCESSES)")
    print("=" * 70)
    
    models = [
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    ]
    modes = ["dense", "active"]
    
    results = {}
    
    for model in models:
        model_name = "Qwen2.5-1.5B" if "Qwen" in model else "Llama-3.2-3B"
        for mode in modes:
            print(f"\n>>> Model: {model_name} | Mode: {mode.upper()}")
            
            # 1. Multi-needle NIAH
            results[(model_name, mode, "multi_needle")] = {}
            for ctx in [4096, 8192, 16384, 32768]:
                ctx_k = f"{ctx // 1024}k"
                print(f"  Running Multi-needle NIAH at {ctx_k}...")
                res = run_isolated_test(model, mode, "multi_needle", ctx)
                if "error" in res:
                    print(f"    ERROR: {res['error']}")
                    results[(model_name, mode, "multi_needle")][ctx_k] = "ERR/OOM"
                else:
                    recall_str = "Y" if res.get("ok") else "N"
                    tps = res.get("decode_tps", 0.0)
                    print(f"    Recall: {recall_str} | TPS: {tps:.1f}")
                    results[(model_name, mode, "multi_needle")][ctx_k] = f"{recall_str} ({tps:.1f} tps)"
                    
            # 2. Synthesis
            results[(model_name, mode, "synthesis")] = {}
            for ctx in [8192, 16384]:
                ctx_k = f"{ctx // 1024}k"
                print(f"  Running Synthesis at {ctx_k}...")
                res = run_isolated_test(model, mode, "synthesis", ctx)
                if "error" in res:
                    print(f"    ERROR: {res['error']}")
                    results[(model_name, mode, "synthesis")][ctx_k] = "ERR/OOM"
                else:
                    score = res.get("score", 0.0)
                    facts = res.get("facts", 0)
                    links = res.get("links", 0)
                    print(f"    Score: {score:.1f}/100 (Facts: {facts}/15, Linkages: {links}/5)")
                    results[(model_name, mode, "synthesis")][ctx_k] = f"{score:.1f}/100"
                    
            # 3. Relational AB
            print("  Running Relational AB...")
            res = run_isolated_test(model, mode, "relational", 3500)
            if "error" in res:
                print(f"    ERROR: {res['error']}")
                results[(model_name, mode, "relational")] = "ERR/OOM"
            else:
                n_corr = res.get("n_correct", 0)
                n_tot = res.get("n_total", 4)
                print(f"    Correct: {n_corr}/{n_tot}")
                results[(model_name, mode, "relational")] = f"{n_corr}/{n_tot}"

    report = []
    report.append("# Rigorous Evaluation Report (Active Runtime)\n")
    report.append("| Model | Mode | Multi-Needle 4k | Multi-Needle 8k | Multi-Needle 16k | Multi-Needle 32k | Synthesis 8k | Synthesis 16k | Relational AB |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    
    for model in ["Qwen2.5-1.5B"]:
        for mode in ["dense", "active"]:
            mn = results[(model, mode, "multi_needle")]
            syn = results[(model, mode, "synthesis")]
            rel = results[(model, mode, "relational")]
            
            report.append(
                f"| {model} | {mode.upper()} | "
                f"{mn.get('4k', '—')} | {mn.get('8k', '—')} | {mn.get('16k', '—')} | {mn.get('32k', '—')} | "
                f"{syn.get('8k', '—')} | {syn.get('16k', '—')} | {rel} |"
            )
            
    report_text = "\n".join(report)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(report_text)
    
    out_file = os.path.join(HERE, "results", "rigorous_eval_report.md")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write(report_text)
    print(f"\nSaved report to {out_file}")


if __name__ == "__main__":
    main()
