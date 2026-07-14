#!/usr/bin/env python3
"""Neighborhood Attention Transformer Paper Evaluation Script.

This script runs comparative prompt evaluations of standard Dense attention against
various DiffKV (Differential KV) configurations (presets, early rank boost, factual store)
using Qwen/Qwen2.5-7B-Instruct. It logs VRAM metrics, prefill time, decode tokens per second (TPS),
and generated quality comparisons.
"""

import os
import sys
import ssl
import urllib3
import requests
from urllib3.exceptions import InsecureRequestWarning

# Global absolute SSL verification bypass for all python libraries (including urllib3, requests, HF)
urllib3.disable_warnings(InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey-patch urllib3 to always use unverified SSL context
import urllib3.util.ssl_
urllib3.util.ssl_.create_urllib3_context = lambda *args, **kwargs: ssl._create_unverified_context()

# Monkey-patch requests
old_merge_settings = requests.Session.merge_environment_settings
def patched_merge_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings
requests.Session.merge_environment_settings = patched_merge_settings

import json
import time
import argparse
import gc
import re
import math
import subprocess

# Add required paths to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

import torch

# Define Prompts
PROMPT1_TEXT = """Use only the supplied text.

Explain why Neighborhood Attention is designed around each token's nearest neighbors rather than fixed non-overlapping windows.

Your answer must explain the reasoning chain:
- the boundary behavior of zero-padded local attention,
- the restriction created by non-overlapping window partitions,
- how per-token neighborhoods change the attention span,
- what Neighborhood Attention approaches as the neighborhood size reaches its maximum,
- and why shifted windows are needed in one design but not as a manual operation in Neighborhood Attention.

Do not summarize the entire work.
Do not discuss benchmark accuracy, datasets, CUDA, or implementation speed.
Do not introduce mechanisms not explicitly stated in the supplied text.

Write one coherent technical explanation of 120 to 180 words."""

PROMPT2_TEXT = """Use only the supplied text.

A researcher claims:
"Neighborhood Attention is simply Window Self-Attention with overlapping windows."

Evaluate this claim.

Identify which parts of the claim are superficially plausible and then explain precisely why the definition of Neighborhood Attention is different.

Your explanation must distinguish the attention span of an individual token from the partitioning of the entire feature map.

Do not use outside knowledge.
Do not discuss experimental results.
Maximum 180 words."""

PAPER_PATH = os.path.join(ACTIVE, "nat_paper.txt")

def format_prompt(paper_content, prompt_instructions):
    return f"""<|im_start|>system
You are a helpful assistant. Answer the user's request strictly using the provided context.<|im_end|>
<|im_start|>user
Provided Text:
{paper_text}

Instructions:
{prompt_instructions}<|im_end|>
<|im_start|>assistant
"""

def analytic_kv_bytes(mgr, seq_len, sid):
    """Calculate the footprint of the DiffKV Cache."""
    L = mgr.num_layers
    Hkv = mgr.kv_heads
    d = mgr.head_dim
    B = mgr.block_size
    r = mgr.rank
    M = mgr.max_blocks
    Dmax = mgr.max_dense_len
    R = mgr.max_residual
    fp16 = 2
    
    kv_tok = Hkv * d * fp16 * 2
    lowrank_block = ((B - 1) * r * fp16
                     + 2 * Hkv * r * d * fp16
                     + 2 * Hkv * d * fp16
                     + 2 * Hkv * d * fp16
                     + 8)
    residual_block_max = R * kv_tok
    per_block = lowrank_block + residual_block_max
    dense_alloc = Dmax * kv_tok
    
    s0 = mgr.sessions.get(sid)
    nb = s0["num_blocks"][0] if s0 else 0
    dl = s0["dense_lens"][0] if s0 else 0
    res_n0 = s0["comp_res_n"][0][:nb] if s0 else []
    res_tokens_used = int(sum(res_n0))
    
    store_used = L * (nb * lowrank_block + res_tokens_used * kv_tok + dl * kv_tok)
    return {
        "store_used_bytes": store_used,
    }

def run_worker(config_name, model_id):
    # Setup environment variables
    os.environ["DIFFKV_FACTUAL_STORE"] = "0"
    os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "0"
    
    if config_name == "dense":
        os.environ["DIFFKV_COMPRESSED_DECODE"] = "0"
        is_compressed = False
    else:
        os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
        is_compressed = True
        
        if config_name == "low_preset":
            os.environ["DIFFKV_PRESET"] = "low"
        elif config_name == "mid_preset":
            os.environ["DIFFKV_PRESET"] = "mid"
        elif config_name == "high_preset":
            os.environ["DIFFKV_PRESET"] = "high"
        elif config_name == "early_boost":
            os.environ["DIFFKV_PRESET"] = "mid"
            os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "1"
        elif config_name == "factual_store":
            os.environ["DIFFKV_PRESET"] = "mid"
            os.environ["DIFFKV_FACTUAL_STORE"] = "1"
        elif config_name == "combined":
            os.environ["DIFFKV_PRESET"] = "mid"
            os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "1"
            os.environ["DIFFKV_FACTUAL_STORE"] = "1"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = "cuda:0"
    else:
        device = "cpu"

    if not os.path.exists(PAPER_PATH):
        raise FileNotFoundError(f"Context paper file not found at {PAPER_PATH}")

    with open(PAPER_PATH, "r", encoding="utf-8") as f:
        paper_text = f.read()

    results = {}
    
    # Wrap in torch.inference_mode() to prevent OOM
    with torch.inference_mode():
        for idx, prompt_instructions in enumerate([PROMPT1_TEXT, PROMPT2_TEXT], 1):
            # Create full prompt (pass paper_text)
            full_prompt = f"<|im_start|>system\nYou are a helpful assistant. Answer the user's request strictly using the provided context.<|im_end|>\n<|im_start|>user\nProvided Text:\n{paper_text}\n\nInstructions:\n{prompt_instructions}<|im_end|>\n<|im_start|>assistant\n"
            
            if is_compressed:
                from serving.hf_diffkv_wrapper import DiffKVHFWrapper
                cfg = {
                    "preset": os.environ.get("DIFFKV_PRESET", "mid"),
                    "serving_mode": "balanced"
                }
                if config_name in ["early_boost", "combined"]:
                    cfg["early_layer_rank_boost"] = True
                if config_name in ["factual_store", "combined"]:
                    cfg["factual_store"] = True
                    
                w = DiffKVHFWrapper(
                    model_id=model_id,
                    config=cfg,
                    torch_dtype=torch.float16,
                    device=device
                )
                w.ensure_loaded()
                
                tok, mgr, model = w.tokenizer, w.manager, w.model
                ids = tok.encode(full_prompt)
                prompt_len = len(ids)
                
                # Reset peak VRAM for prefill
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    
                # Prefill (chunked)
                sid = f"prompt_{idx}"
                mgr.clear_session(sid)
                if not hasattr(w, "_session_token_ids"):
                    w._session_token_ids = {}
                w._session_token_ids[sid] = []
                
                mgr.init_session(sid, prefill_len=prompt_len)
                mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
                model._diffkv_session_ids = [sid]
                
                CH = 512
                t_prefill_start = time.perf_counter()
                for cs in range(0, len(ids), CH):
                    ch = ids[cs:cs+CH]
                    out = model(torch.tensor([ch], device=device), torch.tensor([list(range(cs, cs+len(ch)))], device=device))
                    mgr.compress_deferred_prefill_blocks(sid)
                logits = out.logits[0, -1].float().cpu().numpy()
                prefill_time = time.perf_counter() - t_prefill_start
                
                peak_prefill_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                
                # Isolate decode VRAM
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    
                cur = prompt_len
                gen_ids = []
                t_decode_start = time.perf_counter()
                for _ in range(256):
                    import numpy as np
                    nid = int(np.argmax(logits))
                    if nid in w.stop_token_ids:
                        break
                    gen_ids.append(nid)
                    mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long, device=device))
                    out = model(torch.tensor([[nid]], device=device), torch.tensor([[cur]], device=device))
                    logits = out.logits[0, -1].float().cpu().numpy()
                    cur += 1
                decode_time = time.perf_counter() - t_decode_start
                peak_decode_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                
                generated_text = tok.decode(gen_ids)
                
                # Calculate memory
                kv = analytic_kv_bytes(mgr, prompt_len, sid)
                kv_vram = kv.get("store_used_bytes", 0) / 1e9
                
                try:
                    w.close()
                except Exception:
                    pass
                del w, model, mgr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Eager Dense baseline
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device)
                model.eval()
                
                ids = tok.encode(full_prompt)
                prompt_len = len(ids)
                
                # Reset peak VRAM for prefill
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    
                t_prefill_start = time.perf_counter()
                past_key_values = None
                CH = 512
                for cs in range(0, len(ids), CH):
                    ch = ids[cs:cs+CH]
                    pos = torch.tensor([list(range(cs, cs+len(ch)))], device=device)
                    out = model(torch.tensor([ch], device=device), position_ids=pos, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                logits = out.logits[0, -1].float().cpu().numpy()
                prefill_time = time.perf_counter() - t_prefill_start
                
                peak_prefill_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                
                # Isolate decode VRAM
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    
                cur = prompt_len
                gen_ids = []
                t_decode_start = time.perf_counter()
                for _ in range(256):
                    import numpy as np
                    nid = int(np.argmax(logits))
                    if nid in [tok.eos_token_id, tok.pad_token_id] or tok.decode([nid]) in ["<|im_end|>", "</s>"]:
                        break
                    gen_ids.append(nid)
                    pos = torch.tensor([[cur]], device=device)
                    out = model(torch.tensor([[nid]], device=device), position_ids=pos, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                    logits = out.logits[0, -1].float().cpu().numpy()
                    cur += 1
                decode_time = time.perf_counter() - t_decode_start
                peak_decode_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                
                generated_text = tok.decode(gen_ids)
                
                # Calculate memory
                L = model.config.num_hidden_layers
                Hkv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
                d = model.config.hidden_size // model.config.num_attention_heads
                fp16 = 2
                kv_vram = (L * prompt_len * Hkv * d * fp16 * 2) / 1e9
                
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            results[f"prompt{idx}"] = {
                "prompt_len": prompt_len,
                "generated_tokens": len(gen_ids),
                "prefill_time_s": prefill_time,
                "decode_time_s": decode_time,
                "decode_tps": len(gen_ids) / decode_time if decode_time > 0 else 0.0,
                "peak_prefill_vram_gb": peak_prefill_vram,
                "peak_decode_vram_gb": peak_decode_vram,
                "kv_cache_vram_gb": kv_vram,
                "output_text": generated_text
            }
        
    return results

def generate_report(all_results, model_id):
    from tabulate import tabulate
    report_dir = os.path.join(REPO, "colab")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "nat_evaluation_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Neighborhood Attention Paper Evaluation Report ({model_id})\n\n")
        f.write("This report compares the performance, memory usage, and generation quality of standard **Dense Attention** against **Differential KV (DiffKV)** under various presets, early layer rank boosting, and factual store routing on an A100 GPU.\n\n")
        
        for p_idx in [1, 2]:
            p_key = f"prompt{p_idx}"
            f.write(f"## Prompt {p_idx} Performance & Resource Comparison\n\n")
            
            headers = ["Config", "Prefill Time (s)", "Decode TPS", "Peak Prefill VRAM (GB)", "Peak Decode VRAM (GB)", "KV Cache VRAM (GB)", "Gen Tokens"]
            rows = []
            
            for cfg_name, cfg_res in all_results.items():
                if cfg_res.get("status") == "failed":
                    rows.append([cfg_name, "FAILED", "N/A", "N/A", "N/A", "N/A", "N/A"])
                    continue
                p_res = cfg_res.get(p_key, {})
                rows.append([
                    cfg_name,
                    f"{p_res.get('prefill_time_s', 0):.3f}s",
                    f"{p_res.get('decode_tps', 0):.2f}",
                    f"{p_res.get('peak_prefill_vram_gb', 0):.2f} GB",
                    f"{p_res.get('peak_decode_vram_gb', 0):.2f} GB",
                    f"{p_res.get('kv_cache_vram_gb', 0):.3f} GB",
                    str(p_res.get("generated_tokens", 0))
                ])
            
            f.write(tabulate(rows, headers=headers, tablefmt="github") + "\n\n")
            
        f.write("## Quality and Output Analysis\n\n")
        for p_idx in [1, 2]:
            p_key = f"prompt{p_idx}"
            f.write(f"### Responses to Prompt {p_idx}\n\n")
            
            for cfg_name, cfg_res in all_results.items():
                if cfg_res.get("status") == "failed":
                    continue
                p_res = cfg_res.get(p_key, {})
                text = p_res.get("output_text", "").strip()
                word_count = len(text.split())
                
                f.write(f"#### Configuration: `{cfg_name}` ({word_count} words)\n")
                f.write(f"> {text}\n\n")
                
    print(f"Generated Markdown report at {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="run worker mode for a specific config")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="nat_eval_results.json")
    args = parser.parse_args()

    if args.worker:
        results = run_worker(args.worker, args.model)
        temp_file = f"temp_res_{args.worker}.json"
        with open(temp_file, "w") as f:
            json.dump(results, f)
        return

    configs = [
        "dense",
        "low_preset",
        "mid_preset",
        "high_preset",
        "early_boost",
        "factual_store",
        "combined"
    ]
    all_results = {}

    print(f"=== Starting Neighborhood Attention Paper Evaluation with {args.model} ===", flush=True)

    for cfg in configs:
        print(f"\n>>> Running configuration: {cfg}", flush=True)
        env = os.environ.copy()
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", cfg, "--model", args.model]
        
        p = subprocess.run(cmd, env=env)
        
        temp_file = f"temp_res_{cfg}.json"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                res = json.load(f)
            os.remove(temp_file)
            all_results[cfg] = res
            
            for p_key in ["prompt1", "prompt2"]:
                p_res = res.get(p_key, {})
                print(f"    {p_key} Success: tokens={p_res.get('generated_tokens')}, prefill={p_res.get('prefill_time_s',0):.2f}s, tps={p_res.get('decode_tps',0):.1f}, kv_mem={p_res.get('kv_cache_vram_gb',0):.3f}GB", flush=True)
        else:
            print(f"    FAILED: Subprocess for {cfg} exited without writing results file.", flush=True)
            all_results[cfg] = {"status": "failed", "stderr": "No results temp file written."}

    # Save raw results
    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote raw results to {out_path}", flush=True)

    # Generate visual markdown report
    generate_report(all_results, args.model)

if __name__ == "__main__":
    main()
