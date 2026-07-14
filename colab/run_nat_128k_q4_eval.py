#!/usr/bin/env python3
"""128K Context Evaluation for Qwen2.5-14B in 4-bit.

This script constructs a 128K-token prompt by repeating the 'nat_paper.txt' context
to fill the sequence budget, inserts the custom evaluation prompt at the end,
and benchmarks 4-bit Dense against 4-bit DiffKV on an A100 GPU.
"""

import os
import sys
import ssl
import urllib3
import requests
from urllib3.exceptions import InsecureRequestWarning

# Global SSL verification bypass for firewalls/proxies inside the worker processes
urllib3.disable_warnings(InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

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
import subprocess

# Ensure active runtime path is in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

import torch

# New Prompt
CUSTOM_PROMPT = """Use only the supplied text.

A researcher claims:

"Neighborhood Attention is simply Window Self-Attention with overlapping windows."

Evaluate this claim.

Identify why the claim may initially sound plausible, then explain precisely why it is incorrect.

Your explanation must distinguish:
- partitioning the feature map into windows,
- the attention span assigned to an individual token,
- boundary behavior,
- receptive-field expansion,
- and the role of shifted windows.

Do not use outside knowledge.
Do not discuss experimental accuracy, datasets, CUDA, or implementation speed.
Do not introduce mechanisms not explicitly stated in the supplied text.

Write one coherent technical explanation of 150 to 220 words."""

PAPER_PATH = os.path.join(ACTIVE, "nat_paper.txt")

def build_128k_prompt(tokenizer, target_len=131072):
    with open(PAPER_PATH, "r", encoding="utf-8") as f:
        paper_content = f.read()

    system_prefix = "<|im_start|>system\nYou are a helpful assistant. Answer the user's request strictly using the provided context.<|im_end|>\n<|im_start|>user\nProvided Text:\n"
    suffix = f"\n\nInstructions:\n{CUSTOM_PROMPT}<|im_end|>\n<|im_start|>assistant\n"

    system_overhead = len(tokenizer.encode(system_prefix))
    suffix_overhead = len(tokenizer.encode(suffix))
    available_budget = target_len - system_overhead - suffix_overhead

    paper_tokens = tokenizer.encode(paper_content)
    n_paper_tokens = len(paper_tokens)

    reps = available_budget // n_paper_tokens
    haystack_tokens = paper_tokens * reps

    rem = available_budget - len(haystack_tokens)
    if rem > 0:
        haystack_tokens += paper_tokens[:rem]

    haystack_text = tokenizer.decode(haystack_tokens)
    return f"{system_prefix}{haystack_text}{suffix}"

def run_worker(mode, model_id):
    os.environ["DIFFKV_FACTUAL_STORE"] = "0"
    os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "0"
    
    is_compressed = (mode == "compressed")
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if is_compressed else "0"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Building 128K prompt for {model_id}...", flush=True)
    full_prompt = build_128k_prompt(tokenizer, target_len=131072)

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    t0 = time.perf_counter()

    if is_compressed:
        from serving.hf_diffkv_wrapper import DiffKVHFWrapper
        config = {
            "mode": "fp16",
            "quantization": "nf4",
            "block_size": 256,
            "rank": 16,
            "micro_block_size": 256,
            "preset": "mid",
            "serving_mode": "balanced"
        }
        w = DiffKVHFWrapper(
            model_id=model_id,
            config=config,
            torch_dtype=torch.float16,
            device=device,
            quantization_config=quantization_config
        )
        w.ensure_loaded()
        tok, mgr, model = w.tokenizer, w.manager, w.model

        ids = tok.encode(full_prompt)
        prompt_len = len(ids)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        sid = "nat_128k"
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

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Generate (256 tokens)
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

        L = mgr.num_layers
        Hkv = mgr.kv_heads
        d = mgr.head_dim
        B = mgr.block_size
        r = mgr.rank
        fp16 = 2
        
        kv_tok = Hkv * d * fp16 * 2
        lowrank_block = ((B - 1) * r * fp16
                         + 2 * Hkv * r * d * fp16
                         + 2 * Hkv * d * fp16
                         + 2 * Hkv * d * fp16
                         + 8)
        s0 = mgr.sessions.get(sid)
        nb = s0["num_blocks"][0] if s0 else 0
        dl = s0["dense_lens"][0] if s0 else 0
        res_n0 = s0["comp_res_n"][0][:nb] if s0 else []
        res_tokens_used = int(sum(res_n0))
        kv_vram = (L * (nb * lowrank_block + res_tokens_used * kv_tok + dl * kv_tok)) / 1e9

        try: w.close()
        except: pass
        del w, model, mgr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        ids = tokenizer.encode(full_prompt)
        prompt_len = len(ids)

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

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Generate (256 tokens)
        cur = prompt_len
        gen_ids = []
        t_decode_start = time.perf_counter()
        for _ in range(256):
            import numpy as np
            nid = int(np.argmax(logits))
            if nid in [tokenizer.eos_token_id, tokenizer.pad_token_id] or tokenizer.decode([nid]) in ["<|im_end|>", "</s>"]:
                break
            gen_ids.append(nid)
            pos = torch.tensor([[cur]], device=device)
            out = model(torch.tensor([[nid]], device=device), position_ids=pos, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[0, -1].float().cpu().numpy()
            cur += 1
        decode_time = time.perf_counter() - t_decode_start
        peak_decode_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        generated_text = tokenizer.decode(gen_ids)

        L = model.config.num_hidden_layers
        Hkv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        d = model.config.hidden_size // model.config.num_attention_heads
        fp16 = 2
        kv_vram = (L * prompt_len * Hkv * d * fp16 * 2) / 1e9

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = {
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
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="dense or compressed")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    args = parser.parse_args()

    if args.worker:
        res = run_worker(args.worker, args.model)
        temp_file = f"temp_res_{args.worker}.json"
        with open(temp_file, "w") as f:
            json.dump(res, f)
        return

    print("=== Launching 128K Qwen2.5-14B-Instruct Q4 Benchmark ===", flush=True)

    results = {}
    for mode in ["dense", "compressed"]:
        print(f"\n>>> Running: {mode} mode", flush=True)
        env = os.environ.copy()
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", mode, "--model", args.model]
        
        # Stream stdout in real-time
        p = subprocess.run(cmd, env=env)

        temp_file = f"temp_res_{mode}.json"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                res = json.load(f)
            os.remove(temp_file)
            results[mode] = res
            print(f"    Prefill={res['prefill_time_s']:.2f}s, TPS={res['decode_tps']:.1f}, Cache VRAM={res['kv_cache_vram_gb']:.3f}GB, Peak VRAM={res['peak_decode_vram_gb']:.2f}GB", flush=True)
        else:
            print(f"    FAILED: Subprocess for {mode} exited without writing results file.", flush=True)

    # Compile Markdown Report
    report_path = os.path.join(REPO, "colab", "nat_128k_q4_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Qwen2.5-14B-Instruct (4-bit) 128K context evaluation\n\n")
        f.write("This report benchmarks Standard Dense (4-bit quantized weights) against Differential KV (4-bit quantized weights + Compressed KV cache) on an A100 GPU over a 128,000 token sequence.\n\n")
        
        f.write("## Performance & Resource Summary\n\n")
        headers = ["Mode", "Prefill Time (s)", "Decode TPS", "Peak Prefill VRAM (GB)", "Peak Decode VRAM (GB)", "KV Cache VRAM (GB)", "Tokens Gen"]
        rows = []
        for mode in ["dense", "compressed"]:
            res = results.get(mode, {})
            if not res:
                rows.append([mode, "FAILED", "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue
            rows.append([
                mode,
                f"{res.get('prefill_time_s', 0):.2f}s",
                f"{res.get('decode_tps', 0):.2f}",
                f"{res.get('peak_prefill_vram_gb', 0):.2f} GB",
                f"{res.get('peak_decode_vram_gb', 0):.2f} GB",
                f"{res.get('kv_cache_vram_gb', 0):.3f} GB",
                str(res.get("generated_tokens", 0))
            ])
            
        from tabulate import tabulate
        f.write(tabulate(rows, headers=headers, tablefmt="github") + "\n\n")
        
        f.write("## Generated Responses Side-by-Side\n\n")
        for mode in ["dense", "compressed"]:
            res = results.get(mode, {})
            if not res:
                continue
            text = res.get("output_text", "").strip()
            word_count = len(text.split())
            f.write(f"### Mode: `{mode}` ({word_count} words)\n")
            f.write(f"> {text}\n\n")

    print(f"\nSUCCESS: Report compiled at {report_path}")

if __name__ == "__main__":
    main()
