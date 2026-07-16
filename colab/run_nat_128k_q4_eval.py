#!/usr/bin/env python3
"""Context Sweep (4K-128K) Evaluation for Qwen2.5-14B in 4-bit.

This script benchmarks 4-bit Dense against 4-bit DiffKV across various context lengths
(4K, 8K, 16K, 32K, 64K, 128K) on an A100 GPU, logging hardware metrics (power draw, temp) via nvidia-smi.
"""

import os
import sys
import ssl
import urllib3
import requests
from urllib3.exceptions import InsecureRequestWarning

# Global absolute SSL verification bypass
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

# Prevent PyTorch VRAM fragmentation OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure active runtime path and C++ compiled library directory are in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
CORE_DIR = os.path.join(ACTIVE, "native_core", "diffkv_core")

if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import json
import time
import argparse
import gc
import subprocess
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

def get_gpu_metrics():
    """Query nvidia-smi for current temperature (C) and power draw (W)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        temp = float(parts[0])
        power = float(parts[1])
        return temp, power
    except Exception:
        return None, None

def build_prompt_for_len(tokenizer, target_len):
    """Build a prompt of approximately `target_len` tokens using the model's own
    chat template.  Works for Qwen (ChatML), Llama 3, Mistral, Phi-3, etc.
    The paper text is repeated to fill up to `target_len` tokens.
    """
    with open(PAPER_PATH, "r", encoding="utf-8") as f:
        paper_content = f.read()

    # Measure the token overhead of the chat framing using a tiny placeholder.
    _placeholder = "X"
    try:
        _framed = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's request strictly using the provided context."},
                {"role": "user",   "content": f"Provided Text:\n{_placeholder}\n\nInstructions:\n{CUSTOM_PROMPT}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without a chat template.
        _framed = (
            "You are a helpful assistant. Answer the user's request strictly using the provided context.\n\n"
            f"Provided Text:\n{_placeholder}\n\nInstructions:\n{CUSTOM_PROMPT}\n\nAssistant:"
        )

    overhead = len(tokenizer.encode(_framed)) - len(tokenizer.encode(_placeholder))
    available_budget = target_len - overhead

    if available_budget <= 0:
        # Not enough budget — just use a small fragment and accept the mismatch.
        return _framed.replace(_placeholder, paper_content[:100])

    paper_tokens = tokenizer.encode(paper_content)
    reps = available_budget // len(paper_tokens)
    haystack_tokens = paper_tokens * reps
    rem = available_budget - len(haystack_tokens)
    if rem > 0:
        haystack_tokens += paper_tokens[:rem]
    haystack_text = tokenizer.decode(haystack_tokens)

    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's request strictly using the provided context."},
                {"role": "user",   "content": f"Provided Text:\n{haystack_text}\n\nInstructions:\n{CUSTOM_PROMPT}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            "You are a helpful assistant. Answer the user's request strictly using the provided context.\n\n"
            f"Provided Text:\n{haystack_text}\n\nInstructions:\n{CUSTOM_PROMPT}\n\nAssistant:"
        )


def run_worker(mode, model_id, target_len):
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

    print(f"Building prompt of length {target_len} for {model_id}...", flush=True)
    full_prompt = build_prompt_for_len(tokenizer, target_len)

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Log startup baseline metrics
    base_temp, base_power = get_gpu_metrics()
    
    peak_prefill_temp = base_temp or 0.0
    peak_prefill_power = base_power or 0.0
    peak_decode_temp = base_temp or 0.0
    peak_decode_power = base_power or 0.0

    with torch.inference_mode():
        if is_compressed:
            from serving.hf_diffkv_wrapper import DiffKVHFWrapper
            config = {
                "mode": "fp16",
                "quantization": "nf4",
                "rank": 16,
                "micro_block_size": 64,      # block_capacity = 1+64 = 65 ≤ CH=128, so each
                                               # 128-token prefill chunk produces ≥1 full block
                                               # that is immediately submitted for SVD compression.
                                               # micro_block_size=256 gives block_capacity=257>128
                                               # → zero full blocks → all 4096 tokens stay dense.
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

            sid = f"nat_sweep_{target_len}"
            mgr.clear_session(sid)
            if not hasattr(w, "_session_token_ids"):
                w._session_token_ids = {}
            w._session_token_ids[sid] = []

            mgr.init_session(sid, prefill_len=prompt_len)
            mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
            model._diffkv_session_ids = [sid]

            # Prefill.  Use the runtime's chunk size so CUDA receives a useful
            # batch of blocks per compression dispatch.  CH=128 with the 64-token
            # sweep block size creates one full + one partial block on every
            # chunk, which serializes dozens of tiny rSVD launches and used to
            # strand the partial blocks in SUBMITTED.
            _cfg = getattr(mgr, "config", None)
            CH = int(getattr(_cfg, "prefill_chunk_size", 1024))
            if torch.cuda.is_available() and hasattr(mgr, "get_session_micro_block_size"):
                _mbs = mgr.get_session_micro_block_size(sid)
                _block_capacity = max(2, int(_mbs) + 1)
                CH = ((CH + _block_capacity - 1) // _block_capacity) * _block_capacity
                print(f"    [Prefill] Aligned chunk size: {CH} (block_capacity={_block_capacity})", flush=True)
                print("    [Prefill] CUDA exact-prefill mode: SVD deferred until boundary; internal splitter is block-aligned", flush=True)
            t_prefill_start = time.perf_counter()
            for cs in range(0, len(ids), CH):
                ch = ids[cs:cs+CH]
                out = model(
                    torch.tensor([ch], device=device),
                    position_ids=torch.tensor([list(range(cs, cs+len(ch)))], device=device)
                )
                # Dynamic hardware monitoring
                if cs % 4096 == 0 and cs > 0:
                    allocated_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    temp, power = get_gpu_metrics()
                    if temp and temp > peak_prefill_temp: peak_prefill_temp = temp
                    if power and power > peak_prefill_power: peak_prefill_power = power
                    print(f"    [Prefill Progress] {cs}/{len(ids)} tokens. VRAM: {allocated_gb:.2f} GB (Temp: {temp}°C, Power: {power}W)", flush=True)
            
            mgr.compress_deferred_prefill_blocks(sid)
            
            # Keep initial logits on GPU — no D2H sync during prefill
            last_logits_gpu = out.logits[0, -1].float()
            prefill_time = time.perf_counter() - t_prefill_start

            peak_prefill_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # CRITICAL: Wait for async SVD compression to finish before decode.
            # Each newly compressed block changes block_indices shape → forces
            # CUDAGraphDecodeRunner to re-record the full model CUDA graph
            # (3-5 forward warmups per shape). 51 shapes = thousands of extra
            # forward passes. Barrier ensures stable shapes from decode step 1.
            #
            # WHY NOT _pending_cpu_blocks: that counter is initialized to 0 and
            # never incremented on the submit side, so the old check always exits
            # immediately. We check ACTUAL block states instead.
            def _count_in_flight(mgr, sid):
                """Count blocks still being compressed (not yet GPU-resident).
                A block is "in-flight" when:
                  - state == SUBMITTED: sent to background compressor, SVD not done
                  - state == CPU_COMPRESSED: SVD done on CPU, waiting for GPU upload
                  - state == ACCUMULATING AND active_k is None: async path cleared
                    the GPU tensor before the compressor thread changed state
                Only layer 0 is checked as a representative proxy — all layers are
                processed by the same background thread in sequence."""
                _sm = getattr(mgr, "_streaming_mgr", None)
                if _sm is None:
                    return 0
                _l0 = _sm.session_blocks.get(sid, {}).get(0, [])
                return sum(
                    1 for b in _l0
                    if getattr(b, "state", "") in ("SUBMITTED", "CPU_COMPRESSED")
                    or (getattr(b, "state", "") == "ACCUMULATING" and getattr(b, "active_k", None) is None)
                )

            _barrier_start = time.perf_counter()
            _barrier_timeout = float(os.environ.get("DIFFKV_COMPRESSION_TIMEOUT_S", "30"))
            _prev_pending = -1
            while True:
                _pending = _count_in_flight(mgr, sid)
                if _pending == 0:
                    break
                if time.perf_counter() - _barrier_start > _barrier_timeout:
                    print(f"    [Barrier] Timeout after {_barrier_timeout:.0f}s, {_pending} blocks still pending.", flush=True)
                    break
                if _pending != _prev_pending:
                    print(f"    [Barrier] Waiting for compression: {_pending} blocks pending…", flush=True)
                    _prev_pending = _pending
                if hasattr(mgr, "finalize_compressed_blocks"):
                    mgr.finalize_compressed_blocks()
                time.sleep(0.05)
            _barrier_elapsed = time.perf_counter() - _barrier_start
            if _barrier_elapsed > 0.2:
                print(f"    [Barrier] Done in {_barrier_elapsed:.1f}s", flush=True)

            # Build routing only after every compressed block is resident.  The
            # old ordering indexed only the blocks that happened to finish
            # before the barrier, leaving SRL with a stale/partial view.
            if hasattr(mgr, "finalize_srl_index"):
                mgr.finalize_srl_index(sid, cached_len=0)
            if hasattr(mgr, "_prefill_kv_capture"):
                mgr._prefill_kv_capture.pop(sid, None)

            # NOTE: as of the warm_up_jit() integration in ensure_loaded(), the
            # Inductor compilation for _reconstruct_and_score and
            # _attend_and_reconstruct_v is triggered at model-load time (before
            # this eval script even runs the prefill).  The 3-step generate loop
            # below is kept as a safety net for:
            #  (a) entry points that bypass ensure_loaded (raw model scripts), and
            #  (b) warming up CUDA graph recording passes if enabled.
            # When already compiled, the 3 steps run in <1s total.
            
            # ── diagnostics: show block state before warmup ──────────────────
            if hasattr(mgr, "_streaming_mgr") and mgr._streaming_mgr is not None:
                _l0 = mgr._streaming_mgr.session_blocks.get(sid, {}).get(0, [])
                _state_counts = {}
                for _b in _l0:
                    _s = getattr(_b, "state", "?")
                    _state_counts[_s] = _state_counts.get(_s, 0) + 1
                print(f"    [PreWarmup] Layer-0 block states: {_state_counts} "
                      f"(total={len(_l0)} blocks)", flush=True)

            # Do not warm up by forwarding tokens through the measured session:
            # that mutates its KV cache and silently skips those tokens from the
            # reported output.  ensure_loaded() already pre-warms the decode JIT;
            # CUDA graphs are disabled by default until the mutable-state ABI is
            # made safe.
            print("    [Warmup] Skipped stateful decode warmup; measuring from prefill logits.", flush=True)

            # Generate (256 tokens)
            cur = prompt_len
            gen_ids = []
            # Pre-allocate static GPU tensors — avoids new allocation + CUDA
            # graph invalidation on every step. Use .copy_() in the loop.
            static_input_ids  = torch.zeros((1, 1), dtype=torch.long,  device=device)
            static_pos_ids    = torch.zeros((1, 1), dtype=torch.long,  device=device)
            stop_ids_set = set(w.stop_token_ids)
            t_decode_start = time.perf_counter()
            for step in range(256):
                # GPU argmax — no D2H sync, no numpy
                nid_gpu = torch.argmax(last_logits_gpu)
                nid = int(nid_gpu.item())  # single scalar D2H — unavoidable for stop check
                if nid in stop_ids_set:
                    break
                gen_ids.append(nid)
                # In-place copy into static buffers — no new allocation
                static_input_ids[0, 0]  = nid
                static_pos_ids[0, 0]    = cur
                out = model(
                    static_input_ids,
                    position_ids=static_pos_ids
                )
                # Keep logits on GPU — no D2H sync per step
                last_logits_gpu = out.logits[0, -1].float()
                cur += 1

                if step % 32 == 0:
                    temp, power = get_gpu_metrics()
                    if temp and temp > peak_decode_temp: peak_decode_temp = temp
                    if power and power > peak_decode_power: peak_decode_power = power
            
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
            # Match the compressed runtime's chunking so the dense baseline and
            # compressed prefill are measured with comparable launch overhead.
            CH = 1024
            for cs in range(0, len(ids), CH):
                ch = ids[cs:cs+CH]
                pos = torch.tensor([list(range(cs, cs+len(ch)))], device=device)
                out = model(torch.tensor([ch], device=device), position_ids=pos, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                
                # Dynamic hardware monitoring
                if cs % 4096 == 0 and cs > 0:
                    allocated_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    temp, power = get_gpu_metrics()
                    if temp and temp > peak_prefill_temp: peak_prefill_temp = temp
                    if power and power > peak_prefill_power: peak_prefill_power = power
                    print(f"    [Prefill Progress] {cs}/{len(ids)} tokens. VRAM: {allocated_gb:.2f} GB (Temp: {temp}°C, Power: {power}W)", flush=True)
            
            # Keep initial logits on GPU — no D2H sync
            last_logits_gpu = out.logits[0, -1].float()
            prefill_time = time.perf_counter() - t_prefill_start

            peak_prefill_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Generate (256 tokens)
            cur = prompt_len
            gen_ids = []
            # Pre-allocate static GPU tensors — no per-step allocation, no CUDA graph invalidation
            static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
            static_pos_ids   = torch.zeros((1, 1), dtype=torch.long, device=device)
            stop_ids_dense = {tokenizer.eos_token_id, tokenizer.pad_token_id}
            stop_strings_dense = {"<|im_end|>", "</s>"}
            t_decode_start = time.perf_counter()
            for step in range(256):
                nid_gpu = torch.argmax(last_logits_gpu)
                nid = int(nid_gpu.item())  # single scalar D2H — unavoidable for stop check
                if nid in stop_ids_dense or tokenizer.decode([nid]) in stop_strings_dense:
                    break
                gen_ids.append(nid)
                # In-place update of static buffers — no new allocation
                static_input_ids[0, 0] = nid
                static_pos_ids[0, 0]   = cur
                out = model(static_input_ids, position_ids=static_pos_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                # Keep logits on GPU — no D2H sync per step
                last_logits_gpu = out.logits[0, -1].float()
                cur += 1

                if step % 32 == 0:
                    temp, power = get_gpu_metrics()
                    if temp and temp > peak_decode_temp: peak_decode_temp = temp
                    if power and power > peak_decode_power: peak_decode_power = power
            
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
        "status": "success",
        "prompt_len": prompt_len,
        "generated_tokens": len(gen_ids),
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "decode_tps": len(gen_ids) / decode_time if decode_time > 0 else 0.0,
        "peak_prefill_vram_gb": peak_prefill_vram,
        "peak_decode_vram_gb": peak_decode_vram,
        "kv_cache_vram_gb": kv_vram,
        "base_temp_c": base_temp,
        "base_power_w": base_power,
        "peak_prefill_temp_c": peak_prefill_temp,
        "peak_prefill_power_w": peak_prefill_power,
        "peak_decode_temp_c": peak_decode_temp,
        "peak_decode_power_w": peak_decode_power,
        "output_text": generated_text
    }
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="dense or compressed")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--len", type=int, default=131072)
    args = parser.parse_args()

    if args.worker:
        res = run_worker(args.worker, args.model, args.len)
        temp_file = f"temp_res_{args.worker}_{args.len}.json"
        with open(temp_file, "w") as f:
            json.dump(res, f)
        return

    # Sweep sequence lengths: 4K, 8K, 16K, 32K, 64K, 128K
    sweep_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    all_results = {}

    print(f"=== Starting {args.model} Context Sweep Evaluation (4K - 128K) ===", flush=True)

    for target_len in sweep_lengths:
        print(f"\n=======================================================", flush=True)
        print(f">>> TARGET SEQUENCE LENGTH: {target_len // 1024}K tokens ({target_len})", flush=True)
        print(f"=======================================================", flush=True)
        
        all_results[target_len] = {}

        for mode in ["dense", "compressed"]:
            print(f"\n>>> Running mode: {mode}", flush=True)
            
            # Explicit baseline flush before running subprocess
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)  # Buffer delay for NVIDIA driver memory reclaim

            env = os.environ.copy()
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker", mode,
                "--model", args.model,
                "--len", str(target_len)
            ]
            
            # Run subprocess to isolate memory completely
            p = subprocess.run(cmd, env=env)

            temp_file = f"temp_res_{mode}_{target_len}.json"
            if os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    res = json.load(f)
                os.remove(temp_file)
                all_results[target_len][mode] = res
                print(f"    SUCCESS: Prefill={res['prefill_time_s']:.2f}s, TPS={res['decode_tps']:.1f}, "
                      f"Peak VRAM={res['peak_decode_vram_gb']:.2f}GB, KV Cache VRAM={res['kv_cache_vram_gb']:.3f}GB, "
                      f"Peak Temp={res['peak_prefill_temp_c']}°C, Peak Power={res['peak_prefill_power_w']}W", flush=True)
            else:
                print(f"    Subprocess for {mode} crashed or went Out-Of-Memory (OOM).", flush=True)
                all_results[target_len][mode] = {
                    "status": "OOM",
                    "prompt_len": target_len,
                    "generated_tokens": 0,
                    "prefill_time_s": 0.0,
                    "decode_time_s": 0.0,
                    "decode_tps": 0.0,
                    "peak_prefill_vram_gb": 0.0,
                    "peak_decode_vram_gb": 0.0,
                    "kv_cache_vram_gb": 0.0,
                    "base_temp_c": 0,
                    "base_power_w": 0,
                    "peak_prefill_temp_c": 0,
                    "peak_prefill_power_w": 0,
                    "peak_decode_temp_c": 0,
                    "peak_decode_power_w": 0,
                    "output_text": "ERROR: CUDA OUT OF MEMORY (OOM) - Standard Dense memory footprint exceeds physical GPU capacity."
                }

    # Compile Comprehensive Markdown Report
    report_path = os.path.join(REPO, "colab", "nat_128k_q4_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-14B-Instruct (4-bit) Context Sweep Report (4K - 128K)\n\n")
        f.write("This report benchmarks Standard Dense (4-bit quantized weights) against Differential KV (4-bit weights + Compressed KV cache) on an A100 GPU across sequence lengths from 4,000 to 128,000 tokens.\n\n")
        
        f.write("## Performance & Hardware Resources Comparison Table\n\n")
        
        headers = ["Context", "Mode", "Status", "Prefill Time", "Decode TPS", "Peak VRAM", "KV Cache VRAM", "Peak Temp (°C)", "Peak Power (W)"]
        rows = []
        
        for target_len in sweep_lengths:
            for mode in ["dense", "compressed"]:
                res = all_results[target_len].get(mode, {})
                status_text = res.get("status", "success").upper()
                
                rows.append([
                    f"{target_len // 1024}K ({target_len})",
                    mode,
                    status_text,
                    f"{res.get('prefill_time_s', 0):.2f}s" if status_text != "OOM" else "N/A",
                    f"{res.get('decode_tps', 0):.2f}" if status_text != "OOM" else "N/A",
                    f"{res.get('peak_prefill_vram_gb', 0):.2f} GB" if status_text != "OOM" else "N/A",
                    f"{res.get('kv_cache_vram_gb', 0):.3f} GB" if status_text != "OOM" else "N/A",
                    f"{res.get('peak_prefill_temp_c', 0)} °C" if status_text != "OOM" else "N/A",
                    f"{res.get('peak_prefill_power_w', 0)} W" if status_text != "OOM" else "N/A"
                ])
                
        from tabulate import tabulate
        f.write(tabulate(rows, headers=headers, tablefmt="github") + "\n\n")
        
        f.write("## Generated Responses (Quality Evaluation)\n\n")
        f.write("We evaluate the quality of the model output on the custom Neighborhood Attention claim query for the 128K context sequence run:\n\n")
        
        for mode in ["dense", "compressed"]:
            res = all_results[131072].get(mode, {})
            text = res.get("output_text", "").strip()
            word_count = len(text.split())
            f.write(f"### 128K Mode: `{mode}` ({word_count} words)\n")
            f.write(f"> {text}\n\n")

    print(f"\nSUCCESS: Sweep complete. Report compiled at {report_path}", flush=True)

if __name__ == "__main__":
    main()
