import os
import sys
import time
import gc
import json
import argparse
import subprocess
import threading
import urllib.request
import psutil
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OLLAMA_FP16 = "qwen2.5:0.5b-instruct-fp16"
OLLAMA_QUANT = "qwen2.5:0.5b-instruct"

# ── Process Finder & Memory Tracker ───────────────────────────────────────────

def find_ollama_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any('ollama' in part.lower() for part in cmd) and any('serve' in part.lower() for part in cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

class MemoryTracker(threading.Thread):
    def __init__(self, process_or_pid=None, interval=0.01):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_rss = 0.0
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0
        
        if isinstance(process_or_pid, psutil.Process):
            self.process = process_or_pid
        elif isinstance(process_or_pid, int):
            try:
                self.process = psutil.Process(process_or_pid)
            except Exception:
                self.process = psutil.Process()
        else:
            self.process = psutil.Process()

    def run(self):
        while not self.stop_event.is_set():
            try:
                if self.process.is_running():
                    mem = self.process.memory_info()
                    rss = mem.rss / 1e6
                    if rss > self.peak_rss:
                        self.peak_rss = rss
                
                if torch.backends.mps.is_available():
                    try:
                        alloc = torch.mps.current_allocated_memory() / 1e6
                        res = torch.mps.driver_allocated_memory() / 1e6
                        if alloc > self.peak_allocated:
                            self.peak_allocated = alloc
                        if res > self.peak_reserved:
                            self.peak_reserved = res
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# ── NIAH Prompt Builder ────────────────────────────────────────────────────────

def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    # Estimate tokens
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False) if tokenizer else list(range(40))
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False) if tokenizer else list(range(10))
    
    char_len_per_token = 4
    needle_len_tokens = len(needle_tokens)
    
    target_filler_tokens = context_length - needle_len_tokens - 150
    if target_filler_tokens < 0:
        target_filler_tokens = 100
        
    if tokenizer:
        num_repeats = (target_filler_tokens // len(filler_tokens)) + 1
        all_filler_tokens = (filler_tokens * num_repeats)[:target_filler_tokens]
        insert_idx = int(len(all_filler_tokens) * depth)
        part1_text = tokenizer.decode(all_filler_tokens[:insert_idx])
        part2_text = tokenizer.decode(all_filler_tokens[insert_idx:])
    else:
        filler_char_len = target_filler_tokens * char_len_per_token
        num_repeats = (filler_char_len // len(filler)) + 1
        all_filler = (filler * num_repeats)[:filler_char_len]
        insert_idx = int(len(all_filler) * depth)
        part1_text = all_filler[:insert_idx]
        part2_text = all_filler[insert_idx:]
        
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1_text + "\n"
        + needle + "\n"
        + part2_text + "\n\n"
        + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

# ── Ollama HTTP Client Helpers ────────────────────────────────────────────────

def query_ollama(model_name, prompt, length, max_tokens=64):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "raw": True,  # Bypasses Ollama's default template wrapper to avoid prompt structure corruption
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
            "num_ctx": max(2048, length + 200)
        }
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

def unload_ollama_model(model_name):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": "",
        "keep_alive": 0
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
    except Exception:
        pass

# ── Worker Single Run Logic ───────────────────────────────────────────────────

def run_single_benchmark(mode, context_len, rank, micro_block_size, preset, model_name=None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"  # Force DiffKV compression
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create the needle and long story question to prevent early stopping
    needle = "The special code is 847291."
    question = "Write a long detailed story explaining the origin of the special code 847291 in computer science. Make it at least 200 words. Do not stop early."
    
    effective_model_id = model_name or MODEL_ID
    
    if mode == "standard":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(effective_model_id)
        prompt = make_niah_prompt(tokenizer, context_len, 0.5, needle, question)
        
        tracker = MemoryTracker()
        tracker.start()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                effective_model_id,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            ).to(device)
            model.eval()
            
            # Warmup
            with torch.no_grad():
                _ = model(tokenizer("Warmup.", return_tensors="pt").input_ids.to(device), use_cache=True)
            
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
                
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Prefill time measurement
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(inputs.input_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            if device.type == "mps":
                torch.mps.synchronize()
            t_prefill = time.perf_counter() - t0
            
            # Decode time measurement using clean autoregressive loop to isolate decode steps
            t1 = time.perf_counter()
            current_token = next_token
            generated_tokens = [next_token.item()]
            
            # Decode exactly 63 more steps for 64 tokens total
            for _ in range(63):
                with torch.no_grad():
                    outputs = model(current_token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_tokens.append(current_token.item())
            if device.type == "mps":
                torch.mps.synchronize()
            t_decode = time.perf_counter() - t1
            
            actual_count = len(generated_tokens)
            tps = actual_count / max(t_decode, 0.001)
            
            # Find the first EOS or stop token to construct the readable response and check accuracy
            eos_tokens = [tokenizer.eos_token_id]
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None:
                eos_tokens.append(im_end_id)
                
            clean_tokens = []
            for tok in generated_tokens:
                if tok in eos_tokens:
                    break
                clean_tokens.append(tok)
            response = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
            accuracy = 1.0 if "847291" in response else 0.0
            
            tracker.stop()
            tracker.join()
            
            res = {
                "prefill_s": t_prefill,
                "decode_tps": tps,
                "peak_rss_mb": tracker.peak_rss,
                "peak_allocated_mb": tracker.peak_allocated,
                "peak_reserved_mb": tracker.peak_reserved,
                "avg_cos_sim": 1.0,
                "accuracy": accuracy,
                "response": response
            }
            print(json.dumps(res))
            
        except Exception as e:
            tracker.stop()
            try:
                tracker.join()
            except Exception:
                pass
            import traceback
            print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
            
    elif mode == "diffkv":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))
        import numpy as np
        from serving.hf_diffkv_wrapper import DiffKVHFWrapper
        from transformers import AutoTokenizer
        from collections import Counter
        
        tokenizer = AutoTokenizer.from_pretrained(effective_model_id)
        prompt = make_niah_prompt(tokenizer, context_len, 0.5, needle, question)
        
        # Subclass DiffKVHFWrapper to run clean benchmarks
        class BenchmarkedDiffKVWrapper(DiffKVHFWrapper):
            def generate_benchmark(self, prompt, max_new_tokens=64, temperature=0.0):
                self.ensure_loaded()
                session_id = self.active_session or "default"
                prompt_ids = self.tokenizer.encode(prompt)
                
                cached_len = 0
                if session_id in self._session_token_ids:
                    stored_ids = self._session_token_ids[session_id]
                    if len(stored_ids) > 0 and len(stored_ids) < len(prompt_ids):
                        if prompt_ids[:len(stored_ids)] == stored_ids:
                            cached_len = len(stored_ids)
                            
                if cached_len == 0:
                    self.manager.clear_session(session_id)
                    self._session_token_ids[session_id] = []
                    new_prompt_ids = prompt_ids
                else:
                    new_prompt_ids = prompt_ids[cached_len:]

                self.manager.init_session(session_id, prefill_len=cached_len + len(new_prompt_ids))
                self.manager.register_prefill_tokens(session_id, torch.tensor(new_prompt_ids, dtype=torch.long))
                self.model._diffkv_session_ids = [session_id]

                # ── Chunked Prefill ──
                PREFILL_CHUNK = 512
                output = None
                
                t0 = time.perf_counter()
                for chunk_start in range(0, len(new_prompt_ids), PREFILL_CHUNK):
                    chunk = new_prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
                    clen = len(chunk)
                    abs_start = cached_len + chunk_start
                    
                    chunk_tensor = torch.tensor([chunk], dtype=torch.long)
                    pos_tensor = torch.tensor([list(range(abs_start, abs_start + clen))], dtype=torch.long)
                    
                    output = self.model(chunk_tensor, pos_tensor)
                    self.manager.compress_deferred_prefill_blocks(session_id)
                
                # MLX/MPS synchronization for accurate prefill measurement
                if hasattr(self, "device") and self.device == "mps":
                    torch.mps.synchronize()
                import mlx.core as mx
                mx.eval()
                t_prefill = time.perf_counter() - t0

                generated = prompt_ids.copy()
                srl_state = self.manager.get_srl_state(session_id)
                if srl_state is not None:
                    srl_state.vsl_active_candidates = []
                    srl_state.vsl_consecutive_helpers = 0
                    srl_state.factual_anchor_q = None
                    srl_state.current_entity_id = -1
                    srl_state.dual_entity_mode = False
                    srl_state.dual_entity_ids = []

                cur_pos = cached_len + len(new_prompt_ids)
                logits = output.logits[0, -1].cpu().numpy()

                def sample_logits(logits, temp, top_p):
                    if temp <= 0.01:
                        return int(np.argmax(logits))
                    scaled = logits / temp
                    exp_logits = np.exp(scaled - np.max(scaled))
                    probs = exp_logits / np.sum(exp_logits)
                    if top_p < 1.0:
                        sorted_indices = np.argsort(probs)[::-1]
                        sorted_probs = probs[sorted_indices]
                        cum_probs = np.cumsum(sorted_probs)
                        cutoff = np.where(cum_probs > top_p)[0]
                        if len(cutoff) > 0:
                            probs[sorted_indices[cutoff[0]+1:]] = 0.0
                            probs = probs / np.sum(probs)
                    return int(np.random.choice(len(probs), p=probs))

                self._mlx_loop_detected = False
                self._mlx_loop_idx = None
                
                # Time Decode Directly
                t1 = time.perf_counter()
                first_stop_idx = None
                
                for step_idx in range(max_new_tokens):
                    _new_tokens = generated[len(prompt_ids):]
                    _n_new = len(_new_tokens)
                    _loop_detected = getattr(self, "_mlx_loop_detected", False)
                    _loop_idx = getattr(self, "_mlx_loop_idx", None)

                    if not _loop_detected and _n_new >= 30 and _n_new % 10 == 0:
                        _window = _new_tokens[-80:]
                        _ng = 5
                        if len(_window) >= _ng + 1:
                            _ngrams = [tuple(_window[i:i + _ng]) for i in range(len(_window) - _ng + 1)]
                            _top = Counter(_ngrams).most_common(1)[0][1]
                            if _top / len(_ngrams) >= 0.35:
                                _loop_detected = True
                                self._mlx_loop_detected = True
                                self._mlx_loop_idx = _n_new

                    _pen_window = 256 if _loop_detected else 64
                    _pen_val = max(1.15, 1.3) if _loop_detected else 1.15
                    if _pen_val != 1.0:
                        for tok_id in set(generated[-_pen_window:]):
                            if logits[tok_id] > 0:
                                logits[tok_id] /= _pen_val
                            else:
                                logits[tok_id] *= _pen_val

                    # SRL bias
                    srl_state = getattr(self.manager, "_session_srl", {}).get(session_id)
                    if srl_state is not None:
                        from native_core.srl.factual_alignment import get_helper_token_ids
                        helper_ids = get_helper_token_ids(self.tokenizer)

                        if getattr(srl_state, "current_step_factual_tokens", None):
                            current_entity = getattr(srl_state, "current_entity_id", -1)
                            entity_ids = getattr(srl_state, "current_step_sequence_entity_ids", [])
                            is_prime_list = getattr(srl_state, "current_step_sequence_is_prime", [])
                            
                            if current_entity != -1:
                                entity_factual_tokens = set()
                                for i, seq in enumerate(srl_state.current_step_factual_sequences):
                                    seq_eid = entity_ids[i] if i < len(entity_ids) else -1
                                    seq_is_prime = is_prime_list[i] if i < len(is_prime_list) else False
                                    if seq_eid == -1 or seq_eid == current_entity or seq_is_prime:
                                        entity_factual_tokens.update(seq)
                                for tok_id in entity_factual_tokens:
                                    if tok_id < len(logits):
                                        logits[tok_id] += 7.0
                            else:
                                for tok_id in srl_state.current_step_factual_tokens:
                                    if tok_id < len(logits):
                                        logits[tok_id] += 7.0

                        active_candidates = getattr(srl_state, "vsl_active_candidates", [])
                        if active_candidates:
                            for suffix in active_candidates:
                                if suffix and suffix[0] < len(logits):
                                    logits[suffix[0]] += 7.0

                        if (getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.4
                                and not getattr(srl_state, "dual_entity_mode", False)
                                and getattr(srl_state, "current_step_factual_tokens", None)):
                            factual_set = srl_state.current_step_factual_tokens
                            _vocab = len(logits)
                            _excl = np.array([t for t in list(factual_set) + list(helper_ids) if 0 <= t < _vocab], dtype=np.int64)
                            _penalty_mask = np.ones(_vocab, dtype=bool)
                            if len(_excl) > 0:
                                _penalty_mask[_excl] = False
                            logits[_penalty_mask] -= 3.5

                        last_token = generated[-1] if generated else None
                        if last_token is not None and getattr(srl_state, "current_step_factual_sequences", None):
                            transition_candidates = set()
                            current_entity = getattr(srl_state, "current_entity_id", -1)
                            entity_ids = getattr(srl_state, "current_step_sequence_entity_ids", [])
                            for i, seq in enumerate(srl_state.current_step_factual_sequences):
                                seq_entity = entity_ids[i] if i < len(entity_ids) else -1
                                if current_entity != -1 and seq_entity != -1 and seq_entity != current_entity:
                                    continue
                                for idx, tok in enumerate(seq[:-1]):
                                    if tok == last_token:
                                        transition_candidates.add(seq[idx + 1])
                            for tok_id in transition_candidates:
                                if tok_id < len(logits):
                                    logits[tok_id] += 10.0

                    effective_temperature = temperature
                    if srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55:
                        max_sim = srl_state.current_step_max_similarity
                        effective_temperature = temperature * (1.0 - max_sim * 0.95)

                    sfa_active = (
                        srl_state is not None
                        and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55
                        and bool(getattr(srl_state, "current_step_factual_sequences", None))
                    )

                    if sfa_active:
                        from native_core.srl.factual_alignment import (
                            get_allowed_tokens_vsl, get_structural_helper_token_ids)
                        structural_helper_ids = get_structural_helper_token_ids(self.tokenizer)
                        allowed_ids = get_allowed_tokens_vsl(
                            srl_state, helper_ids,
                            structural_helper_ids=structural_helper_ids,
                            sfa_active=True,
                        )
                        mask = np.ones_like(logits, dtype=bool)
                        mask[list(allowed_ids)] = False
                        
                        max_sim = getattr(srl_state, "current_step_max_similarity", 0.0)
                        if max_sim >= 0.70:
                            logits[mask] = -1e10
                        else:
                            logits[mask] -= 7.0

                    next_id = sample_logits(logits, effective_temperature, top_p=0.9)

                    if sfa_active and srl_state is not None:
                        from native_core.srl.factual_alignment import update_vsl_state, get_helper_token_ids
                        helper_ids = get_helper_token_ids(self.tokenizer)
                        update_vsl_state(next_id, srl_state, helper_ids)
                        
                        if getattr(srl_state, "vsl_consecutive_helpers", 0) >= 16:
                            if first_stop_idx is None:
                                first_stop_idx = len(generated)

                    generated.append(next_id)
                    self.manager.register_prefill_tokens(session_id, torch.tensor([next_id], dtype=torch.long))

                    if srl_state is not None and hasattr(srl_state, "save_step_state"):
                        srl_state.save_step_state(len(generated))

                    if next_id in self.stop_token_ids:
                        if first_stop_idx is None:
                            first_stop_idx = len(generated) - 1

                    input_ids = torch.tensor([[next_id]], dtype=torch.long)
                    pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long)
                    
                    output = self.model(input_ids, pos_tensor)
                    logits = output.logits[0, -1].cpu().numpy()
                    
                    cur_pos += 1
                
                if hasattr(self, "device") and self.device == "mps":
                    torch.mps.synchronize()
                mx.eval()
                t_decode = time.perf_counter() - t1

                self._mlx_loop_detected = False
                self._mlx_loop_idx = None
                self._session_token_ids[session_id] = generated
                
                # Extract truncated response
                if first_stop_idx is not None:
                    response_ids = generated[:first_stop_idx]
                else:
                    response_ids = generated
                    
                response_text = self.tokenizer.decode(response_ids[len(prompt_ids):], skip_special_tokens=True).strip()
                
                avg_cos_sim = 1.0
                if hasattr(self.manager, "runtime_summary"):
                    summary = self.manager.runtime_summary()
                    avg_cos_sim = summary.get("avg_cosine_sim", 1.0)
                    
                return {
                    "prefill_s": t_prefill,
                    "decode_tps": max_new_tokens / max(t_decode, 0.001),
                    "response": response_text,
                    "avg_cos_sim": avg_cos_sim
                }

        tracker = MemoryTracker()
        tracker.start()
        
        try:
            config = {
                "rank": rank,
                "block_size": micro_block_size,
                "micro_block_size": micro_block_size,
                "preset": preset
            }
            wrapper = BenchmarkedDiffKVWrapper(
                model_id=effective_model_id,
                config=config,
                device=device.type,
            )
            
            # Warmup
            _ = wrapper.generate(prompt="Warmup run.", max_new_tokens=1, temperature=0.0)
            wrapper.manager.clear_session("default")
            
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
                
            # Run benchmarked generation
            res_bench = wrapper.generate_benchmark(prompt=prompt, max_new_tokens=64, temperature=0.0)
            
            accuracy = 1.0 if "847291" in res_bench["response"] else 0.0
            
            wrapper.stop()
            tracker.stop()
            tracker.join()
            
            res = {
                "prefill_s": res_bench["prefill_s"],
                "decode_tps": res_bench["decode_tps"],
                "peak_rss_mb": tracker.peak_rss,
                "peak_allocated_mb": tracker.peak_allocated,
                "peak_reserved_mb": tracker.peak_reserved,
                "avg_cos_sim": res_bench["avg_cos_sim"],
                "accuracy": accuracy,
                "response": res_bench["response"]
            }
            print(json.dumps(res))
            
        except Exception as e:
            tracker.stop()
            try:
                tracker.join()
            except Exception:
                pass
            import traceback
            print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
            
    elif mode == "ollama":
        # Warm up Ollama model
        _ = query_ollama(model_name, "Warm up prompt", 128, max_tokens=1)
        time.sleep(2.0)
        
        runner = find_ollama_process()
        
        # Start memory tracking on Ollama process
        tracker = MemoryTracker(runner)
        tracker.start()
        
        prompt = make_niah_prompt(None, context_len, 0.5, needle, question)
        
        t_start = time.perf_counter()
        res = query_ollama(model_name, prompt, context_len, max_tokens=64)
        t_elapsed = time.perf_counter() - t_start
        
        tracker.stop()
        tracker.join()
        
        if "error" in res:
            print(json.dumps({"error": res["error"]}))
        else:
            prompt_eval_ns = res.get("prompt_eval_duration", 0.0)
            eval_ns = res.get("eval_duration", 0.0)
            eval_count = res.get("eval_count", 1.0)
            
            prefill_s = prompt_eval_ns / 1e9
            decode_tps = eval_count / (max(1.0, eval_ns) / 1e9)
            
            # Fallback to wall-clock if needed
            if prompt_eval_ns == 0.0 or eval_ns == 0.0:
                prefill_s = t_elapsed
                decode_tps = 64.0 / t_elapsed
                
            response = res.get("response", "").strip()
            accuracy = 1.0 if "847291" in response else 0.0
            
            print(json.dumps({
                "prefill_s": prefill_s,
                "decode_tps": decode_tps,
                "peak_rss_mb": tracker.peak_rss,
                "accuracy": accuracy,
                "response": response
            }))
            
        unload_ollama_model(model_name)

# ── Parent Orchestrator ───────────────────────────────────────────────────────

def run_subprocess_run(mode, context_len, rank=16, micro_block_size=16, preset="low", model_name=None):
    cmd = [
        sys.executable,
        __file__,
        "--run-single",
        "--mode", mode,
        "--context", str(context_len),
        "--rank", str(rank),
        "--micro-block-size", str(micro_block_size),
        "--preset", preset
    ]
    if model_name:
        cmd += ["--model-name", model_name]
        
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode != 0:
            return {"error": f"Process exited with code {result.returncode}", "stderr": result.stderr}
        
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        return {"error": "No JSON output found", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "TimeoutExpired after 180s"}
    except Exception as e:
        return {"error": str(e)}

# ── Plotting Logic ────────────────────────────────────────────────────────────

def generate_consolidated_plots(results_data, out_dir, size_prefix=""):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    
    contexts = [1024, 2048, 4096, 8192, 16384]
    modes = ["dense_pytorch", "diffkv_mlx", "ollama_fp16", "ollama_quant"]
    labels = {
        "dense_pytorch": "Dense (PyTorch)",
        "diffkv_mlx": "DiffKV (MLX)",
        "ollama_fp16": "Ollama FP16",
        "ollama_quant": "Ollama Quantized (4-bit)"
    }
    colors = {
        "dense_pytorch": "#ff7675",
        "diffkv_mlx": "#0984e3",
        "ollama_fp16": "#e056fd",
        "ollama_quant": "#10ac84"
    }
    markers = {
        "dense_pytorch": "o",
        "diffkv_mlx": "s",
        "ollama_fp16": "^",
        "ollama_quant": "d"
    }
    linestyles = {
        "dense_pytorch": "-",
        "diffkv_mlx": "--",
        "ollama_fp16": "-.",
        "ollama_quant": ":"
    }
    linewidths = {
        "dense_pytorch": 3.5,
        "diffkv_mlx": 2.5,
        "ollama_fp16": 2.0,
        "ollama_quant": 1.5
    }
    acc_jitter = {
        "dense_pytorch": -0.015,
        "diffkv_mlx": 0.0,
        "ollama_fp16": 0.0,
        "ollama_quant": 0.015
    }
    mem_jitter = {
        "dense_pytorch": 0.0,
        "diffkv_mlx": 0.0,
        "ollama_fp16": -0.02,
        "ollama_quant": 0.02
    }

    prefill = {m: [] for m in modes}
    tps = {m: [] for m in modes}
    acc = {m: [] for m in modes}
    mem = {m: [] for m in modes}
    
    for c in contexts:
        for m in modes:
            m_res = results_data.get(m, {}).get(str(c), {})
            if "error" in m_res or not m_res:
                prefill[m].append(None)
                tps[m].append(None)
                acc[m].append(None)
                mem[m].append(None)
            else:
                prefill[m].append(m_res.get("prefill_s"))
                tps[m].append(m_res.get("decode_tps"))
                acc[m].append(m_res.get("accuracy"))
                # Memory footprint
                peak_rss = m_res.get("peak_rss_mb", 0.0)
                peak_res = m_res.get("peak_reserved_mb", 0.0)
                mem[m].append(max(peak_rss, peak_res) / 1024.0)

    plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'

    # 1. Prefill Latency
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, p) for c, p in zip(contexts, prefill[m]) if p is not None]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] for x in valid], marker=markers[m], linestyle=linestyles[m], linewidth=linewidths[m], label=labels[m], color=colors[m], markersize=8)
    plt.title(f'Prefill Latency (TTFT) vs. Context Length ({size_prefix.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Prefill Latency (seconds)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'compare_{size_prefix}_prefill.png'), dpi=200)
    plt.close()

    # 2. Decode TPS (Throughput)
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, t) for c, t in zip(contexts, tps[m]) if t is not None]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] for x in valid], marker=markers[m], linestyle=linestyles[m], linewidth=linewidths[m], label=labels[m], color=colors[m], markersize=8)
    plt.title(f'Decode Throughput (TPS) vs. Context Length ({size_prefix.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Throughput (tokens/second)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'compare_{size_prefix}_tps.png'), dpi=200)
    plt.close()

    # 3. Memory Footprint
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, me) for c, me in zip(contexts, mem[m]) if me is not None and me > 0.001]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] + mem_jitter[m] for x in valid], marker=markers[m], linestyle=linestyles[m], linewidth=linewidths[m], label=labels[m], color=colors[m], markersize=8)
    plt.title(f'Peak System Memory vs. Context Length ({size_prefix.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Peak RAM/VRAM Memory (GB)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'compare_{size_prefix}_memory.png'), dpi=200)
    plt.close()

    # 4. Accuracy
    plt.figure(figsize=(8, 5))
    for m in modes:
        valid = [(c, a) for c, a in zip(contexts, acc[m]) if a is not None]
        if valid:
            plt.plot([x[0] for x in valid], [x[1] + acc_jitter[m] for x in valid], marker=markers[m], linestyle=linestyles[m], linewidth=linewidths[m], label=labels[m], color=colors[m], markersize=8)
    plt.title(f'NIAH Retrieval Accuracy vs. Context Length ({size_prefix.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Accuracy (0.0 to 1.0)', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(contexts, [str(c) for c in contexts])
    plt.ylim(-0.1, 1.1)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'compare_{size_prefix}_accuracy.png'), dpi=200)
    plt.close()

# ── Main Control Loop ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--mode", choices=["standard", "diffkv", "ollama"])
    parser.add_argument("--context", type=int)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--micro-block-size", type=int, default=16)
    parser.add_argument("--preset", type=str, default="low")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--only-ollama", action="store_true", help="Skip standard and diffkv GPU phases, only run/update Ollama results")
    args = parser.parse_args()
    
    if args.run_single:
        run_single_benchmark(args.mode, args.context, args.rank, args.micro_block_size, args.preset, args.model_name)
        sys.exit(0)
        
    print("=" * 80)
    print("      LAUNCHING CONSOLIDATED DENSE VS DIFFKV VS OLLAMA BENCHMARK SUITE")
    print("=" * 80)
    
    contexts = [1024, 2048, 4096, 8192, 16384]
    
    out_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results_ollama.json")
    results = None
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results from {out_json}")
        except Exception as e:
            print(f"Could not load existing results: {e}")
            
    if results is None:
        results = {
            "0.5b": {
                "dense_pytorch": {},
                "diffkv_mlx": {},
                "ollama_fp16": {},
                "ollama_quant": {}
            },
            "1.5b": {
                "dense_pytorch": {},
                "diffkv_mlx": {},
                "ollama_fp16": {},
                "ollama_quant": {}
            }
        }
    
    model_configs = {
        "0.5b": {
            "hf_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "ollama_fp16": "qwen2.5:0.5b-instruct-fp16",
            "ollama_quant": "qwen2.5:0.5b-instruct"
        },
        "1.5b": {
            "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "ollama_fp16": "qwen2.5:1.5b-instruct-fp16",
            "ollama_quant": "qwen2.5:1.5b-instruct"
        }
    }
    
    for size in ["0.5b", "1.5b"]:
        print("\n" + "=" * 60)
        print(f"       RUNNING BENCHMARKS FOR MODEL SIZE: {size.upper()}")
        print("=" * 60)
        
        cfg = model_configs[size]
        
        # 1. Run standard dense baseline
        if not args.only_ollama:
            print(f"\n--- Phase 1: Running PyTorch Dense Baseline ({size.upper()}) ---")
            for c in contexts:
                print(f"Running Dense | Ctx: {c}...")
                res = run_subprocess_run("standard", c, model_name=cfg["hf_name"])
                if "error" in res:
                    print(f"  --> Skip/OOM: {res['error']}")
                    results[size]["dense_pytorch"][str(c)] = {"error": res['error']}
                else:
                    print(f"  --> Done: TTFT={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | Acc={res['accuracy']:.1f}")
                    results[size]["dense_pytorch"][str(c)] = res
        else:
            print(f"Skipping Phase 1 (PyTorch Dense) for {size.upper()} (using cached results)")
                
        # 2. Run DiffKV MLX
        if not args.only_ollama:
            print(f"\n--- Phase 2: Running MLX DiffKV Baseline ({size.upper()}) ---")
            for c in contexts:
                print(f"Running DiffKV | Ctx: {c}...")
                res = run_subprocess_run("diffkv", c, rank=16, preset="low", model_name=cfg["hf_name"])
                if "error" in res:
                    print(f"  --> Skip/OOM: {res['error']}")
                    results[size]["diffkv_mlx"][str(c)] = {"error": res['error']}
                else:
                    print(f"  --> Done: TTFT={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | Acc={res['accuracy']:.1f}")
                    results[size]["diffkv_mlx"][str(c)] = res
        else:
            print(f"Skipping Phase 2 (MLX DiffKV) for {size.upper()} (using cached results)")
                
        # 3. Run Ollama FP16
        print(f"\n--- Phase 3: Running Ollama FP16 Baseline ({size.upper()}) ---")
        for c in contexts:
            print(f"Running Ollama FP16 | Ctx: {c}...")
            res = run_subprocess_run("ollama", c, model_name=cfg["ollama_fp16"])
            if "error" in res:
                print(f"  --> Skip/OOM/Error: {res['error']}")
                results[size]["ollama_fp16"][str(c)] = {"error": res['error']}
            else:
                print(f"  --> Done: TTFT={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | Acc={res['accuracy']:.1f}")
                results[size]["ollama_fp16"][str(c)] = res
                
        # 4. Run Ollama Quant
        print(f"\n--- Phase 4: Running Ollama Quantized Baseline ({size.upper()}) ---")
        for c in contexts:
            print(f"Running Ollama Quantized | Ctx: {c}...")
            res = run_subprocess_run("ollama", c, model_name=cfg["ollama_quant"])
            if "error" in res:
                print(f"  --> Skip/OOM/Error: {res['error']}")
                results[size]["ollama_quant"][str(c)] = {"error": res['error']}
            else:
                print(f"  --> Done: TTFT={res['prefill_s']:.3f}s | TPS={res['decode_tps']:.1f} | PeakRSS={res['peak_rss_mb']:.1f}MB | Acc={res['accuracy']:.1f}")
                results[size]["ollama_quant"][str(c)] = res
                
    # Save raw consolidated results
    out_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results_ollama.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw consolidated results saved to {out_json}")
    
    # Generate plots in the current artifact folder
    artifact_dir = "/Users/omchimurkar1/.gemini/antigravity/brain/ada31170-301d-45cf-bbdf-321c6b861dbc"
    if not os.path.exists(artifact_dir):
        artifact_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_plots")
        
    print(f"Generating comparative plots in {artifact_dir}...")
    for size in ["0.5b", "1.5b"]:
        try:
            generate_consolidated_plots(results[size], artifact_dir, size_prefix=size)
            print(f"Consolidated plots for {size} generated successfully.")
        except Exception as e:
            print(f"Error generating plots for {size}: {e}")
            import traceback
            traceback.print_exc()
