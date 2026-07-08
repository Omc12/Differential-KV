#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
import json
import argparse
import shutil
import psutil

# Define absolute paths relative to repo root
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Set custom cache directories under the repo to isolate all downloads and JIT compiles.
# This makes it easy to completely delete all files on cleanup without affecting the global user home.
CACHE_DIR = os.path.join(REPO_ROOT, "remote_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["TRITON_CACHE_DIR"] = os.path.join(CACHE_DIR, "triton")
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = os.path.join(CACHE_DIR, "torch_kernel")

# ───────────────────────────── Compilation Helpers ─────────────────────────────

def compile_active():
    """Compile the CPython Active CUDA extension using setup.py."""
    print("\n🔨 Compiling DiffKV Active (CPython CUDA Extension)...")
    ext_dir = os.path.join(REPO_ROOT, "ACTIVE_RUNTIME", "native_core", "diffkv_core")
    
    # Run setup.py build_ext --inplace
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    res = subprocess.run(cmd, cwd=ext_dir, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Active compilation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        return False
    print("✅ Active compilation succeeded.")
    return True

def compile_native():
    """Compile the Native C++ executable using CMake with CUDA enabled."""
    print("\n🔨 Compiling DiffKV Native C++ (with CUDA)...")
    native_dir = os.path.join(REPO_ROOT, "diffkv_native")
    build_dir = os.path.join(native_dir, "build")
    
    # Remove old build directory if exists to ensure clean compile
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    # Configure CMake with GGML_CUDA=ON
    configure_cmd = ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON"]
    res_configure = subprocess.run(configure_cmd, cwd=native_dir, capture_output=True, text=True)
    if res_configure.returncode != 0:
        print(f"❌ CMake configuration failed:\nSTDOUT:\n{res_configure.stdout}\nSTDERR:\n{res_configure.stderr}")
        return False
        
    # Build
    build_cmd = ["cmake", "--build", "build", "-j"]
    res_build = subprocess.run(build_cmd, cwd=native_dir, capture_output=True, text=True)
    if res_build.returncode != 0:
        print(f"❌ CMake build failed:\nSTDOUT:\n{res_build.stdout}\nSTDERR:\n{res_build.stderr}")
        return False
        
    print("✅ Native compilation succeeded.")
    return True

# ────────────────────────────── Telemetry Samplers ──────────────────────────────

def get_native_vram(pid):
    """Query nvidia-smi for the specific VRAM footprint of the given PID (in GB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8")
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) == 2 and int(parts[0].strip()) == pid:
                return float(parts[1].strip()) / 1024.0  # MiB -> GB
    except Exception:
        pass
    return 0.0

class VRAMSampler(threading.Thread):
    """Polls nvidia-smi at 20 Hz to measure peak VRAM of a C++ subprocess."""
    def __init__(self, pid, poll_interval=0.05):
        super().__init__(daemon=True)
        self.pid = pid
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.peak_vram_gb = 0.0

    def run(self):
        while not self.stop_event.is_set():
            vram = get_native_vram(self.pid)
            if vram > self.peak_vram_gb:
                self.peak_vram_gb = vram
            time.sleep(self.poll_interval)

    def stop(self):
        self.stop_event.set()

class SystemRAMSampler(threading.Thread):
    """Polls process tree RSS memory at 20 Hz."""
    def __init__(self, pid, poll_interval=0.05):
        super().__init__(daemon=True)
        self.pid = pid
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.peak_ram_gb = 0.0

    def run(self):
        try:
            main_proc = psutil.Process(self.pid)
        except Exception:
            return

        while not self.stop_event.is_set():
            try:
                procs = [main_proc] + main_proc.children(recursive=True)
                rss_sum = sum(p.memory_info().rss for p in procs)
                ram_gb = rss_sum / 1e9
                if ram_gb > self.peak_ram_gb:
                    self.peak_ram_gb = ram_gb
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def stop(self):
        self.stop_event.set()

# ────────────────────────────── Download Helper ──────────────────────────────

def download_models(model_id, gguf_repo, gguf_filename):
    """Download HF model and/or GGUF file to isolated cache directory."""
    print(f"\n📥 Downloading weights to isolated cache: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Download HuggingFace weights (for Active/Dense)
    if model_id:
        print(f"👉 Downloading HF model {model_id}...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # This will download using the HF_HOME env var configured above
        AutoTokenizer.from_pretrained(model_id)
        print("✓ Tokenizer downloaded.")
        
    # Download GGUF file (for Native)
    gguf_path = None
    if gguf_repo and gguf_filename:
        print(f"👉 Downloading GGUF {gguf_filename} from {gguf_repo}...")
        from huggingface_hub import hf_hub_download
        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_filename, local_dir=CACHE_DIR)
        print(f"✓ GGUF downloaded to {gguf_path}.")
        
    return gguf_path

# ──────────────────────────── Benchmarking Runners ────────────────────────────

def run_active_benchmark(model_id, prompt_text, gen_len):
    """Run DiffKV Active (PyTorch/Triton) benchmark and record metrics."""
    print("\n🏃 Running DiffKV Active Telemetry Sweep...")
    # Add Active Runtime directory to Python path
    active_runtime_dir = os.path.join(REPO_ROOT, "ACTIVE_RUNTIME")
    sys.path.insert(0, active_runtime_dir)
    
    import torch
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    # Measure memory baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_rss = psutil.Process().memory_info().rss / 1e9
    
    # Load wrapper (forces compilation of Triton/CUDA if needed)
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DiffKVHFWrapper(model_id=model_id, config=cfg, device="cuda")
    wrapper.ensure_loaded()
    
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model
    ids = tok.encode(prompt_text)
    prompt_tokens = len(ids)
    
    # Warmup
    wsid = "warmup"
    mgr.clear_session(wsid)
    wrapper._session_token_ids[wsid] = []
    mgr.init_session(wsid, prefill_len=1)
    mgr.register_prefill_tokens(wsid, torch.tensor([ids[0]], dtype=torch.long, device="cuda"))
    model._diffkv_session_ids = [wsid]
    with torch.no_grad():
        _w = model(torch.tensor([[ids[0]]], dtype=torch.long, device="cuda"),
                   torch.tensor([[0]], dtype=torch.long, device="cuda"))
        _ = _w.logits[0, -1].cpu().numpy()
    mgr.clear_session(wsid)
    
    # Clear memory tracking for actual run
    torch.cuda.reset_peak_memory_stats()
    
    sid = "bench"
    mgr.clear_session(sid)
    wrapper._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device="cuda"))
    model._diffkv_session_ids = [sid]
    
    # Prefill timing
    CH = 512
    t0 = time.perf_counter()
    output = None
    with torch.no_grad():
        for cs in range(0, len(ids), CH):
            chunk = ids[cs:cs + CH]
            ct = torch.tensor([chunk], dtype=torch.long, device="cuda")
            pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long, device="cuda")
            output = model(ct, pt)
            mgr.compress_deferred_prefill_blocks(sid)
        logits = output.logits[0, -1].cpu().numpy()  # force materialization
    prefill_time = time.perf_counter() - t0
    
    # Decode timing
    cur_pos = len(ids)
    generated = []
    ttft_time = None
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(gen_len):
            nid = int(logits.argmax())
            generated.append(nid)
            
            if i == 0:
                ttft_time = time.perf_counter() - t0 + prefill_time
                
            mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long, device="cuda"))
            it = torch.tensor([[nid]], dtype=torch.long, device="cuda")
            pp = torch.tensor([[cur_pos]], dtype=torch.long, device="cuda")
            output = model(it, pp)
            logits = output.logits[0, -1].cpu().numpy()
            cur_pos += 1
            
    decode_time = time.perf_counter() - t0
    decode_tps = len(generated) / decode_time if decode_time > 0 else 0.0
    
    peak_vram = torch.cuda.max_memory_allocated("cuda") / 1e9
    peak_ram = psutil.Process().memory_info().rss / 1e9
    
    # Release reference
    del model, wrapper, mgr
    torch.cuda.empty_cache()
    
    return {
        "engine": "active",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": len(generated),
        "prefill_s": prefill_time,
        "ttft_s": ttft_time,
        "decode_tps": decode_tps,
        "peak_vram_gb": peak_vram,
        "peak_ram_gb": peak_ram
    }

def run_dense_benchmark(model_id, prompt_text, gen_len):
    """Run standard Hugging Face AutoModelForCausalLM (Dense full-KV) baseline."""
    print("\n🏃 Running Dense Baseline Telemetry Sweep...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    
    ids = tokenizer.encode(prompt_text)
    prompt_tokens = len(ids)
    
    # Warmup
    with torch.no_grad():
        _w = model(torch.tensor([ids[:4]], device="cuda"))
        _ = _w.logits[0, -1].cpu().numpy()
        
    torch.cuda.reset_peak_memory_stats()
    
    # Prefill timing
    t0 = time.perf_counter()
    past_key_values = None
    CH = 512
    with torch.no_grad():
        for cs in range(0, len(ids), CH):
            chunk = ids[cs:cs + CH]
            inputs = torch.tensor([chunk], device="cuda")
            outputs = model(inputs, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1].cpu().numpy()
    prefill_time = time.perf_counter() - t0
    
    # Decode timing
    cur_pos = len(ids)
    generated = []
    ttft_time = None
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(gen_len):
            nid = int(logits.argmax())
            generated.append(nid)
            
            if i == 0:
                ttft_time = time.perf_counter() - t0 + prefill_time
                
            inputs = torch.tensor([[nid]], device="cuda")
            outputs = model(inputs, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1].cpu().numpy()
            
    decode_time = time.perf_counter() - t0
    decode_tps = len(generated) / decode_time if decode_time > 0 else 0.0
    
    peak_vram = torch.cuda.max_memory_allocated("cuda") / 1e9
    peak_ram = psutil.Process().memory_info().rss / 1e9
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return {
        "engine": "dense",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": len(generated),
        "prefill_s": prefill_time,
        "ttft_s": ttft_time,
        "decode_tps": decode_tps,
        "peak_vram_gb": peak_vram,
        "peak_ram_gb": peak_ram
    }

def run_native_benchmark(gguf_path, prompt_text, gen_len, ctx_len):
    """Run C++ Native binary over stdin/stdout and capture telemetry."""
    print("\n🏃 Running DiffKV Native Telemetry Sweep...")
    import re
    binary_path = os.path.join(REPO_ROOT, "diffkv_native", "build", "diffkv_native")
    
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Native binary not found at {binary_path}. Did you compile it?")
        
    env = os.environ.copy()
    env.update({
        "DIFFKV_MAX_CTX_TK": str(ctx_len + gen_len + 512),
        "DIFFKV_MICRO_BLOCK_SIZE": "256",
        "DIFFKV_PREFILL_CHUNK_SIZE": "512",
        "DIFFKV_MAX_TOKENS": str(gen_len),
        "DIFFKV_USE_GPU": "1",
        "DIFFKV_NATIVE_ATTN": "1",
        "DIFFKV_TEMPERATURE": "0.0",
        "DIFFKV_TOP_P": "1.0",
        "DIFFKV_REPETITION_PENALTY": "1.0",
        "DIFFKV_DBG_PREFILL_TIME": "1",
        "DIFFKV_TIME_DECODE": "1",
        "DIFFKV_COMPRESSOR_THREADS": "4",
        "DIFFKV_ENABLE_FACTUAL": "1",
    })
    
    # Launch subprocess
    proc = subprocess.Popen(
        [binary_path, gguf_path, "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, env=env
    )
    
    # Start VRAM & RAM Samplers
    vram_sampler = VRAMSampler(proc.pid)
    ram_sampler = SystemRAMSampler(proc.pid)
    vram_sampler.start()
    ram_sampler.start()
    
    stderr_chunks = []
    def _read_err():
        try:
            for line in iter(proc.stderr.readline, b""):
                stderr_chunks.append(line.decode("utf-8", "replace"))
        except Exception:
            pass
            
    et = threading.Thread(target=_read_err, daemon=True)
    et.start()
    
    READY, RESP, FIN = b"__READY__", b"__RESPONSE__", b"__FINISH__"
    
    def _read_until(token):
        buf = b""
        while token not in buf:
            c = os.read(proc.stdout.fileno(), 65536)
            if not c:
                raise RuntimeError("Native C++ binary crashed or exited prematurely.")
            buf += c
        return buf
        
    # Wait for binary to load weights and print __READY__
    _read_until(READY)
    
    # Format and send prompt
    clean_prompt = prompt_text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
    t_send = time.perf_counter()
    proc.stdin.write((clean_prompt + "\n").encode("utf-8"))
    proc.stdin.flush()
    
    # Wait for prefill to end and first token to generate
    buf = _read_until(RESP)
    after = buf.split(RESP, 1)[1]
    t_resp = time.perf_counter()
    
    # Gather tokens until finished
    resp = after
    t_first = t_resp if resp.strip() else None
    while FIN not in resp:
        c = os.read(proc.stdout.fileno(), 65536)
        if not c:
            break
        if t_first is None and c.strip():
            t_first = time.perf_counter()
        resp += c
    t_fin = time.perf_counter()
    
    text = resp.split(FIN, 1)[0].decode("utf-8", "replace").strip()
    
    # Signal shutdown
    try:
        proc.stdin.write(b"exit\n")
        proc.stdin.flush()
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
        
    # Stop samplers
    vram_sampler.stop()
    ram_sampler.stop()
    vram_sampler.join()
    ram_sampler.join()
    
    err = "".join(stderr_chunks)
    
    # Parse prefill timing and token counts from logs
    m = re.search(r"\[PREFILL_TIME\]\s+L=(\d+).*?TOTAL=([0-9.]+)s", err, re.S)
    prompt_tokens = int(m.group(1)) if m else len(prompt_text) // 4
    prefill_s = float(m.group(2)) if m else (t_resp - t_send)
    
    steps = re.findall(r"\[Timing Step (\d+)\]", err)
    gen_tokens = (max(int(s) for s in steps) + 1) if steps else gen_len
    
    decode_s = t_fin - (t_first if t_first else t_resp)
    decode_tps = gen_tokens / decode_s if decode_s > 0 else 0.0
    ttft_s = (t_first - t_send) if t_first else prefill_s
    
    return {
        "engine": "native",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prefill_s": prefill_s,
        "ttft_s": ttft_s,
        "decode_tps": decode_tps,
        "peak_vram_gb": vram_sampler.peak_vram_gb,
        "peak_ram_gb": ram_sampler.peak_ram_gb
    }

# ────────────────────────────── Cleanup Sequence ──────────────────────────────

def perform_cleanup():
    """Wipe cache folders, compilation outputs, and JIT directories completely."""
    print("\n🧹 Initiating Secure Destruct & Cleanup...")
    
    # 1. Remove isolated download cache and JIT directories
    if os.path.exists(CACHE_DIR):
        print(f"👉 Deleting download cache and JIT outputs: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        
    # 2. Clean Active runtime build artifacts
    active_core_dir = os.path.join(REPO_ROOT, "ACTIVE_RUNTIME", "native_core", "diffkv_core")
    active_build = os.path.join(active_core_dir, "build")
    active_egg = os.path.join(active_core_dir, "diffkv_core.egg-info")
    
    for folder in (active_build, active_egg):
        if os.path.exists(folder):
            print(f"👉 Deleting folder: {folder}")
            shutil.rmtree(folder, ignore_errors=True)
            
    # Remove compiled CPython .so shared libraries
    for file in os.listdir(active_core_dir):
        if file.endswith(".so") or file.endswith(".dylib") or file.endswith(".pyd"):
            path = os.path.join(active_core_dir, file)
            print(f"👉 Deleting compiled extension: {path}")
            os.remove(path)
            
    # 3. Clean Native C++ build artifacts
    native_build = os.path.join(REPO_ROOT, "diffkv_native", "build")
    if os.path.exists(native_build):
        print(f"👉 Deleting native build folder: {native_build}")
        shutil.rmtree(native_build, ignore_errors=True)
        
    # 4. Remove temporary summary outputs
    for file in os.listdir(REPO_ROOT):
        if file.startswith("run_native_") and file.endswith(".log"):
            os.remove(os.path.join(REPO_ROOT, file))
            
    print("✨ Cleanup complete! No code artifacts, model cache, or Triton JIT traces remain.")

# ─────────────────────────────────── Main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DiffKV Remote CUDA Benchmark Orchestrator")
    parser.add_argument("--compile", action="store_true", help="Compile both Active and Native engines")
    parser.add_argument("--cleanup", action="store_true", help="Perform secure cleanup/destruct")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF Model ID for Active/Dense")
    parser.add_argument("--gguf-repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF", help="Hugging Face GGUF Repo ID")
    parser.add_argument("--gguf-file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf", help="GGUF Filename")
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384], help="Context lengths to benchmark")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--output", default="benchmark_results.json", help="Path to save JSON telemetry stats")
    
    args = parser.parse_args()
    
    if args.cleanup:
        perform_cleanup()
        return
        
    if args.compile:
        s1 = compile_active()
        s2 = compile_native()
        if not (s1 and s2):
            sys.exit(1)
        return
        
    # Ensure dependencies are loaded
    try:
        import huggingface_hub
        import transformers
    except ImportError:
        print("❌ Required packages not installed. Please run: pip install transformers huggingface_hub psutil")
        sys.exit(1)
        
    # Download weights
    gguf_path = download_models(args.model_id, args.gguf_repo, args.gguf_file)
    
    # Import prompt builder from bench_common
    sys.path.insert(0, HERE)
    from bench_common import build_niah_prompt
    
    results = []
    
    for ctx in args.contexts:
        print(f"\n====================== Sweeping Context: {ctx} tokens ======================")
        
        # Build prompt of target context length
        prompt_text, actual_tokens = build_niah_prompt(ctx)
        print(f"✓ Prompt generated. Length: {actual_tokens} tokens.")
        
        # Run Dense Baseline
        try:
            dense_res = run_dense_benchmark(args.model_id, prompt_text, args.gen_len)
            dense_res["ctx_target"] = ctx
            dense_res["actual_ctx"] = actual_tokens
            results.append(dense_res)
            print(f"  Dense: Prefill={dense_res['prefill_s']:.2f}s | TTFT={dense_res['ttft_s']:.2f}s | TPS={dense_res['decode_tps']:.1f} | VRAM={dense_res['peak_vram_gb']:.2f}GB")
        except Exception as e:
            print(f"  Dense: FAILED ({e})")
            
        # Run Active Sweep
        try:
            active_res = run_active_benchmark(args.model_id, prompt_text, args.gen_len)
            active_res["ctx_target"] = ctx
            active_res["actual_ctx"] = actual_tokens
            results.append(active_res)
            print(f"  Active: Prefill={active_res['prefill_s']:.2f}s | TTFT={active_res['ttft_s']:.2f}s | TPS={active_res['decode_tps']:.1f} | VRAM={active_res['peak_vram_gb']:.2f}GB")
        except Exception as e:
            print(f"  Active: FAILED ({e})")
            
        # Run Native Sweep
        try:
            native_res = run_native_benchmark(gguf_path, prompt_text, args.gen_len, ctx)
            native_res["ctx_target"] = ctx
            native_res["actual_ctx"] = actual_tokens
            results.append(native_res)
            print(f"  Native: Prefill={native_res['prefill_s']:.2f}s | TTFT={native_res['ttft_s']:.2f}s | TPS={native_res['decode_tps']:.1f} | VRAM={native_res['peak_vram_gb']:.2f}GB")
        except Exception as e:
            print(f"  Native: FAILED ({e})")
            
    # Write stats to file
    out_path = os.path.join(REPO_ROOT, args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Telemetry stats saved successfully to: {out_path}")

if __name__ == "__main__":
    main()
