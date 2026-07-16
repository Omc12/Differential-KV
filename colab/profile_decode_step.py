"""
profile_decode_step_v2.py  —  Instrumented decode profiler using EXACT eval setup

Run on Lightning AI:
    cd /home/zeus/Differential-KV && python colab/profile_decode_step_v2.py 2>&1 | tee /home/zeus/profile_v2.txt

Mirrors run_nat_128k_q4_eval.py setup exactly, then patches kv_manager methods
to add per-call timing and report a breakdown after 20 decode steps.
"""

import sys, os, time
sys.path.insert(0, "/home/zeus/Differential-KV/ACTIVE_RUNTIME")
os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ── 1. Load model (same as eval) ──────────────────────────────────────────────
from transformers import BitsAndBytesConfig
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
config = {
    "mode": "fp16", "quantization": "nf4",
    "block_size": 256, "rank": 16,
    "micro_block_size": 256, "preset": "mid", "serving_mode": "balanced"
}

print("Loading model...")
t0 = time.perf_counter()
w = DiffKVHFWrapper(model_id=MODEL_ID, config=config,
                    torch_dtype=torch.float16, device=device,
                    quantization_config=quantization_config)
w.ensure_loaded()
tok, mgr, model = w.tokenizer, w.manager, w.model
print(f"Loaded in {time.perf_counter()-t0:.1f}s\n")

# ── 2. Build 4K prompt (same as eval) ─────────────────────────────────────────
from transformers import AutoTokenizer
TARGET_LEN = 4096

# Build prompt exactly like build_prompt_for_len
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_base = "The following is a detailed technical document about machine learning and artificial intelligence systems. "
_full = (_base * ((TARGET_LEN * 6) // len(_base) + 1))
ids = _tok.encode(_full, add_special_tokens=False)[:TARGET_LEN]
prompt_len = len(ids)
print(f"Prompt tokens: {prompt_len}")

# ── 3. Session setup (IDENTICAL to eval) ──────────────────────────────────────
sid = "profile_v2_session"
mgr.clear_session(sid)
if not hasattr(w, "_session_token_ids"):
    w._session_token_ids = {}
w._session_token_ids[sid] = []

mgr.init_session(sid, prefill_len=prompt_len)
mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
model._diffkv_session_ids = [sid]

# ── 4. Prefill (same as eval) ─────────────────────────────────────────────────
CH = int(getattr(getattr(mgr, "config", None), "prefill_chunk_size", 1024))
if torch.cuda.is_available() and hasattr(mgr, "get_session_micro_block_size"):
    _mbs = mgr.get_session_micro_block_size(sid)
    _block_capacity = max(2, int(_mbs) + 1)
    CH = ((CH + _block_capacity - 1) // _block_capacity) * _block_capacity
print("Running prefill...", flush=True)
t_pre = time.perf_counter()
out = None
with torch.inference_mode():
    for cs in range(0, len(ids), CH):
        ch = ids[cs:cs+CH]
        out = model(
            input_ids=torch.tensor([ch], device=device),
            position_ids=torch.tensor([list(range(cs, cs+len(ch)))], device=device),
            use_cache=True,
        )
last_logits_gpu = out.logits[0, -1].float()
mgr.compress_deferred_prefill_blocks(sid)
print(f"Prefill done in {time.perf_counter()-t_pre:.2f}s\n")

# ── 5. Exact same post-prefill steps as eval ──────────────────────────────────
# Compression barrier: inspect block states, not only _pending_cpu_blocks.
# GPU-rSVD does not use the CPU pending counter, while a CPU fallback may still
# have SUBMITTED blocks that must be finalized before profiling decode.
_bar = time.perf_counter()
while True:
    if hasattr(mgr, "finalize_compressed_blocks"):
        mgr.finalize_compressed_blocks()
    _blocks = getattr(mgr, "_streaming_mgr", None)
    _blocks = _blocks.session_blocks.get(sid, {}) if _blocks is not None else {}
    _p = sum(
        1 for _layer in _blocks.values() for _block in _layer
        if getattr(_block, "state", None) in ("SUBMITTED", "CPU_COMPRESSED")
    )
    if _p <= 0:
        break
    if time.perf_counter() - _bar > float(os.environ.get("DIFFKV_COMPRESSION_TIMEOUT_S", "30")):
        print(f"WARNING: compression barrier timed out with {_p} blocks pending", flush=True)
        break
    time.sleep(0.05)

if hasattr(mgr, "finalize_srl_index"):
    mgr.finalize_srl_index(sid, cached_len=0)
if hasattr(mgr, "_prefill_kv_capture"):
    mgr._prefill_kv_capture.pop(sid, None)

# ── 6. Instrument mgr methods ─────────────────────────────────────────────────
_timers = {
    "ingest_streaming":          [0.0, 0],
    "get_cached_decode_blocks":  [0.0, 0],
    "assemble_dense_window_kv":  [0.0, 0],
    "finalize_compressed_blocks":[0.0, 0],
}

_orig_ingest = mgr.ingest_streaming
def _t_ingest(*a, **kw):
    t = time.perf_counter()
    r = _orig_ingest(*a, **kw)
    _timers["ingest_streaming"][0] += (time.perf_counter()-t)*1000
    _timers["ingest_streaming"][1] += 1
    return r
mgr.ingest_streaming = _t_ingest

if hasattr(mgr, "get_cached_decode_blocks"):
    _orig_gcd = mgr.get_cached_decode_blocks
    def _t_gcd(*a, **kw):
        t = time.perf_counter()
        r = _orig_gcd(*a, **kw)
        _timers["get_cached_decode_blocks"][0] += (time.perf_counter()-t)*1000
        _timers["get_cached_decode_blocks"][1] += 1
        return r
    mgr.get_cached_decode_blocks = _t_gcd

if hasattr(mgr, "assemble_dense_window_kv"):
    _orig_adw = mgr.assemble_dense_window_kv
    def _t_adw(*a, **kw):
        t = time.perf_counter()
        r = _orig_adw(*a, **kw)
        _timers["assemble_dense_window_kv"][0] += (time.perf_counter()-t)*1000
        _timers["assemble_dense_window_kv"][1] += 1
        return r
    mgr.assemble_dense_window_kv = _t_adw

if hasattr(mgr, "finalize_compressed_blocks"):
    _orig_fcb = mgr.finalize_compressed_blocks
    def _t_fcb(*a, **kw):
        t = time.perf_counter()
        r = _orig_fcb(*a, **kw)
        _timers["finalize_compressed_blocks"][0] += (time.perf_counter()-t)*1000
        _timers["finalize_compressed_blocks"][1] += 1
        return r
    mgr.finalize_compressed_blocks = _t_fcb

# ── 7. Warmup 3 decode steps ──────────────────────────────────────────────────
static_in  = torch.zeros((1, 1), dtype=torch.long, device=device)
static_pos = torch.zeros((1, 1), dtype=torch.long, device=device)
cur = prompt_len

print("Warming up 3 decode steps...")
with torch.inference_mode():
    for _ in range(3):
        nid = int(torch.argmax(last_logits_gpu).item())
        static_in[0, 0] = nid
        static_pos[0, 0] = cur
        out = model(input_ids=static_in, position_ids=static_pos, use_cache=True)
        last_logits_gpu = out.logits[0, -1].float()
        cur += 1
torch.cuda.synchronize()

# Reset timers after warmup
for k in _timers:
    _timers[k] = [0.0, 0]

print("Warmup done.\n")

# ── 8. Profile 20 decode steps (synced) ──────────────────────────────────────
N = 20
step_times = []
print(f"Profiling {N} decode steps...")

with torch.inference_mode():
    for step in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        nid = int(torch.argmax(last_logits_gpu).item())
        t_argmax = time.perf_counter()

        static_in[0, 0] = nid
        static_pos[0, 0] = cur
        out = model(input_ids=static_in, position_ids=static_pos, use_cache=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        last_logits_gpu = out.logits[0, -1].float()
        cur += 1

        total_ms  = (t1 - t0) * 1000
        argmax_ms = (t_argmax - t0) * 1000
        model_ms  = (t1 - t_argmax) * 1000
        step_times.append(total_ms)
        print(f"  Step {step+1:2d}: total={total_ms:7.1f}ms  argmax={argmax_ms:.1f}ms  model={model_ms:.1f}ms")

avg_ms = sum(step_times) / len(step_times)
tps    = 1000.0 / avg_ms

print(f"\nAvg: {avg_ms:.1f}ms/token  ({tps:.1f} TPS)")

# ── 9. Phase breakdown ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Phase breakdown (over all profiled steps):")
print(f"{'='*60}")
total_tracked = 0.0
for name, (ms, calls) in _timers.items():
    if calls > 0:
        print(f"  {name:<35} {ms:8.1f}ms total  ({calls} calls @ {ms/calls:.3f}ms/call)")
        total_tracked += ms
    else:
        print(f"  {name:<35}    0 calls — NOT BEING CALLED!")

tracked_per_step = total_tracked / N
model_per_step   = sum(t1-t0 for _ in step_times) / N if step_times else avg_ms
gpu_other = avg_ms - tracked_per_step

print(f"\n  Tracked Python overhead:   {tracked_per_step:.1f}ms/step")
print(f"  GPU + untracked overhead:  {gpu_other:.1f}ms/step")
print(f"  Dense baseline:            ~88ms/step (11.3 TPS)")
print(f"  Total excess:              {avg_ms - 88:.1f}ms/step")

print(f"""
KEY INTERPRETATION:
  - If 'ingest_streaming' = 0 calls → DiffKV decode path NOT active!
  - If tracked overhead << total → bottleneck is in GPU / Triton kernels or
    inside model.forward() code NOT instrumented here (e.g. SRL routing,
    factual store query — these are inside diffkv_attention.py, not mgr)
  - If 'get_cached_decode_blocks' = 0 calls → session_ids not wired correctly
""")

# Restore originals
mgr.ingest_streaming = _orig_ingest
if hasattr(mgr, "get_cached_decode_blocks"):
    mgr.get_cached_decode_blocks = _orig_gcd
if hasattr(mgr, "assemble_dense_window_kv"):
    mgr.assemble_dense_window_kv = _orig_adw
if hasattr(mgr, "finalize_compressed_blocks"):
    mgr.finalize_compressed_blocks = _orig_fcb
