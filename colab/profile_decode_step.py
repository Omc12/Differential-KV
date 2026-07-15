"""
profile_decode_step.py
Run on Lightning AI to pinpoint exactly where the 1.4 TPS bottleneck is.
Usage: python colab/profile_decode_step.py
"""
import sys, os, time, gc
import torch

# ── Path setup identical to run_nat_128k_q4_eval.py ─────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(REPO, "ACTIVE_RUNTIME")
NATIVE  = os.path.join(RUNTIME, "native_core")
DIFFKV_CORE = os.path.join(NATIVE, "diffkv_core")
for p in [RUNTIME, NATIVE, DIFFKV_CORE]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
transformers.logging.set_verbosity_error()

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
device   = "cuda:0"
N_STEPS  = 30     # measure 30 decode steps
N_WARMUP = 3

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Building 4K prompt...")
unit = "The quick brown fox jumped over the lazy dog. "
prompt_text = unit * (4096 // len(tok.encode(unit)) + 1)
ids = tok.encode(prompt_text)[:4096]
print(f"Prompt length: {len(ids)} tokens")

# ── Dense baseline ────────────────────────────────────────────────────
print("\n=== DENSE BASELINE ===")
from transformers import AutoModelForCausalLM
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
dense_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=qcfg,
                                                    device_map="auto", trust_remote_code=True)
dense_model.eval()

# Prefill
CH = 128; past_kv = None
with torch.no_grad():
    for cs in range(0, len(ids), CH):
        ch = ids[cs:cs+CH]
        pos = torch.tensor([list(range(cs, cs+len(ch)))], device=device)
        out = dense_model(torch.tensor([ch], device=device), position_ids=pos,
                          past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values

static_in  = torch.zeros((1,1), dtype=torch.long, device=device)
static_pos = torch.zeros((1,1), dtype=torch.long, device=device)
last_logits = out.logits[0,-1].float()
cur = len(ids)

# Warmup
with torch.no_grad():
    for _ in range(N_WARMUP):
        nid = int(torch.argmax(last_logits).item())
        static_in[0,0] = nid; static_pos[0,0] = cur
        out2 = dense_model(static_in, position_ids=static_pos, past_key_values=past_kv, use_cache=True)
        last_logits = out2.logits[0,-1].float()
        past_kv = out2.past_key_values
        cur += 1

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for step in range(N_STEPS):
        nid = int(torch.argmax(last_logits).item())
        static_in[0,0] = nid; static_pos[0,0] = cur
        out2 = dense_model(static_in, position_ids=static_pos, past_key_values=past_kv, use_cache=True)
        last_logits = out2.logits[0,-1].float()
        past_kv = out2.past_key_values
        cur += 1
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"Dense TPS: {N_STEPS/dt:.1f}  ({dt*1000/N_STEPS:.1f}ms/tok)")

del dense_model, past_kv; gc.collect(); torch.cuda.empty_cache()

# ── Compressed — per-section timing ─────────────────────────────────
print("\n=== COMPRESSED TIMING ===")
from serving.hf_diffkv_wrapper import DiffKVWrapper

w = DiffKVWrapper(model_id=MODEL_ID, device=device, rank=8,
                  block_size=64, max_blocks=512)
model = w.model
mgr   = w.manager

sid = "prof_session"
mgr.init_session(sid, prefill_len=len(ids))
mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
model._diffkv_session_ids = [sid]

# Prefill
with torch.no_grad():
    for cs in range(0, len(ids), CH):
        ch = ids[cs:cs+CH]
        out = model(torch.tensor([ch], device=device),
                    torch.tensor([list(range(cs, cs+len(ch)))], device=device))
        mgr.compress_deferred_prefill_blocks(sid)

last_logits = out.logits[0,-1].float()
cur = len(ids)

static_in  = torch.zeros((1,1), dtype=torch.long, device=device)
static_pos = torch.zeros((1,1), dtype=torch.long, device=device)

# ── cProfile the hot decode loop ────────────────────────────────────
print("\n--- cProfile: top 30 functions by cumulative time ---")
import cProfile, pstats, io

def one_step():
    nid = int(torch.argmax(last_logits).item())
    static_in[0,0] = nid; static_pos[0,0] = cur
    with torch.no_grad():
        out3 = model(static_in, static_pos)
    return out3.logits[0,-1].float()

# Warmup
for _ in range(N_WARMUP):
    last_logits = one_step()

torch.cuda.synchronize()

pr = cProfile.Profile()
pr.enable()
for _ in range(N_STEPS):
    last_logits = one_step()
torch.cuda.synchronize()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())

# ── Wall-clock TPS ───────────────────────────────────────────────────
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for step in range(N_STEPS):
        nid = int(torch.argmax(last_logits).item())
        static_in[0,0] = nid; static_pos[0,0] = cur
        out3 = model(static_in, static_pos)
        last_logits = out3.logits[0,-1].float()
        cur += 1
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"\nCompressed TPS: {N_STEPS/dt:.1f}  ({dt*1000/N_STEPS:.1f}ms/tok)")
