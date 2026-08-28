"""Does graph replay produce the SAME TEXT as eager? Speed is worthless if not.

Runs the same prompt twice in separate processes (ARM=eager|graph) and prints a
hash plus the text, so replay can be diffed against eager rather than assumed
correct. The routed path is expected to DIVERGE (frozen dense window); the bypass
path is expected to match exactly.
"""
import hashlib, io, os, sys, time
from contextlib import redirect_stdout
import torch
ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME")); sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS
for k, v in BEST_DECODE_DEFAULTS.items(): os.environ.setdefault(k, v)
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
REP = int(os.environ.get("REP", "1400"))
NEW = int(os.environ.get("NEW", "48"))
w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16", "preset": "mid"},
                        device="cuda")
w.ensure_loaded()
tok = w.tokenizer
ctx = "The archive records a long sequence of unremarkable events. " * REP
prompt = tok.apply_chat_template(
    [{"role": "user", "content": ctx + "\nWrite a detailed essay about archives."}],
    tokenize=False, add_generation_prompt=True)
buf = io.StringIO()
t0 = time.perf_counter()
with redirect_stdout(buf):
    out = w.generate(prompt=prompt, max_new_tokens=NEW, temperature=0.0,
                     top_p=1.0, repetition_penalty=1.0)
dt = time.perf_counter() - t0
gen = str(out).rsplit("assistant", 1)[-1].strip()
print(f"ARM={os.environ.get('ARM','?')} ctx={len(tok(prompt).input_ids)} "
      f"wall={dt:.1f}s md5={hashlib.md5(gen.encode()).hexdigest()[:16]}")
print(f"TEXT: {gen[:220]}")
