"""Is the wall-vs-timer decode gap real GPU work, or removable host overhead?

DKV_TIME_ATTN measures the attention path. Wall clock is ~12% higher. Either the
difference is the rest of the model forward (irreducible) or it is host overhead
(fixable), and those call for opposite responses. Profile CUDA kernels over K
decode steps and split by whether the kernel belongs to attention.
"""
import io, os, sys
from contextlib import redirect_stdout
import torch
from torch.profiler import ProfilerActivity, profile
ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME")); sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS
for k, v in BEST_DECODE_DEFAULTS.items(): os.environ.setdefault(k, v)
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
REP, K = int(os.environ.get("REP", "1400")), int(os.environ.get("K", "40"))
w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16", "preset": "mid"},
                        device="cuda")
w.ensure_loaded(); tok = w.tokenizer
ctx = "The archive records a long sequence of unremarkable events. " * REP
prompt = tok.apply_chat_template(
    [{"role": "user", "content": ctx + "\nWrite a long essay about archives."}],
    tokenize=False, add_generation_prompt=True)
def gen(n):
    buf = io.StringIO()
    with redirect_stdout(buf):
        w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0,
                   repetition_penalty=1.0)
gen(4); torch.cuda.synchronize()
def prof(n):
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        gen(n); torch.cuda.synchronize()
    d = {}
    for e in pr.key_averages():
        us = getattr(e, "self_device_time_total", 0) or 0
        if us > 0: d[e.key] = d.get(e.key, 0) + us
    return d
base, full = prof(1), prof(1 + K)
delta = {k: full.get(k, 0) - base.get(k, 0) for k in full}
delta = {k: v for k, v in delta.items() if v > 0}
ATT = ("fmha", "attention", "efficient", "flash", "softmax", "reconstruct",
       "bmm", "sparse", "remat")
att = sum(v for k, v in delta.items() if any(a in k.lower() for a in ATT))
tot = sum(delta.values())
print(f"\nDECODE CUDA over {K} tokens: total {tot/1e3:.1f} ms "
      f"({tot/1e3/K:.2f} ms/token)")
print(f"  attention-ish kernels {att/1e3/K:6.2f} ms/token  ({100*att/tot:4.1f}%)")
print(f"  everything else       {(tot-att)/1e3/K:6.2f} ms/token  "
      f"({100*(tot-att)/tot:4.1f}%)")
print("  top kernels:")
for k, v in sorted(delta.items(), key=lambda x: -x[1])[:8]:
    print(f"    {v/1e3/K:6.3f} ms/tok  {k[:78]}")
