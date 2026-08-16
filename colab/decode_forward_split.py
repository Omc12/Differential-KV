"""Is decode's host time INSIDE the model forward, or outside it?

Decode is ~39% GPU-idle. CUDA graphs can only remove host time that sits INSIDE
the captured region (the model forward). hf_dkv_wrapper records that routed
replay measured 85 ms/token against eager's 79 -- slower, while skipping all of
the forward's Python -- which would mean the host cost is in the generate loop
instead. That determines whether the graph work is worth doing at all, so measure
it rather than assume either way.

Wraps model.forward with a timer and compares its total against wall.
"""
import io, os, sys, time
from contextlib import redirect_stdout
import torch
ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME")); sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS
for k, v in BEST_DECODE_DEFAULTS.items(): os.environ.setdefault(k, v)
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
REP, NEW = int(os.environ.get("REP", "1400")), int(os.environ.get("NEW", "80"))
w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16",
        "preset": os.environ.get("PRESET", "mid")}, device="cuda")
w.ensure_loaded()
tok = w.tokenizer
ctx = "The archive records a long sequence of unremarkable events. " * REP
prompt = tok.apply_chat_template(
    [{"role": "user", "content": ctx + "\nWrite a long detailed essay about archives."}],
    tokenize=False, add_generation_prompt=True)

STATS = {"n": 0, "t": 0.0}
_orig = w.model.forward
def timed(*a, **kw):
    t0 = time.perf_counter()
    r = _orig(*a, **kw)
    STATS["t"] += time.perf_counter() - t0
    STATS["n"] += 1
    return r
w.model.forward = timed

def run(n):
    buf = io.StringIO()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with redirect_stdout(buf):
        w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0,
                   repetition_penalty=1.0)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

run(4)
STATS["n"], STATS["t"] = 0, 0.0
t1 = run(1); f1n, f1t = STATS["n"], STATS["t"]
STATS["n"], STATS["t"] = 0, 0.0
tN = run(NEW); fNn, fNt = STATS["n"], STATS["t"]
# subtract the 1-token run so prefill cancels in both wall and forward time
dec_wall = tN - t1
dec_fwd  = fNt - f1t
dec_n    = fNn - f1n
print(f"\nDECODE {dec_n} forwards after prefill")
print(f"  wall                    {dec_wall:7.2f} s  ({1000*dec_wall/dec_n:6.2f} ms/tok)")
print(f"  inside model.forward    {dec_fwd:7.2f} s  ({1000*dec_fwd/dec_n:6.2f} ms/tok)"
      f"  {100*dec_fwd/dec_wall:5.1f}%")
print(f"  OUTSIDE the forward     {dec_wall-dec_fwd:7.2f} s  "
      f"({1000*(dec_wall-dec_fwd)/dec_n:6.2f} ms/tok)  "
      f"{100*(dec_wall-dec_fwd)/dec_wall:5.1f}%")
