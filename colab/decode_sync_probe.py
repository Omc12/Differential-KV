"""GPU->CPU syncs during DECODE, with the DKV frame that caused each.

CUDA graph capture of the routed decode path dies with
cudaErrorStreamCaptureInvalidated, which is what a synchronisation inside the
captured region produces. Every sync listed here that sits INSIDE model.forward
is a capture blocker and has to go before graphs can work.
"""
import io, os, sys, traceback, warnings
from collections import Counter
from contextlib import redirect_stdout
import torch
ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME")); sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
os.environ.setdefault("DKV_GRAPH_SAFE_ROUTING", "1")
from serving.decode_config import BEST_DECODE_DEFAULTS
for k, v in BEST_DECODE_DEFAULTS.items(): os.environ.setdefault(k, v)
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
REP = int(os.environ.get("REP", "1400"))
HITS = Counter()
INSIDE = {"v": False}

def _show(message, category, filename, lineno, file=None, line=None):
    frame = "?"
    for f in reversed(traceback.extract_stack()[:-1]):
        if "Differential KV" in f.filename and "dec_sync" not in f.filename:
            frame = f"{f.filename.replace(ROOT,'').replace(chr(92),'/')}:{f.lineno} {f.name}"
            break
    HITS[("IN-FORWARD " if INSIDE["v"] else "outside    ") + frame] += 1

w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16",
        "preset": os.environ.get("PRESET", "mid")}, device="cuda")
w.ensure_loaded()
_orig = w.model.forward
def marked(*a, **kw):
    INSIDE["v"] = True
    try:
        return _orig(*a, **kw)
    finally:
        INSIDE["v"] = False
w.model.forward = marked
tok = w.tokenizer
ctx = "The archive records a long sequence of unremarkable events. " * REP
prompt = tok.apply_chat_template(
    [{"role": "user", "content": ctx + "\nWrite an essay about archives."}],
    tokenize=False, add_generation_prompt=True)
def gen(n):
    buf = io.StringIO()
    with redirect_stdout(buf):
        w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0,
                   repetition_penalty=1.0)
gen(3); torch.cuda.synchronize()
HITS.clear()
warnings.showwarning = _show
warnings.simplefilter("always")
torch.cuda.set_sync_debug_mode("warn")
try:
    gen(6)
finally:
    torch.cuda.set_sync_debug_mode("default")
tot = sum(HITS.values())
inside = sum(v for k, v in HITS.items() if k.startswith("IN-FORWARD"))
print(f"\nDECODE SYNCS over 6 tokens: {tot} total, {inside} INSIDE model.forward")
print("  (only IN-FORWARD ones block CUDA graph capture)")
for frame, n in HITS.most_common(14):
    print(f"  {n:4d}  {frame}"[:150])
