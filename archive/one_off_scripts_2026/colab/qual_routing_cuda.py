"""Qualitative comparison: does routing change the ANSWER, not just retrieval?

linkbench and the needle sweep both score one extracted fact. They cannot see
whether a response is coherent, whether it uses the context it was given, or
whether it invents things -- and that is exactly where routing might matter even
though every retrieval metric says it does not.

So: one long structured context, one question that requires pulling several
distant parts together, and three arms generating a real answer.

  dkv-k16     the shipped router, 16 blocks per layer per token
  dkv-all     DKV_TOPK_BLOCKS=0, attend EVERY block -- the ceiling of any
              routing scheme, since no router can select better than all
  dense       plain HF attention over the whole cache

Outputs are printed in full, unlabelled order fixed, for a human (or me) to rank.
Greedy decoding so the only differences are the attention paths.

Env: MODEL, ARM=dkv-k16|dkv-all|dense, NEW.
"""
import io
import os
import sys
import time
from contextlib import redirect_stdout

import torch

ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
ARM = os.environ.get("ARM", "dkv-k16")
NEW = int(os.environ.get("NEW", "320"))
CHUNK = int(os.environ.get("CHUNK", "1024"))

# ── The context ──────────────────────────────────────────────────────────────
# Deliberately NOT a needle in filler. Five separate "departments", each with its
# own budget, headcount, a policy, and a dependency on another department. The
# question needs facts from at least four of them plus the dependency chain, so a
# model that only retrieves one block cannot answer it, and a model that retrieves
# everything but loses coherence will show that instead.
DEPTS = [
    ("Harbour Logistics", "4.2 million credits", 118, "may not sign contracts "
     "longer than 18 months", "Meteorology"),
    ("Meteorology", "1.9 million credits", 44, "must publish forecasts within "
     "six hours of collection", "Signal Relay"),
    ("Signal Relay", "3.7 million credits", 91, "operates only between 0400 and "
     "2200 station time", "Harbour Logistics"),
    ("Archive Services", "0.8 million credits", 23, "retains records for exactly "
     "seven years before transfer", "Meteorology"),
    ("Hull Inspection", "5.6 million credits", 140, "requires two independent "
     "sign-offs per vessel", "Signal Relay"),
]
FILLER = ("Routine station activity continued without incident during the "
          "reporting period, and no exceptions were logged. ")


def build_prompt(tok, target_tokens=16000):
    parts = ["STATION OPERATIONS HANDBOOK — ANNUAL REVIEW\n\n"]
    for i, (name, budget, heads, policy, dep) in enumerate(DEPTS):
        parts.append(f"\n\nSECTION {i + 1}: {name}\n")
        parts.append(f"The {name} department was allocated {budget} for the "
                     f"period and maintained a headcount of {heads}. "
                     f"Under standing policy, {name} {policy}. "
                     f"Operationally, {name} depends on {dep} and cannot "
                     f"complete its cycle if {dep} is unavailable.\n")
        parts.append(FILLER * 60)          # push sections far apart
    body = "".join(parts)
    while len(tok(body).input_ids) < target_tokens:
        body += FILLER * 40
    return body


QUESTION = (
    "\n\nUsing only the handbook above, answer in prose:\n"
    "1. Which department has the largest budget, and which has the smallest?\n"
    "2. Total the headcount across all five departments.\n"
    "3. Trace the dependency chain starting from Archive Services as far as it "
    "goes, naming each department in order.\n"
    "4. Name one department whose policy would prevent it from operating "
    "continuously around the clock, and say why.\n"
    "5. Hull Inspection needs a vessel cleared overnight. Working through the "
    "dependency chain and the standing policies, explain whether that is "
    "possible and which department is the obstacle.\n"
)

# Ground truth, for grading afterwards:
#  1. largest Hull Inspection 5.6M, smallest Archive Services 0.8M
#  2. 118+44+91+23+140 = 416
#  3. Archive Services -> Meteorology -> Signal Relay -> Harbour Logistics
#     -> Meteorology (cycle)
#  4. Signal Relay, operates only 0400-2200


def run_dense(prompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda",
                                                 dtype=torch.float16).eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        past = None
        for s in range(0, ids.shape[1], CHUNK):
            out = model(input_ids=ids[:, s:s + CHUNK], past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
        nxt = int(out.logits[:, -1, :].argmax())
        got = [nxt]
        for _ in range(NEW - 1):
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[:, -1, :].argmax())
            if nxt == tok.eos_token_id:
                break
            got.append(nxt)
    return tok.decode(got, skip_special_tokens=True), int(ids.shape[1])


def run_dkv(prompt, attend_all):
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    PRESET = os.environ.get("PRESET", "ultra")
    os.environ["DKV_PRESET"] = PRESET
    if attend_all:
        os.environ["DKV_TOPK_BLOCKS"] = "0"
    from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16",
                                                    "preset": PRESET},
                            device="cuda")
    w.ensure_loaded()
    tok = w.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ntok = len(tok(text).input_ids)
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = w.generate(prompt=text, max_new_tokens=NEW, temperature=0.0,
                         top_p=1.0, repetition_penalty=1.0)
    if isinstance(out, dict):
        out = out.get("text") or out.get("output") or str(out)
    out = str(out)
    # The wrapper returns prompt + completion. Cut at the last line of the
    # question so only the model's own words are compared.
    marker = "is the obstacle."
    idx = out.rfind(marker)
    if idx != -1:
        out = out[idx + len(marker):]
    return out, ntok


def main():
    from transformers import AutoTokenizer
    tok0 = AutoTokenizer.from_pretrained(MODEL)
    prompt = build_prompt(tok0) + QUESTION
    t0 = time.perf_counter()
    if ARM == "dense":
        text, ntok = run_dense(prompt)
    else:
        text, ntok = run_dkv(prompt, attend_all=(ARM == "dkv-all"))
    dt = time.perf_counter() - t0
    print(f"\n===== ARM={ARM}  ctx={ntok} tokens  wall={dt:.1f}s =====")
    print(text.strip()[:2600])
    print(f"===== END {ARM} =====", flush=True)


if __name__ == "__main__":
    main()
