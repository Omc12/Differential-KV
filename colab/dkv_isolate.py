#!/usr/bin/env python3
"""Isolate WHERE DKV CUDA decode diverges — prefill vs decode, bypass vs engaged.

The garbage output (degenerate loops) at BOTH needle depths (0.1 compressed AND
0.9 dense-window) means the decode-attention path is broken independent of
compression. This probe narrows it to a single stage in ONE run, so we instrument
the right place next instead of reading 1500 lines blind.

For a set of context lengths it:
  1. runs the SAME needle prompt through PLAIN dense HF (ground truth), and
  2. runs it through DKV,
and prints, for each, (a) the argmax of the PREFILL last-position logits (the
first generated token — isolates prefill fidelity) and (b) the first few greedy
tokens. It also reports whether DKV BYPASSED (short prompt → original attn) or
ENGAGED (its own path).

Reading the result:
  * short ctx (bypass) matches dense, longer ctx (engaged) diverges  → the DKV
    engaged path is the bug (prefill or decode per which token diverges).
  * DKV prefill-argmax != dense prefill-argmax                    → PREFILL is
    broken (cross-chunk attention through the manager) — decode is downstream.
  * prefill-argmax matches but the greedy continuation degenerates    → DECODE is
    broken (reconstruction / RoPE / assembly), prefill is fine.
Run on the A100:  python colab/dkv_isolate.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)


def make_prompt(tok, ctx_len, needle, question, depth=0.9):
    filler = ("Quantum computing is a multidisciplinary field comprising aspects of computer "
              "science, physics, and mathematics that utilizes quantum mechanics to solve complex "
              "problems faster than on classical computers. ")
    ftoks = tok.encode(filler, add_special_tokens=False)
    ntoks = tok.encode(needle, add_special_tokens=False)
    reps = max(1, (ctx_len // max(1, len(ftoks))) + 1)
    body = (ftoks * reps)[:max(64, ctx_len - len(ntoks) - 40)]
    ins = int(len(body) * depth)
    body = body[:ins] + ntoks + body[ins:]
    ctx = tok.decode(body)
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ctx + "\n" + question}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return ctx + "\n" + question + "\nAssistant:"


def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    """EXACT copy of ACTIVE_RUNTIME/tests/test_niah.py:make_niah_prompt — the
    builder whose 8k placements CUDA mangles. Use --builder niah to reproduce
    the pytest failure here (with a dense baseline for comparison)."""
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    target_filler_tokens = context_length - len(needle_tokens) - 100
    if target_filler_tokens < 0:
        target_filler_tokens = 100
    num_repeats = (target_filler_tokens // len(filler_tokens)) + 1
    all_filler_tokens = (filler_tokens * num_repeats)[:target_filler_tokens]
    insert_idx = int(len(all_filler_tokens) * depth)
    part1_text = tokenizer.decode(all_filler_tokens[:insert_idx])
    part2_text = tokenizer.decode(all_filler_tokens[insert_idx:])
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1_text + "\n" + needle + "\n" + part2_text + "\n\n"
        + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def dense_probe(model_id, prompt, device, tok, n=12):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = m(ids, use_cache=True)
    first = int(out.logits[0, -1].argmax())
    past = out.past_key_values
    gen = [first]
    cur = ids.shape[1]
    nxt = torch.tensor([[first]], device=device)
    with torch.no_grad():
        for _ in range(n - 1):
            o = m(nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            t = int(o.logits[0, -1].argmax())
            gen.append(t)
            nxt = torch.tensor([[t]], device=device)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return first, tok.decode(gen)


def dkv_probe(model_id, prompt, device, n=12):
    from serving.hf_dkv_wrapper import DKVHFWrapper
    w = DKVHFWrapper(model_id, config={"rank": 32}, device=device)
    txt = w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
    sid = w.active_session or "default"
    all_ids = w._session_token_ids.get(sid, [])
    ptoks = len(w.tokenizer.encode(prompt))
    gen = all_ids[ptoks:]
    first = gen[0] if gen else None
    decoded = w.tokenizer.decode(gen, skip_special_tokens=True)
    try:
        w.close()
    except Exception:
        pass
    return first, decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctxs", default="200,1500,4000")   # 200 likely bypass; 1500/4000 engage
    ap.add_argument("--depth", type=float, default=0.9,
                    help="needle depth 0..1 (0.1 = oldest/most-compressed region, the hard case)")
    ap.add_argument("--builder", choices=["isolate", "niah"], default="isolate",
                    help="isolate=make_prompt (retrieves); niah=EXACT test_niah.py builder (CUDA mangles at 8k)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."

    print(f"model={args.model} device={device} depth={args.depth} builder={args.builder}\n")
    for ctx in [int(c) for c in args.ctxs.split(",")]:
        if args.builder == "niah":
            prompt = make_niah_prompt(tok, ctx, args.depth, needle, question)
        else:
            prompt = make_prompt(tok, ctx, needle, question, depth=args.depth)
        plen = len(tok.encode(prompt))
        d_first, d_txt = dense_probe(args.model, prompt, device, tok)
        k_first, k_txt = dkv_probe(args.model, prompt, device)
        d_ft = tok.decode([d_first]) if d_first is not None else "?"
        k_ft = tok.decode([k_first]) if k_first is not None else "?"
        match = "SAME" if d_first == k_first else "DIVERGE"
        print(f"[ctx~{ctx} plen={plen}]")
        print(f"   DENSE  first-token={d_ft!r}  greedy={d_txt!r}")
        print(f"   DKV first-token={k_ft!r}  greedy={k_txt!r}")
        print(f"   prefill first-token: {match}"
              f"{'  ← DKV prefill logits already wrong' if d_first != k_first else '  ← prefill OK; check decode continuation'}\n")


if __name__ == "__main__":
    main()
