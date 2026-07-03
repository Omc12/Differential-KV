#!/usr/bin/env python3
"""NIAH recall + decode-TPS harness for the DiffKV active runtime.

Loads the wrapper ONCE and loops over (ctx, depth), building a needle prompt at
each, running greedy decode, and checking the needle in the GENERATED tokens only
(NOT the prompt-inclusive response — that would false-positive on the planted
needle). The decode path is selected by DIFFKV_COMPRESSED_DECODE in the env,
exactly as in production. Run once per mode:

    DIFFKV_COMPRESSED_DECODE=1 python benchmarks/niah_recall.py --ctx 4096 8192 16384 --depths 0.1 0.5 0.9
    DIFFKV_COMPRESSED_DECODE=0 python benchmarks/niah_recall.py --ctx 4096 8192 16384 --depths 0.1 0.5 0.9

TPS here is approximate (model reused across cells, per-cell kernel recompiles);
use bench_worker.py for authoritative TPS. This harness's job is the recall gate.
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

NEEDLE = "OMEGA-7741-DELTA"
NEEDLE_SENT = f"The secret passcode is {NEEDLE}."
QUESTION = "What is the secret passcode? Repeat it exactly."

NEEDLES_MULTI = ["OMEGA-7741-DELTA", "SIGMA-9923-BETA", "THETA-1105-ALPHA"]
NEEDLE_SENTS_MULTI = [
    "The first secret passcode is OMEGA-7741-DELTA.",
    "The second secret passcode is SIGMA-9923-BETA.",
    "The third secret passcode is THETA-1105-ALPHA."
]
QUESTION_MULTI = "What are the three secret passcodes? List them all in order."

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)


def build_prompt(tok, ctx, depth):
    filler = tok.encode(FILLER, add_special_tokens=False)
    needle = tok.encode(NEEDLE_SENT + "\n", add_special_tokens=False)
    q = tok.encode(QUESTION, add_special_tokens=False)
    budget = ctx - len(needle) - len(q) - 80
    if budget < 100:
        budget = 100
    reps = budget // len(filler) + 1
    allf = (filler * reps)[:budget]
    at = int(len(allf) * depth)
    p1 = tok.decode(allf[:at])
    p2 = tok.decode(allf[at:])
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + p1 + "\n" + NEEDLE_SENT + "\n" + p2 + "\n\n"
        + QUESTION + "<|im_end|>\n<|im_start|>assistant\n"
    )


def build_multi_needle_prompt(tok, ctx):
    filler = tok.encode(FILLER, add_special_tokens=False)
    sents_tokens = []
    for s in NEEDLE_SENTS_MULTI:
        sents_tokens.extend(tok.encode(s + "\n", add_special_tokens=False))
    q_tokens = tok.encode(QUESTION_MULTI, add_special_tokens=False)
    
    budget = ctx - len(sents_tokens) - len(q_tokens) - 80
    if budget < 100:
        budget = 100
    reps = budget // len(filler) + 1
    allf = (filler * reps)[:budget]
    
    at1 = int(len(allf) * 0.25)
    at2 = int(len(allf) * 0.50)
    at3 = int(len(allf) * 0.75)
    
    p1 = tok.decode(allf[:at1])
    p2 = tok.decode(allf[at1:at2])
    p3 = tok.decode(allf[at2:at3])
    p4 = tok.decode(allf[at3:])
    
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + p1 + "\n" + NEEDLE_SENTS_MULTI[0] + "\n"
        + p2 + "\n" + NEEDLE_SENTS_MULTI[1] + "\n"
        + p3 + "\n" + NEEDLE_SENTS_MULTI[2] + "\n"
        + p4 + "\n\n" + QUESTION_MULTI + "<|im_end|>\n<|im_start|>assistant\n"
    )


def run(model_id, contexts, depths, gen, use_bench=False, rank=16, multi_needle=False):
    import numpy as np
    import torch
    import mlx.core as mx  # noqa: F401
    sys.path.insert(0, HERE)
    if use_bench:
        from bench_common import build_niah_prompt  # the harder, on-topic NIAH prompt
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper

    cfg = {"quantization": "int4", "rank": rank, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DiffKVHFWrapper(model_id=model_id, config=cfg)
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model

    mode = os.environ.get("DIFFKV_COMPRESSED_DECODE", "auto")
    src = "multi-needle" if multi_needle else ("bench_common" if use_bench else "ai-history")
    print(f"DIFFKV_COMPRESSED_DECODE={mode}  gen={gen}  prompt={src}", flush=True)
    print(f"{'ctx':>7} {'depth':>5} {'recall':>6} {'tps':>6}   sample", flush=True)

    iter_depths = [0.5] if (use_bench or multi_needle) else depths
    results = []
    for ctx in contexts:
        for depth in iter_depths:
            if multi_needle:
                prompt = build_multi_needle_prompt(tok, ctx)
            elif use_bench:
                prompt, _ = build_niah_prompt(ctx, tok)
            else:
                prompt = build_prompt(tok, ctx, depth)
            ids = tok.encode(prompt)
            sid = "niah"
            mgr.clear_session(sid)
            wrapper._session_token_ids[sid] = []
            mgr.init_session(sid, prefill_len=len(ids))
            mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
            model._diffkv_session_ids = [sid]

            CH = 512
            output = None
            for cs in range(0, len(ids), CH):
                chunk = ids[cs:cs + CH]
                ct = torch.tensor([chunk], dtype=torch.long)
                pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
                output = model(ct, pt)
                mgr.compress_deferred_prefill_blocks(sid)
            logits = output.logits[0, -1].cpu().numpy()

            cur = len(ids)
            generated = []
            # one warmup decode step before timing
            nid = int(np.argmax(logits))
            generated.append(nid)
            mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
            output = model(torch.tensor([[nid]], dtype=torch.long),
                           torch.tensor([[cur]], dtype=torch.long))
            logits = output.logits[0, -1].cpu().numpy()
            cur += 1

            t0 = time.perf_counter()
            steps = 0
            for _ in range(gen - 1):
                nid = int(np.argmax(logits))
                generated.append(nid)
                steps += 1
                text_so_far = tok.decode(generated)
                if multi_needle:
                    if all(n in text_so_far for n in NEEDLES_MULTI):
                        break
                else:
                    if NEEDLE in text_so_far:
                        break
                mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
                output = model(torch.tensor([[nid]], dtype=torch.long),
                               torch.tensor([[cur]], dtype=torch.long))
                logits = output.logits[0, -1].cpu().numpy()
                cur += 1
            dt = time.perf_counter() - t0

            text = tok.decode(generated)
            if multi_needle:
                ok = all(n in text for n in NEEDLES_MULTI)
            else:
                ok = NEEDLE in text
            tps = steps / dt if dt > 0 else 0.0
            results.append((ctx, depth, ok, tps))
            print(f"{ctx:>7} {depth:>5.1f} {('Y' if ok else 'N'):>6} {tps:>6.1f}   {text[:140]!r}",
                  flush=True)

    print(f"\nRECALL: {sum(1 for _,_,ok,_ in results if ok)}/{len(results)} cells", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", type=int, default=[4096, 8192, 16384])
    ap.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--bench", action="store_true",
                    help="use the harder on-topic bench_common NIAH prompt")
    ap.add_argument("--multi-needle", action="store_true",
                    help="use 3 distinct passcodes distributed at different depths")
    ap.add_argument("--rank", type=int, default=16, help="SVD rank for block compression")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    
    # Auto-increase generation length for multi-needle if not customized
    generation_length = args.gen
    if args.multi_needle and args.gen == 24:
        generation_length = 64
        
    run(args.model, args.ctx, args.depths, generation_length, use_bench=args.bench, rank=args.rank, multi_needle=args.multi_needle)
