"""Stage 0: where does the HOST time go on the generate() path?

Three optimisations were tried by inference from op tables and none moved decode
(Path A vs B: 0%; routing top-K 4x cut: 19%; static gather buffers: 0%). The
common failure was guessing which host work dominates instead of measuring it.
This measures it.

Why not profile_decode_step.py: that harness drives model(input_ids=...)
directly, bypassing the wrapper's session setup and routing, so DKV's fused
decode never engages (its "dkv" bucket reads 0.0 ms and the COMBINED banner is
absent). It profiles a path nobody runs.

This profiles generate(), sorts by SELF CPU time, and prints the accept/abort
number the CUDA-graph plan asks for.

    python colab/profile_decode_host.py --model Qwen/Qwen2.5-1.5B-Instruct --ctx 16000
"""
import argparse
import os
import sys
import time

os.environ.setdefault("DKV_TRITON_STRICT", "1")
os.environ.setdefault("DKV_USE_ATTENTION_INTERFACE", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))

import torch
from torch.profiler import profile, ProfilerActivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=192,
                    help="MUST be large. generate() re-prefills every call, so a\n                         small count leaves the window prefill-dominated: at 16\n                         steps the top CPU entries were rSVD compression, not\n                         decode at all.")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()

    from colab.bench_dkv_tps import build_prompt          # same prompt builder
    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper

    w = PyTorchDKVHFWrapper(model_id=args.model, config={"mode": "fp16"},
                            device="cuda")
    w.ensure_loaded()
    body = build_prompt(w.tokenizer, args.ctx)
    prompt = w.tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False,
        add_generation_prompt=True)
    ntok = len(w.tokenizer(prompt).input_ids)

    # Warm up: JIT, Inductor, allocator, and the first-call routing build.
    w.generate(prompt=prompt, max_new_tokens=8, temperature=0.0, top_p=1.0,
               repetition_penalty=1.0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        r = w.generate(prompt=prompt, max_new_tokens=args.steps, temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gen = max(len(w.tokenizer(r).input_ids) - ntok, 1)

    ev = prof.key_averages()
    # NOTE: this window INCLUDES one prefill (generate re-prefills every call),
    # so per-token figures below are upper bounds. The point is the RANKING and
    # the CPU/GPU split, both of which prefill does not distort in kind.
    self_cpu = sum(e.self_cpu_time_total for e in ev) / 1e6
    self_cuda = sum(getattr(e, "self_device_time_total",
                            getattr(e, "self_cuda_time_total", 0)) for e in ev) / 1e6
    launches = sum(e.count for e in ev
                   if getattr(e, "self_device_time_total",
                              getattr(e, "self_cuda_time_total", 0)) > 0)

    print("\n" + "=" * 78)
    print(f"  {args.model}   prompt {ntok} tok   generated {gen} tok")
    print(f"  wall              {wall:8.3f} s")
    print(f"  self CPU total    {self_cpu:8.3f} s   ({100*self_cpu/wall:5.1f}% of wall)")
    print(f"  self GPU total    {self_cuda:8.3f} s   ({100*self_cuda/wall:5.1f}% of wall)")
    print(f"  GPU-emitting ops  {launches:8d}   ({launches/gen:8.0f} per generated token)")
    print("=" * 78)

    verdict = 100 * self_cuda / wall
    print("\n  CUDA-GRAPH PLAN, STAGE 0 CRITERION")
    if verdict > 60:
        print(f"  GPU busy {verdict:.0f}% > 60%  ->  ABORT. The cost is real compute;")
        print("  graphs remove dispatch, not compute. Do not build Stages 1-4.")
    elif verdict < 25:
        print(f"  GPU busy {verdict:.0f}% < 25%  ->  PROCEED. Dispatch-bound, which is")
        print("  exactly what CUDA graphs eliminate.")
    else:
        print(f"  GPU busy {verdict:.0f}% is between 25% and 60% -> inconclusive.")
        print("  Graphs would help but are not obviously the biggest lever.")

    print(f"\n  TOP {args.topk} BY SELF CPU TIME  (this is what a graph removes)")
    print("  " + "-" * 74)
    rows = sorted(ev, key=lambda e: e.self_cpu_time_total, reverse=True)[:args.topk]
    for e in rows:
        ms = e.self_cpu_time_total / 1e3
        print(f"  {e.key[:52]:52} {ms:9.1f} ms {100*ms/(self_cpu*1e3):6.1f}%  n={e.count}")
    print("=" * 78)
    print("\nRead the CPU column, not the CUDA one. Three optimisations aimed at")
    print("GPU-side work already returned ~0; the remaining cost is host-side.")
    _prefill_frac = ntok / max(ntok + gen, 1)
    print(f"\nWINDOW COMPOSITION: {ntok} prefilled vs {gen} generated tokens.")
    if gen < 64:
        print("  !! PREFILL-DOMINATED. generate() re-prefills every call, so these")
        print("     rankings describe COMPRESSION, not decode. Re-run with more")
        print("     --steps before drawing any decode conclusion.")


if __name__ == "__main__":
    main()
