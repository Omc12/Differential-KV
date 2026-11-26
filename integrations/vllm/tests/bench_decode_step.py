"""Time one layer's decode step, piece by piece (CUDA events, after warmup).

    python integrations/vllm/tests/bench_decode_step.py --ctx 16384

Pieces: bounds reset, fused store, page lists (all / routed), fused decode;
and, for scale, vLLM's own Triton unified attention on 16-bit pages of the
same shape when it can be imported.
"""
import argparse
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def timeit(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000          # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--hq", type=int, default=12)
    ap.add_argument("--hk", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    from dkv_vllm.slot_codecs import K8V4
    from dkv_vllm.triton_decode import k8v4_decode, k8v4_store
    from dkv_vllm.codec_attn import DKVCodecAttentionImpl
    dev, BS = "cuda", a.bs
    c = K8V4(a.d)
    P = (a.ctx + 512 + BS - 1) // BS
    B = P + 8
    pages = torch.zeros(B, a.hk, BS, c.slot_bytes, dtype=torch.uint8, device=dev)
    kmin = torch.full((B + 1, a.hk, a.d), float("inf"), device=dev)
    kmax = torch.full_like(kmin, float("-inf"))
    k = torch.randn(B * BS, a.hk, a.d, device=dev, dtype=torch.float16)
    for s0 in range(0, B * BS, 4096):                     # fill the cache
        sl = torch.arange(s0, min(s0 + 4096, B * BS), device=dev, dtype=torch.int32)
        k8v4_store(k[sl.long()], k[sl.long()], sl, pages, kmin, kmax)
    bt = torch.randperm(B, device=dev)[:P].to(torch.int32).unsqueeze(0)
    seq = torch.tensor([a.ctx], device=dev, dtype=torch.int32)
    q = torch.randn(1, a.hq, a.d, device=dev, dtype=torch.float16)
    one_k = torch.randn(1, a.hk, a.d, device=dev, dtype=torch.float16)
    one_slot = torch.tensor([int(bt[0, a.ctx // BS]) * BS + a.ctx % BS], device=dev,
                            dtype=torch.int32)
    impl = types.SimpleNamespace(num_heads=a.hq, num_kv_heads=a.hk, route_tokens=2048,
                                 window=1024, _kmin=kmin, _kmax=kmax, _want_stats=False,
                                 stats={})
    impl._arange = lambda n, d_: DKVCodecAttentionImpl._arange(impl, n, d_)
    pl = lambda route: DKVCodecAttentionImpl._page_lists(impl, q, seq, bt, BS, route)  # noqa
    fresh = torch.tensor([B], device=dev)
    res = {
        "bounds reset (2 x index_fill)": timeit(lambda: (kmin.index_fill_(0, fresh, 1.0),
                                                         kmax.index_fill_(0, fresh, 1.0))),
        "fused store (1 token)": timeit(lambda: k8v4_store(one_k, one_k, one_slot, pages,
                                                           kmin, kmax)),
        "page lists, all": timeit(lambda: pl(False)),
        "page lists, routed": timeit(lambda: pl(True)),
    }
    plist, lg = pl(False)
    res["fused decode, all pages"] = timeit(lambda: k8v4_decode(q, pages, plist, lg, seq,
                                                                a.d ** -0.5))
    plist_r, lg_r = pl(True)
    res["fused decode, routed pages"] = timeit(lambda: k8v4_decode(q, pages, plist_r, lg_r,
                                                                   seq, a.d ** -0.5))
    print(f"ctx {a.ctx}, Hq {a.hq}, Hk {a.hk}, D {a.d}, {P} pages; "
          f"routed pages kept: {int((plist_r >= 0).sum())}")
    for n, us in res.items():
        print(f"  {n:<32} {us:8.1f} us")


if __name__ == "__main__":
    main()
