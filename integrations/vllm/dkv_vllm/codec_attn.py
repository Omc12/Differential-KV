"""DKV attention over codec pages in vLLM (phases 2 and 3).

PHASE 2 -- codec pages. Every token's K and V are written into its page slot
through a per-token codec (slot_codecs.py); the page holds `slot_bytes` per
head per token instead of 2 * head_dim * 2, which is how vLLM sizes the pool
(more tokens per GPU). Attention reads a request's pages, decodes them, and
attends. The query chunk's own tokens use their exact 16-bit K/V (they are in
hand), so a prefill chunk attends itself exactly -- KIVI's convention.

PHASE 3 -- routed decode (opt-in, DKV_VLLM_ROUTE_TOKENS > 0). Each page keeps a per-channel min / max of its keys
(Quest's bound, the same summary DKV's router scores), updated on every cache
write. For a decode step on a long context, pages are scored by the bound
on q.k and only the best `DKV_VLLM_ROUTE_TOKENS` worth of pages are decoded
and attended, plus the first page (attention sink) and the most recent
`DKV_VLLM_WINDOW` tokens, which are always read. Off for short contexts, where
reading everything is cheap.

This is a REFERENCE implementation in PyTorch: correct for prefill, chunked
prefill, decode and mixed batches, measured against stock vLLM by the phase
gates. It loops over requests and is not captured in CUDA graphs; fused kernels
come after the design is proven.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from .slot_codecs import make_slot_codec


def _alloc_note(what, n):
    if os.environ.get("DKV_VLLM_DEBUG") == "1":
        print(f"[dkv_vllm dbg] allocating {what} ({n}); capturing="
              f"{torch.cuda.is_current_stream_capturing()}", flush=True)


def codec_name() -> str:
    return (os.environ.get("DKV_VLLM_CODEC") or "").strip()


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class DKVCodecAttentionImpl(TritonAttentionImpl):
    """Codec pages (+ routed decode when DKV_VLLM_ROUTE_TOKENS > 0)."""

    _announced = False
    _instances: list = []

    def __init__(self, num_heads, head_size, scale, num_kv_heads, *args, **kwargs):
        super().__init__(num_heads, head_size, scale, num_kv_heads, *args, **kwargs)
        unsupported = [n for n, v in (
            ("sliding window", getattr(self, "sliding_window", None) not in (None, (-1, -1))),
            ("logit softcap", bool(getattr(self, "logits_soft_cap", 0) or 0)),
            ("attention sinks", getattr(self, "sinks", None) is not None),
            ("alibi", getattr(self, "alibi_slopes", None) is not None)) if v]
        if unsupported:
            raise NotImplementedError(
                "dkv_vllm codec pages do not support " + ", ".join(unsupported) +
                " yet; unset DKV_VLLM_CODEC to run this model on the delegating backend")
        self.codec = make_slot_codec(codec_name(), num_kv_heads, head_size)
        DKVCodecAttentionImpl._instances.append(self)
        # OFF by default: at 16k-32k on Qwen2.5-1.5B the fused kernel reads
        # every page faster than routing scores them (32k: 107.8 tok/s all
        # pages vs 99.9 routed, which also lost 1 of 6 needles). Opt in for
        # contexts long enough that reading 10-20% of the pages pays.
        self.route_tokens = _int_env("DKV_VLLM_ROUTE_TOKENS", 0)
        self.window = _int_env("DKV_VLLM_WINDOW", 1024)
        self._kmin = self._kmax = None          # per-page key bounds [blocks, H, D]
        # counters: decode_steps, routed_steps, pages_read, pages_total; a GPU
        # tensor, so counting never syncs (and the step stays capturable)
        self._stat_t = None
        self._want_stats = bool(os.environ.get("DKV_VLLM_STATS"))
        if not DKVCodecAttentionImpl._announced:
            DKVCodecAttentionImpl._announced = True
            _path = os.environ.get("DKV_VLLM_STATS")
            if _path:
                import atexit
                import json

                def _dump():
                    tot = {}
                    names = ("decode_steps", "routed_steps", "pages_read", "pages_total")
                    for inst in DKVCodecAttentionImpl._instances:
                        vals = inst.stats_values()
                        for k_, v_ in zip(names, vals):
                            tot[k_] = tot.get(k_, 0) + v_
                    tot["layers"] = len(DKVCodecAttentionImpl._instances)
                    with open(_path, "w") as fh:
                        json.dump(tot, fh)
                atexit.register(_dump)
            print(f"[dkv_vllm] DKV codec pages active: codec={self.codec.name} "
                  f"slot={self.codec.slot_bytes} B/head/token "
                  f"(16-bit: {4 * head_size}), route_tokens={self.route_tokens}, "
                  f"window={self.window}", flush=True)

    def _count(self, device, steps, routed, read, total):
        if not self._want_stats:
            return
        if self._stat_t is None:
            self._stat_t = torch.zeros(4, device=device, dtype=torch.float64)
        # element-wise adds of Python scalars or device tensors: no host-to-
        # device copy, which CUDA-graph capture does not allow
        for i, x in enumerate((steps, routed, read, total)):
            self._stat_t[i] += x

    def stats_values(self):
        return [0, 0, 0, 0] if self._stat_t is None else [int(x) for x in self._stat_t.tolist()]

    # ── the cache as bytes [blocks, H, page_tokens, C] ───────────────────────
    def _pages(self, kv_cache):
        c = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
        if c.dim() != 4 or c.shape[1] != self.num_kv_heads or c.shape[-1] < self.codec.slot_bytes:
            raise RuntimeError(f"dkv_vllm: unexpected KV page tensor {tuple(c.shape)} "
                               f"{kv_cache.dtype}; expected [blocks, {self.num_kv_heads}, "
                               f"tokens, >= {self.codec.slot_bytes}] bytes")
        return c

    def _bounds(self, n_blocks, device):
        """Per-page key min / max, fp32 (the fused store updates them with
        atomics). One spare row past the last page takes masked writes, so
        resetting fresh pages needs no host sync."""
        # GROW-ONLY: CUDA graphs capture the tensor this returns, so it must
        # stay the same object afterwards. vLLM hands the startup capture a
        # larger cache view (8121 blocks) than the run (6217); replacing the
        # tensor on that change left the captured decode routing against
        # bounds nothing wrote (needle 2/6 under graphs, 6/6 eager).
        if self._kmin is not None and self._kmin.shape[0] >= n_blocks + 1:
            return self._kmin, self._kmax
        if self._kmin is not None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("dkv_vllm: page bounds would grow during CUDA-graph capture")
        _alloc_note("page bounds", n_blocks)
        shape = (n_blocks + 1, self.num_kv_heads, self.head_size)
        self._kmin = torch.full(shape, float("inf"), device=device, dtype=torch.float32)
        self._kmax = torch.full(shape, float("-inf"), device=device, dtype=torch.float32)
        return self._kmin, self._kmax

    # ── write ────────────────────────────────────────────────────────────────
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        n = slot_mapping.shape[0]
        if n <= 0:
            return
        pages = self._pages(kv_cache)
        bs, nb = pages.shape[2], pages.shape[0]
        slot = slot_mapping[:n]
        k = key[:n].view(n, self.num_kv_heads, self.head_size)
        v = value[:n].view(n, self.num_kv_heads, self.head_size)
        kmin, kmax = self._bounds(nb, pages.device)
        # A page's first token resets its bounds (pages are reused after free).
        # Padding (-1) and non-first tokens write the spare row instead.
        sl = slot.long()
        fresh = torch.where((sl >= 0) & (sl % bs == 0), sl // bs, nb)
        kmin.index_fill_(0, fresh, float("inf"))
        kmax.index_fill_(0, fresh, float("-inf"))
        if self.codec.name == "k8v4" and k.is_cuda and os.environ.get("DKV_VLLM_FUSED", "1") != "0":
            from .triton_decode import k8v4_store
            k8v4_store(k, v, slot.to(torch.int32).contiguous(), pages, kmin, kmax)
            return
        ok = sl >= 0                                    # reference path
        blk, off = sl // bs, sl % bs
        blk_w = torch.where(ok, blk, nb)
        sel = ok.nonzero().squeeze(1)
        pages[blk[sel], :, off[sel]] = self.codec.encode(k[sel], v[sel])
        idx = blk_w.view(-1, 1, 1).expand_as(k)
        kf = k.float()
        kmin.scatter_reduce_(0, idx, kf, reduce="amin", include_self=True)
        kmax.scatter_reduce_(0, idx, kf, reduce="amax", include_self=True)

    # ── read ─────────────────────────────────────────────────────────────────
    def _route(self, q, n_pages, pages_idx, bs, seq_len):
        """Pages to read for one decode query q [Hq, D]: sink + recent window +
        the top pages by the q.k upper bound over the rest."""
        always = {0}
        first_recent = max(0, (seq_len - self.window) // bs)
        always.update(range(first_recent, n_pages))
        cand = [p for p in range(n_pages) if p not in always]
        k_pages = max(0, self.route_tokens // bs)
        if not cand or len(cand) <= k_pages:
            return list(range(n_pages)), False
        g = self.num_heads // self.num_kv_heads
        kmin = self._kmin[pages_idx[cand]].float()                 # [c, Hk, D]
        kmax = self._kmax[pages_idx[cand]].float()
        qg = q.float().view(self.num_kv_heads, g, -1)              # [Hk, g, D]
        ub = torch.maximum(qg.unsqueeze(0) * kmin.unsqueeze(2),
                           qg.unsqueeze(0) * kmax.unsqueeze(2)).sum(-1)   # [c, Hk, g]
        score = ub.amax(dim=(1, 2))                                  # best head
        top = torch.topk(score, k_pages).indices.tolist()
        keep = sorted(always | {cand[i] for i in top})
        return keep, True

    def _page_lists(self, q, seq_lens, block_table, bs, route):
        """Every decode request's pages to read as padded tensors (physical id,
        logical index; -1 padded), with NO host sync: the padding is the
        block table's own width and `route` was decided on the host."""
        R, P = block_table.shape
        n_pages = (seq_lens + bs - 1) // bs                               # [R]
        ar = self._arange(P, q.device)
        logical = ar.expand(R, P)
        valid = logical < n_pages.unsqueeze(1)
        if not route:
            # every page, already in logical order: nothing to compact
            plist = torch.where(valid, block_table, -1).to(torch.int32)
            lg = torch.where(valid, logical, -1).to(torch.int32)
            n = n_pages.sum()
            self._count(q.device, R, 0, n, n)
            return plist.contiguous(), lg.contiguous()
        k_pages = min(max(1, self.route_tokens // bs), P)
        first_recent = ((seq_lens - self.window).clamp(min=0) // bs).unsqueeze(1)
        always = valid & ((logical == 0) | (logical >= first_recent))
        cand = valid & ~always
        if q.is_cuda:
            # Quest's bound per page, computed in place by a Triton kernel:
            # gathering the bounds as [R, P, H, D] tensors cost ~0.5 GB at
            # vLLM's largest CUDA-graph batch (OOM during capture at 32k).
            from .triton_decode import page_scores
            score = page_scores(q.contiguous(), self._kmin, self._kmax, block_table)
        else:
            phys = block_table.long().clamp(min=0)
            g = self.num_heads // self.num_kv_heads
            qg = q.float().view(R, self.num_kv_heads, g, -1)              # [R, Hk, g, D]
            kmin, kmax = self._kmin[phys], self._kmax[phys]                # [R, P, Hk, D]
            score = (torch.einsum("rhgd,rphd->rphg", qg.clamp(min=0), kmax)
                     + torch.einsum("rhgd,rphd->rphg", qg.clamp(max=0), kmin)).amax(dim=(2, 3))
        score = score.masked_fill(~cand, float("-inf"))
        top = torch.topk(score, k_pages, dim=1).indices
        keep = always | (torch.zeros_like(cand).scatter_(1, top, True) & cand)
        # compact kept pages to the front, in logical order: cumsum + scatter
        dst = torch.where(keep, keep.cumsum(1) - 1, P)                    # P = dump column
        plist = torch.full((R, P + 1), -1, device=q.device, dtype=torch.int32)
        lg = torch.full((R, P + 1), -1, device=q.device, dtype=torch.int32)
        plist.scatter_(1, dst, block_table.to(torch.int32))
        lg.scatter_(1, dst, logical.to(torch.int32))
        self._count(q.device, R, R, keep.sum(), n_pages.sum())
        return plist[:, :P].contiguous(), lg[:, :P].contiguous()

    def _arange(self, n, device):
        a = getattr(self, "_ar_cache", None)
        if a is None or a.shape[0] < n or a.device != device:
            _alloc_note("arange", n)
            a = self._ar_cache = torch.arange(max(n, 1024), device=device)
        return a[:n]

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)                      # profiling run
        pages = self._pages(kv_cache)
        bs = pages.shape[2]
        n_tok = attn_metadata.num_actual_tokens
        # FAST PATH: a decode-only batch on the k8v4 codec -> fused kernel.
        if (attn_metadata.max_query_len == 1 and self.codec.name == "k8v4"
                and query.is_cuda and os.environ.get("DKV_VLLM_FUSED", "1") != "0"):
            from .triton_decode import k8v4_decode
            # Static shapes only (batch size, block-table width): this path is
            # captured in a CUDA graph and replayed with new metadata, so it
            # must not depend on the live token count or decide on the host.
            # Routing is always computed; a short sequence simply keeps every
            # page (its sink + window cover it).
            R = attn_metadata.seq_lens.shape[0]
            q = query[:R].view(R, self.num_heads, self.head_size)
            route = self.route_tokens > 0
            plist, logical = self._page_lists(q, attn_metadata.seq_lens,
                                              attn_metadata.block_table, bs, route)
            o = k8v4_decode(q.contiguous(), pages, plist, logical,
                            attn_metadata.seq_lens.to(torch.int32), self.scale)
            output[:R].view(R, self.num_heads, self.head_size).copy_(o)
            return output
        qsl = attn_metadata.query_start_loc.tolist()
        seq = attn_metadata.seq_lens.tolist()
        bt = attn_metadata.block_table
        g = self.num_heads // self.num_kv_heads
        out = output[:n_tok].view(n_tok, self.num_heads, self.head_size)
        q_all = query[:n_tok].view(n_tok, self.num_heads, self.head_size)
        k_new = key[:n_tok].view(n_tok, self.num_kv_heads, self.head_size)
        v_new = value[:n_tok].view(n_tok, self.num_kv_heads, self.head_size)
        for r in range(len(seq)):
            a, b = qsl[r], qsl[r + 1]
            ql, L = b - a, seq[r]
            if ql <= 0:
                continue
            n_pages = (L + bs - 1) // bs
            pidx = bt[r, :n_pages].long()
            ctx_len = L - ql                            # tokens before this chunk
            routed = False
            if ql == 1 and self.route_tokens > 0:
                self._count(q_all.device, 1, 0, 0, 0)
                keep, routed = self._route(q_all[a], n_pages, pidx, bs, L)
            else:
                keep = list(range(n_pages))
            sel = torch.as_tensor(keep, device=pidx.device)
            kc, vc = self.codec.decode(
                pages[pidx[sel]].permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads,
                                                             pages.shape[-1]),
                dtype=query.dtype)                       # [len(keep)*bs, Hk, D]
            pos = (sel.view(-1, 1) * bs + torch.arange(bs, device=sel.device)).view(-1)
            valid = pos < L                              # the last page's tail is padding
            # the chunk's own tokens: exact 16-bit K/V, not their codes (padding
            # rows past L are not the chunk's: they would index past k_new)
            own = (pos >= ctx_len) & valid
            if bool(own.any()):
                j = (pos[own] - ctx_len).long()
                kc[own] = k_new[a + j]
                vc[own] = v_new[a + j]
            kc, vc, pos = kc[valid], vc[valid], pos[valid]
            if ql == 1:
                self._count(q_all.device, 0, int(routed), len(keep), n_pages)
            # Memory-bounded: SDPA reads the KV heads directly (enable_gqa, no
            # repeated K/V) and queries go in tiles, so the mask and scores
            # stay small. vLLM sizes the KV cache from a profiling run in which
            # this path does not execute, so its temporaries must be modest.
            kh = kc.transpose(0, 1).unsqueeze(0)                    # [1, Hk, S, D]
            vh = vc.transpose(0, 1).unsqueeze(0)
            tile = _int_env("DKV_VLLM_PREFILL_TILE", 512)
            for t0 in range(0, ql, tile):
                t1 = min(ql, t0 + tile)
                qh = q_all[a + t0:a + t1].transpose(0, 1).unsqueeze(0)       # [1, Hq, t, D]
                qpos = torch.arange(ctx_len + t0, ctx_len + t1, device=pos.device)
                mask = pos.view(1, -1) <= qpos.view(-1, 1)                  # causal [t, S]
                o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask,
                                                   scale=self.scale, enable_gqa=True)
                out[a + t0:a + t1] = o[0].transpose(0, 1)
        return output
