"""Per-token KV codecs for vLLM pages.

vLLM writes the KV cache one token at a time (`do_kv_cache_update` gets the
new tokens and their slots), so a codec used inside vLLM must encode each
(token, KV head) on its own: one fixed-size byte slot per head per token. That
slot is the page's `state_content_bytes`, which is how vLLM sizes the pool.

    SlotCodec.encode(k, v) -> uint8 [N, H, C]     k, v: [N, H, D]
    SlotCodec.decode(slots) -> (k, v)              slots: [M, H, C]

Built in: `k8v4` -- keys 8-bit per token and head (one fp16 scale), values
4-bit per token in groups of 32 channels (fp16 scale and zero per group,
KIVI's value layout). Keys stay at 8 bits because KIVI's 4-bit keys are
per-CHANNEL over 32-token groups, which a per-token write cannot form; vLLM's
own TurboQuant `k8v4` preset makes the same split.

Any DKV codec (runtime/kv_codecs.py) whose state is per token -- the function
form, e.g. examples/int8_codec.py -- also works here: `DeltaCodecSlots` packs
its state tensors into the byte slot, so the same codec runs in both runtimes.
"""
from __future__ import annotations

import math

import torch


def _align(n: int, a: int = 16) -> int:
    return (n + a - 1) // a * a


class SlotCodec:
    name = "base"
    slot_bytes = 0

    def encode(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, slots: torch.Tensor):
        raise NotImplementedError


class K8V4(SlotCodec):
    """[k int8 D | k scale fp16 | v 4-bit D/2 | v scale fp16 D/32 | v zero fp16 D/32]."""

    name = "k8v4"

    def __init__(self, head_dim: int, group: int = 32):
        if head_dim % group or head_dim % 2:
            raise ValueError(f"k8v4: head_dim {head_dim} must be a multiple of {group}")
        self.D, self.G = head_dim, group
        ng = head_dim // group
        self._off = [0, head_dim, head_dim + 2, head_dim + 2 + head_dim // 2,
                     head_dim + 2 + head_dim // 2 + 2 * ng]
        self._end = self._off[-1] + 2 * ng
        self.slot_bytes = _align(self._end)

    def encode(self, k, v):
        N, H, D = k.shape
        out = torch.zeros(N, H, self.slot_bytes, dtype=torch.uint8, device=k.device)
        kf = k.float()
        ks = kf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
        kq = torch.round(kf / ks).clamp(-127, 127).to(torch.int8)
        vg = v.float().view(N, H, D // self.G, self.G)
        lo, hi = vg.amin(-1, keepdim=True), vg.amax(-1, keepdim=True)
        vs = (hi - lo).clamp(min=1e-8) / 15.0
        vq = torch.round((vg - lo) / vs).clamp(0, 15).to(torch.uint8).view(N, H, D)
        vpk = vq[..., 0::2] | (vq[..., 1::2] << 4)
        o = self._off
        out[..., o[0]:o[1]] = kq.view(torch.uint8)
        out[..., o[1]:o[2]] = ks.half().view(torch.uint8)
        out[..., o[2]:o[3]] = vpk
        out[..., o[3]:o[4]] = vs.squeeze(-1).half().contiguous().view(torch.uint8).view(N, H, -1)
        out[..., o[4]:self._end] = lo.squeeze(-1).half().contiguous().view(torch.uint8).view(N, H, -1)
        return out

    def decode(self, slots, dtype=torch.float16):
        M, H, _ = slots.shape
        D, G, o = self.D, self.G, self._off
        s = slots.contiguous()
        kq = s[..., o[0]:o[1]].contiguous().view(torch.int8).float()
        ks = s[..., o[1]:o[2]].contiguous().view(torch.float16).float()
        k = (kq * ks).to(dtype)
        vpk = s[..., o[2]:o[3]]
        vq = torch.stack([vpk & 0xF, vpk >> 4], -1).view(M, H, D).float()
        vs = s[..., o[3]:o[4]].contiguous().view(torch.float16).float()
        vz = s[..., o[4]:self._end].contiguous().view(torch.float16).float()
        v = (vq.view(M, H, D // G, G) * vs.unsqueeze(-1) + vz.unsqueeze(-1)).view(M, H, D)
        return k, v.to(dtype)


class DeltaCodecSlots(SlotCodec):
    """A DKV codec (runtime/kv_codecs.py) whose state is per token, as a slot
    codec: encode each token as a one-row block and pack its state tensors."""

    def __init__(self, delta_codec_factory, num_kv_heads: int, head_dim: int, name: str):
        self.name = name
        c = delta_codec_factory(1, num_kv_heads, head_dim, "cpu", dtype=torch.float16)
        if not (c.holds_keys and c.holds_values):
            raise ValueError(f"codec {name!r}: vLLM pages need a codec holding keys AND values")
        self.codec, self.H, self.D = c, num_kv_heads, head_dim
        self._spec = []          # (name, per-token-per-head shape, dtype, byte offset, nbytes)
        off = 0
        for n, (shape, dt) in c.state_spec().items():
            if len(shape) < 2 or shape[0] != 1 or shape[1] != num_kv_heads:
                raise ValueError(f"codec {name!r}: state {n!r} is not per token and head "
                                 f"(shape {shape}); KIVI-style per-channel keys need "
                                 f"token groups, which a per-token cache write cannot form")
            tail = tuple(shape[2:])
            nb = math.prod(tail) * torch.empty((), dtype=dt).element_size()
            self._spec.append((n, tail, dt, off, nb))
            off += nb
        if off >= 4 * head_dim:
            # e.g. KIVI's per-CHANNEL keys: over a one-token "group" their
            # scale and zero are the token itself, so nothing is saved.
            raise ValueError(f"codec {name!r}: {off} bytes per token and head is no "
                             f"smaller than 16-bit K+V ({4 * head_dim}); a per-token "
                             f"cache write cannot form its groups")
        self.slot_bytes = _align(off)

    def encode(self, k, v):
        N, H, D = k.shape
        st = {n: torch.zeros((N, 1, H) + tail, dtype=dt, device=k.device)
              for n, tail, dt, _o, _b in self._spec}
        idx = torch.arange(N, device=k.device)
        self.codec.encode(st, idx, k.float().unsqueeze(1), v.float().unsqueeze(1), 1)
        out = torch.zeros(N, H, self.slot_bytes, dtype=torch.uint8, device=k.device)
        for n, _t, _dt, o, nb in self._spec:
            out[..., o:o + nb] = st[n][:, 0].contiguous().view(torch.uint8).reshape(N, H, nb)
        return out

    def decode(self, slots, dtype=torch.float16):
        M, H, _ = slots.shape
        s = slots.contiguous()
        rows = {n: s[..., o:o + nb].contiguous().view(dt).reshape((M, 1, H) + tail)
                for n, tail, dt, o, nb in self._spec}
        k = self.codec.decode_keys(rows, dtype)[:, 0]
        v = self.codec.decode_values(rows, dtype)[:, 0]
        return k, v


def make_slot_codec(name: str, num_kv_heads: int, head_dim: int) -> SlotCodec:
    """`k8v4`, or any DKV codec name / package.module:Attr (per-token only)."""
    if name == "k8v4":
        return K8V4(head_dim)
    try:
        from runtime.kv_codecs import resolve_factory
    except ImportError as e:                                     # noqa: BLE001
        raise ImportError("DKV codecs other than k8v4 need DKV's ACTIVE_RUNTIME on "
                          "PYTHONPATH (runtime.kv_codecs)") from e
    return DeltaCodecSlots(resolve_factory(name), num_kv_heads, head_dim, name)
