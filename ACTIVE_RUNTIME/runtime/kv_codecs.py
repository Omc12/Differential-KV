"""Pluggable KV codecs: how a compressed block stores its K and V rows.

DKV's store keeps, per block, an EXACT anchor token and the other rows as
anchor-relative DELTAS. A codec decides how those deltas are held:

    codes    (default)  keys per-channel and values per-token, both
                        DKV_QUANT_BITS wide -- KIVI's layout
    hybrid              keys per-channel 4-bit; values in the low-rank factor
    lowrank             neither: both halves in the joint low-rank factor plus
                        exact residual rows (DKV_LOWRANK=1)

Whatever half a codec does NOT hold falls back to the low-rank factor, so a
codec only implements the halves it claims. Everything else -- routing, the
recency window, the paged pool, tiered offload, prefill tiling, the decode
arena -- is shared and codec-agnostic.

SWAPPING A CODEC (any of these, nothing else to change)
----------------
    DKV_STORE=codes | hybrid | lowrank | <registered name>
    DKV_STORE=mypackage.mymodule:my_codec          # imported on demand
    DKVHFWrapper(model_id, config={"codec": my_codec, ...})

`my_codec` may be a registered name, a DeltaCodec subclass, a factory, or
ANY object with `encode` and `decode` (a module, a class, a namespace).

WRITING A CODEC: two functions
------------------------------
    def encode(x):      # x [n, T, H, D] float32 anchor-relative deltas
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
        return {"q": (x / s).round().to(torch.int8), "s": s.half()}

    def decode(st):     # -> [n, T, H, D]
        return st["q"].float() * st["s"].float()

    register_codec("int8", encode=encode, decode=decode)   # or the forms above

DKV infers the storage layout from one encode call, and does the rest:
allocation and growth, tiered offload, routing, prefill tiling, decode. The
pair codes keys and values; `values=False` codes keys only (values stay in the
low-rank factor), and `encode_values=` / `decode_values=` give values their
own pair. `examples/int8_codec.py` is this codec as a module.

For control the function form does not give (a cheap single-row key decode
for the router, data-dependent layouts), subclass `DeltaCodec` instead.

Contract (checked by tests/test_kv_codecs.py for every registered codec):
  * deltas arrive as [N, S, H, D] (S <= max_seq_len rows actually written);
    decode returns [n, max_seq_len, H, D]; rows past a slot's length are
    never read, so their content is free.
  * `encode` must fully overwrite the slots it is given (slots are reused).
  * decode is a pure function of the gathered state rows, so the pool can
    slice them per tile, move them between tiers, or batch them.
"""
from __future__ import annotations

import importlib
import math
from typing import Callable, Dict, Optional, Tuple

import torch

Spec = Dict[str, Tuple[Tuple[int, ...], torch.dtype]]


class DeltaCodec:
    """Base class. A codec with neither flag set stores nothing (low rank)."""

    name = "base"
    holds_keys = False
    holds_values = False
    # Nominal widths, for byte accounting and logs. Built-ins set the truth.
    key_bits = 16
    value_bits = 16

    def __init__(self, max_seq_len: int, num_kv_heads: int, head_dim: int,
                 device, dtype=torch.float16):
        self.T, self.H, self.D = int(max_seq_len), int(num_kv_heads), int(head_dim)
        self.device, self.dtype = device, dtype

    # ── geometry ────────────────────────────────────────────────────────────
    def supports(self) -> Optional[str]:
        """None if this geometry is supported, else the reason it is not."""
        return None

    def state_spec(self) -> Spec:
        """{name: (per-slot shape, dtype)}; the pool prepends the slot axis."""
        return {}

    def slot_bytes(self) -> int:
        return sum(math.prod(shape) * torch.empty((), dtype=dt).element_size()
                   for shape, dt in self.state_spec().values())

    # ── data ────────────────────────────────────────────────────────────────
    def encode(self, state: Dict[str, torch.Tensor], pidx: torch.Tensor,
               key_deltas: Optional[torch.Tensor], value_deltas: Optional[torch.Tensor],
               write_seq: int) -> None:
        """Write [N, S, H, D] anchor-relative deltas into slots `pidx`."""

    def decode_keys(self, rows: Dict[str, torch.Tensor], dtype) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not hold keys")

    def decode_values(self, rows: Dict[str, torch.Tensor], dtype) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not hold values")

    def decode_key_rows(self, rows: Dict[str, torch.Tensor], pos: torch.Tensor,
                        dtype) -> torch.Tensor:
        """Keys at row positions `pos` [n, R] (>= 0) -> [n, R, H, D]. The router
        reads these; override when a codec can decode single rows cheaply."""
        k = self.decode_keys(rows, dtype)
        idx = pos.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.D)
        return torch.gather(k, 1, idx)

    def describe(self) -> str:
        halves = [h for h, on in (("K", self.holds_keys), ("V", self.holds_values)) if on]
        return f"{self.name} ({'+'.join(halves) or 'nothing'} in codes)"


class LowRankCodec(DeltaCodec):
    """Holds nothing: both halves live in the joint low-rank factor."""
    name = "lowrank"


class _KeyCodesMixin:
    """Per-channel key codes, groups of `kq_group` tokens (KIVI's key layout)."""

    def _key_spec(self) -> Spec:
        G = (self.T + self.kq_group - 1) // self.kq_group
        return {"kq_codes": ((self.T, self.H, self.D * self.key_bits // 8), torch.uint8),
                "kq_scale": ((G, self.H, self.D), torch.float16),
                "kq_zero": ((G, self.H, self.D), torch.float16)}

    def _encode_keys(self, state, pidx, key_deltas, write_seq):
        from runtime.native_block_pool import quantize_keys_per_channel
        codes, scale, zero = quantize_keys_per_channel(
            key_deltas[:, :write_seq], self.key_bits, self.kq_group)
        state["kq_codes"][pidx] = 0
        state["kq_codes"][pidx, :write_seq] = codes
        g = scale.shape[1]
        state["kq_scale"][pidx] = 0.0
        state["kq_zero"][pidx] = 0.0
        state["kq_scale"][pidx, :g] = scale
        state["kq_zero"][pidx, :g] = zero

    def decode_keys(self, rows, dtype):
        from runtime.native_block_pool import dequantize_keys_per_channel
        return dequantize_keys_per_channel(rows["kq_codes"], rows["kq_scale"],
                                           rows["kq_zero"], self.key_bits,
                                           self.kq_group, dtype)

    def decode_key_rows(self, rows, pos, dtype):
        # Only the gathered rows are dequantized (the router's residual keys).
        from runtime.native_block_pool import _unpack_codes
        p = pos.long()
        n = torch.arange(p.shape[0], device=p.device).unsqueeze(1)
        codes = _unpack_codes(rows["kq_codes"][n, p], self.key_bits)       # [n, R, H, D]
        g = p // self.kq_group
        vals = codes.float() * rows["kq_scale"][n, g].float() + rows["kq_zero"][n, g].float()
        return vals.to(dtype)


class HybridCodec(_KeyCodesMixin, DeltaCodec):
    """Keys as per-channel codes; values stay in the low-rank factor."""
    name = "hybrid"
    holds_keys = True

    def __init__(self, *a, key_bits: int = 4, kq_group: int = 32, **kw):
        super().__init__(*a, **kw)
        self.key_bits, self.kq_group = int(key_bits), int(kq_group)

    def state_spec(self):
        return self._key_spec()

    def encode(self, state, pidx, key_deltas, value_deltas, write_seq):
        if key_deltas is None:
            raise ValueError("hybrid codec: the compressor passed no key deltas")
        self._encode_keys(state, pidx, key_deltas, write_seq)


class KiviCodec(_KeyCodesMixin, DeltaCodec):
    """KIVI's layout: keys per-channel (groups of 32 tokens), values per-token
    (groups of 32 channels), both `bits` wide. Holds the whole block."""
    name = "codes"
    holds_keys = True
    holds_values = True

    def __init__(self, *a, bits: int = 4, kq_group: int = 32, vq_group: int = 32,
                 key_bits: Optional[int] = None, value_bits: Optional[int] = None, **kw):
        super().__init__(*a, **kw)
        self.key_bits = int(key_bits or bits)
        self.value_bits = int(value_bits or bits)
        self.kq_group, self.vq_group = int(kq_group), int(vq_group)

    def supports(self):
        if self.D % self.vq_group:
            return (f"head_dim {self.D} is not a multiple of the value group "
                    f"{self.vq_group}")
        return None

    def state_spec(self):
        s = self._key_spec()
        s.update({
            "vq_codes": ((self.T, self.H, self.D * self.value_bits // 8), torch.uint8),
            "vq_scale": ((self.T, self.H, self.D // self.vq_group), torch.float16),
            "vq_zero": ((self.T, self.H, self.D // self.vq_group), torch.float16)})
        return s

    def encode(self, state, pidx, key_deltas, value_deltas, write_seq):
        # Either half may come alone (the pool's legacy per-half writers); the
        # pool's batched write checks that both arrive.
        if key_deltas is not None:
            self._encode_keys(state, pidx, key_deltas, write_seq)
        if value_deltas is None:
            return
        from runtime.native_block_pool import quantize_values_per_token
        codes, scale, zero = quantize_values_per_token(
            value_deltas[:, :write_seq], self.value_bits, self.vq_group)
        state["vq_codes"][pidx] = 0
        state["vq_scale"][pidx] = 0.0
        state["vq_zero"][pidx] = 0.0
        state["vq_codes"][pidx, :write_seq] = codes
        state["vq_scale"][pidx, :write_seq] = scale
        state["vq_zero"][pidx, :write_seq] = zero

    def decode_values(self, rows, dtype):
        from runtime.native_block_pool import dequantize_values_per_token
        return dequantize_values_per_token(rows["vq_codes"], rows["vq_scale"],
                                           rows["vq_zero"], self.value_bits,
                                           self.vq_group, dtype)


class FunctionCodec(DeltaCodec):
    """A codec from two plain functions -- the easiest way to plug one in.

        encode(x)    x: [n, T, H, D] float32 anchor-relative deltas
                     -> {name: tensor [n, ...]}   (any dtypes / trailing shapes)
        decode(st)   st: that dict, for any n blocks
                     -> [n, T, H, D]

    The storage layout is inferred by calling encode once on a zero block, so
    encode's output shapes must depend only on the input shape. By default the
    same pair codes both keys and values; pass keys=False to code values only
    (not allowed: the factor then has nowhere to keep the keys) or
    values=False for keys only (values then stay in the low-rank factor), or
    give values their own pair with encode_values= / decode_values=.
    """

    def __init__(self, *a, encode, decode, keys: bool = True, values: bool = True,
                 encode_values=None, decode_values=None, name: str = "custom",
                 key_bits: int = 16, value_bits: int = 16, **kw):
        super().__init__(*a, **kw)
        self.name = name
        self.holds_keys, self.holds_values = bool(keys), bool(values)
        self.key_bits, self.value_bits = int(key_bits), int(value_bits)
        self._enc = {"k": encode, "v": encode_values or encode}
        self._dec = {"k": decode, "v": decode_values or decode}
        self._spec: Spec = {}
        for half, on in (("k", self.holds_keys), ("v", self.holds_values)):
            if not on:
                continue
            probe = self._enc[half](torch.zeros(1, self.T, self.H, self.D))
            if not isinstance(probe, dict) or not probe:
                raise TypeError(f"codec {name!r}: encode must return a non-empty "
                                f"dict of tensors, got {type(probe).__name__}")
            for n, t in probe.items():
                if not torch.is_tensor(t) or t.dim() < 1 or t.shape[0] != 1:
                    raise TypeError(f"codec {name!r}: encode output {n!r} must be a "
                                    f"tensor with the block axis first")
                self._spec[f"{half}_{n}"] = (tuple(t.shape[1:]), t.dtype)

    def state_spec(self):
        return dict(self._spec)

    def _encode_half(self, half, state, pidx, x, write_seq):
        full = torch.zeros(x.shape[0], self.T, self.H, self.D, device=x.device)
        full[:, :write_seq] = x[:, :write_seq].float()
        for n, t in self._enc[half](full).items():
            dst = state[f"{half}_{n}"]
            dst[pidx] = t.to(device=dst.device, dtype=dst.dtype)

    def encode(self, state, pidx, key_deltas, value_deltas, write_seq):
        if self.holds_keys and key_deltas is not None:
            self._encode_half("k", state, pidx, key_deltas, write_seq)
        if self.holds_values and value_deltas is not None:
            self._encode_half("v", state, pidx, value_deltas, write_seq)

    def _decode_half(self, half, rows, dtype):
        p = half + "_"
        return self._dec[half]({k[len(p):]: v for k, v in rows.items()
                                if k.startswith(p)}).to(dtype)

    def decode_keys(self, rows, dtype):
        return self._decode_half("k", rows, dtype)

    def decode_values(self, rows, dtype):
        return self._decode_half("v", rows, dtype)


# ── registry ────────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, Callable[..., DeltaCodec]] = {}
_BUILTIN = frozenset({"lowrank", "codes", "hybrid"})


def _as_factory(obj, name: str) -> Callable[..., DeltaCodec]:
    """Anything a user might hand over -> factory(T, H, D, device, dtype=...).

    Accepted: a DeltaCodec subclass; a factory function returning a codec; or
    any object with `encode` and `decode` attributes (a module, a class, a
    namespace) -- wrapped as a FunctionCodec, reading its optional `keys`,
    `values`, `encode_values`, `decode_values` attributes too."""
    if isinstance(obj, type) and issubclass(obj, DeltaCodec):
        return obj
    if callable(getattr(obj, "encode", None)) and callable(getattr(obj, "decode", None)):
        extra = {k: getattr(obj, k) for k in ("keys", "values", "encode_values",
                                              "decode_values", "key_bits", "value_bits")
                 if getattr(obj, k, None) is not None}

        def _f(*a, **kw):
            return FunctionCodec(*a, encode=obj.encode, decode=obj.decode,
                                 name=name, **extra, **kw)
        return _f
    if callable(obj):
        return obj
    raise TypeError(f"codec {name!r}: expected a DeltaCodec subclass, a factory, "
                    f"or an object with encode/decode; got {type(obj).__name__}")


def register_codec(name: str, factory=None, *, encode=None, decode=None,
                   **options) -> str:
    """Make a codec selectable as DKV_STORE=<name> (or config={"codec": name}).

        register_codec("mycodec", MyCodec)                    # a DeltaCodec subclass
        register_codec("int8", encode=enc, decode=dec)        # two functions
        register_codec("int8k", encode=enc, decode=dec, values=False)   # keys only

    Re-registering a name replaces it, except the built-ins (lowrank, codes,
    hybrid), which keep their meaning. Returns the name."""
    key = str(name).strip().lower()
    if key in _BUILTIN and key in _REGISTRY:
        raise ValueError(f"{key!r} is a built-in codec name; register yours under "
                         f"another name")
    if encode is not None or decode is not None:
        if encode is None or decode is None:
            raise TypeError("register_codec: give both encode= and decode=")

        def factory(*a, _e=encode, _d=decode, _o=options, **kw):
            return FunctionCodec(*a, encode=_e, decode=_d, name=key, **_o, **kw)
    elif factory is None:
        raise TypeError("register_codec: give a factory/class, or encode= and decode=")
    else:
        factory = _as_factory(factory, key)
    _REGISTRY[key] = factory
    return key


def registered_codecs():
    return sorted(_REGISTRY)


def _builtin_factories():
    import os

    def _bits(var, default):
        try:
            b = int(os.environ.get(var, default) or default)
        except ValueError:
            b = default
        return b if b in (2, 4, 8) else default

    def _codes(*a, **kw):
        return KiviCodec(*a, bits=_bits("DKV_QUANT_BITS", 4), **kw)

    def _hybrid(*a, **kw):
        return HybridCodec(*a, key_bits=4, **kw)

    return {"lowrank": LowRankCodec, "codes": _codes, "hybrid": _hybrid}


for _n, _f in _builtin_factories().items():
    register_codec(_n, _f)


def resolve_factory(name: str) -> Callable[..., DeltaCodec]:
    """A registered name, or `package.module:Attr` imported on demand."""
    key = str(name).strip()
    f = _REGISTRY.get(key.lower())
    if f is not None:
        return f
    if ":" in key:
        mod, attr = key.split(":", 1)
        register_codec(key, getattr(importlib.import_module(mod), attr))
        return _REGISTRY[key.lower()]
    raise KeyError(f"unknown KV codec {name!r}; registered: {registered_codecs()} "
                   f"(or pass package.module:Class)")


def is_registered(name: str) -> bool:
    key = str(name).strip()
    return key.lower() in _REGISTRY or ":" in key
