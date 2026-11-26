"""A lossless codec: the instrument for "is it the store or the machinery?".

Key deltas are held in float32 (the difference of two bf16/fp16 numbers is
exact there) and values in the model's 16-bit dtype (they arrive in it), so the
store loses nothing. Anything DKV still differs from dense by, run with this
codec, comes from the rest of the pipeline -- routing, the recency window,
prefill, the decode kernels -- not from compression.

    DKV_STORE=examples.lossless_codec:LOSSLESS

Four times the bytes of the default store; a diagnostic, not a store to ship.
"""
from types import SimpleNamespace

import torch


def _enc_keys(x):
    return {"x": x.float()}


def _enc_values(x):
    return {"x": x.to(torch.bfloat16)}


def _dec(st):
    return st["x"].float()


LOSSLESS = SimpleNamespace(encode=_enc_keys, decode=_dec,
                           encode_values=_enc_values, decode_values=_dec,
                           key_bits=32, value_bits=16)
