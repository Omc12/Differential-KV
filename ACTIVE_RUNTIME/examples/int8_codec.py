"""A complete DKV KV codec in a dozen lines: per-token symmetric int8.

Use it with no change to DKV:

    DKV_STORE=examples.int8_codec:INT8      (run from ACTIVE_RUNTIME/)
    DKVHFWrapper(model_id, config={"codec": "examples.int8_codec:INT8"})

or register it yourself:

    from runtime.kv_codecs import register_codec
    from examples.int8_codec import encode, decode
    register_codec("int8", encode=encode, decode=decode)

A codec sees ANCHOR-RELATIVE deltas, x [n, T, H, D] float32 (T = the block's
rows, H = KV heads, D = head_dim), and returns a dict of block-first tensors;
`decode` gets that dict back for any number of blocks. That is the whole
contract -- DKV infers the storage layout from one `encode` call.
"""
import types

import torch


def encode(x: torch.Tensor) -> dict:
    """One scale per (token, head); 8-bit integers."""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    return {"q": torch.round(x / s).clamp(-127, 127).to(torch.int8), "s": s.half()}


def decode(state: dict) -> torch.Tensor:
    return state["q"].float() * state["s"].float()


# What DKV_STORE=examples.int8_codec:INT8 points at: any object with
# encode/decode works (this module itself would too: examples.int8_codec:...).
INT8 = types.SimpleNamespace(encode=encode, decode=decode, key_bits=8, value_bits=8)
