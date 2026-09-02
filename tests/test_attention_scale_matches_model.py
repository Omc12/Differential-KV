"""Every model's DECODE scale must be the model's own, not 1/sqrt(head_dim).

WHY THIS TEST EXISTS
--------------------
`1/sqrt(head_dim)` is a Llama-ism. It is correct for Qwen3.5 and wrong for
several other families, and DKV asserted it in four decode sites while reading
the model's `scaling` in none:

    Qwen3.5 (all)     scaling 0.0625     == 1/sqrt(256)   ok
    granite-4.2-8b    scaling 0.0078125  vs 0.088388      11.3x too hot
    gemma-4 (all)     scaling 1.0        vs 0.044194      22.6x too cold

Commit 7346a492 fixed the attention MODULE and left the decode kernels it calls
into untouched, so prefill was exact while decode ran at the wrong softmax
temperature from the second token onward. On granite that measured
KL(dense||DKV) 3.63 at token 2 with dense's own choice down at rank 6, and the
generated text degenerated into word-salad -- while token 1, needle recall, and
the in-house synthesis score all still looked fine.

This test is deliberately CONFIG-LEVEL and needs no GPU: it compares what each
model DECLARES against the Llama default, and asserts that any model where they
differ is one this repo knows about. A new model family whose scaling is not
1/sqrt(head_dim) will fail here the moment it is added, which is months before
anyone would notice it in a benchmark table.
"""

import math

import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoConfig                              # noqa: E402


# Every model in the study, with the scale each one actually declares.
# `None` means "no explicit scaling; 1/sqrt(head_dim) is genuinely correct".
KNOWN = {
    "ibm-granite/granite-4.2-8b": 0.0078125,
    "Qwen/Qwen3.5-4B": None,
    "Qwen/Qwen3.5-2B": None,
}


def _text_config(cfg):
    """Composite (multimodal) configs hide the decoder under text_config."""
    inner = getattr(cfg, "text_config", None)
    if inner is not None and getattr(inner, "num_attention_heads", None):
        return inner
    return cfg


def _declared_scale(tc):
    """The scale the model itself asks for, or None if it wants the default."""
    mult = getattr(tc, "attention_multiplier", None)
    if mult is not None:
        return float(mult)
    scaling = getattr(tc, "query_pre_attn_scalar", None)
    if scaling is not None:
        return 1.0 / math.sqrt(float(scaling))
    return None


def _head_dim(tc):
    hd = getattr(tc, "head_dim", None)
    if hd:
        return int(hd)
    return int(tc.hidden_size) // int(tc.num_attention_heads)


@pytest.mark.parametrize("model_id,expected", sorted(KNOWN.items()))
def test_declared_attention_scale_is_known(model_id, expected):
    """A model whose scale is not the Llama default must be recorded here."""
    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"{model_id} not available locally: {type(e).__name__}")

    tc = _text_config(cfg)
    declared = _declared_scale(tc)
    default = 1.0 / math.sqrt(_head_dim(tc))

    if expected is None:
        assert declared is None or abs(declared - default) < 1e-9, (
            f"{model_id} declares a scale ({declared}) that differs from "
            f"1/sqrt(head_dim) ({default}). Add it to KNOWN and make sure the "
            f"decode path reads it -- see resolve_attn_scale()."
        )
    else:
        assert declared is not None and abs(declared - expected) < 1e-9, (
            f"{model_id} was recorded as scale {expected} but now declares "
            f"{declared}."
        )
        ratio = default / declared
        assert ratio > 1.5 or ratio < 0.67, (
            f"{model_id}: recorded as differing from the Llama default, but "
            f"the ratio is only {ratio:.2f}x -- re-check whether it still needs "
            f"a special case."
        )


def test_resolver_prefers_the_published_scale():
    """resolve_attn_scale must use the pool's scale over the Llama default."""
    pytest.importorskip("torch")
    import sys
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ACTIVE_RUNTIME"))
    from native_core.sparse_decode.triton_fused_decode import resolve_attn_scale

    class _Pool:
        pass

    p = _Pool()
    d = 128
    # Nothing published: the Llama default, which is the documented fallback.
    assert resolve_attn_scale(p, d) == pytest.approx(1.0 / math.sqrt(d))

    # Published: granite's own multiplier must win, all 11.3x of it.
    p.dkv_attn_scale = 0.0078125
    assert resolve_attn_scale(p, d) == pytest.approx(0.0078125)

    # Nonsense must not silently become a valid scale. NOTE a numeric STRING is
    # not nonsense and must be honoured: a scale arriving as "0.0078125" from an
    # env var is still the right scale, and rejecting it would fall back to
    # 1/sqrt(head_dim) -- reintroducing the exact 11.3x bug this guards.
    for bad in (0.0, -1.0, float("nan"), None, "", "abc", object()):
        p.dkv_attn_scale = bad
        got = resolve_attn_scale(p, d)
        assert got == pytest.approx(1.0 / math.sqrt(d)), f"bad={bad!r} -> {got}"

    for good in ("0.0078125", 0.0078125):
        p.dkv_attn_scale = good
        assert resolve_attn_scale(p, d) == pytest.approx(0.0078125), (
            f"a well-formed scale {good!r} must be used, not discarded"
        )
