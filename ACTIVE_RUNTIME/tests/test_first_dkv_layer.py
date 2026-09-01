"""first_dkv_layer_index must adapt to ANY attention/non-attention interleaving.

Thirteen once-per-token gates in dkv_forward key on "is this the first layer DKV
attends" -- among them finalize_compressed_blocks, which PUBLISHES
background-compressed blocks into the pool. They used to compare against model
layer 0, which is dead on any architecture whose layer 0 is not an attention
layer, and that silently disabled all thirteen on hybrids.

The question these tests answer is the one that matters for the next model, not
this one: does a different interleaving need new code? It must not. The index is
derived from the same predicate the patch loop uses (`hasattr(layer,
"self_attn")`), so the two cannot disagree about which layers DKV sees.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.dkv_attention import first_dkv_layer_index


class _Attn:
    def __init__(self):
        self.self_attn = object()


class _Linear:      # gated-delta-net / mamba-style layer: no KV cache to intercept
    pass


def _layers(pattern):
    """pattern: string of 'A' (attention) and 'L' (linear)."""
    return [_Attn() if c == "A" else _Linear() for c in pattern]


def test_dense_model_first_layer_is_zero():
    # Qwen2.5 / Llama / Mistral: every layer is attention.
    assert first_dkv_layer_index(_layers("A" * 28)) == 0


def test_qwen35_hybrid_every_fourth_layer():
    # Qwen3.5-2B: attention at 3, 7, 11, 15, 19, 23.
    pattern = "".join("A" if i % 4 == 3 else "L" for i in range(24))
    assert first_dkv_layer_index(_layers(pattern)) == 3


def test_front_loaded_linear_layers():
    # A model that runs several linear layers before any attention.
    assert first_dkv_layer_index(_layers("LLLLLAAAA")) == 5


def test_attention_first_then_linear():
    # The opposite interleaving -- attention up front, linear afterwards.
    assert first_dkv_layer_index(_layers("AAALLLLL")) == 0


def test_irregular_interleaving():
    # No period at all. Nothing in the implementation may assume one.
    assert first_dkv_layer_index(_layers("LLALALLLA")) == 2


def test_single_attention_layer_at_the_end():
    assert first_dkv_layer_index(_layers("LLLLLLLA")) == 7


def test_no_attention_layers_does_not_raise():
    # Degenerate: the patch loop wraps nothing, so the value is unused. It must
    # still return an int rather than raising StopIteration during patching.
    assert first_dkv_layer_index(_layers("LLLL")) == 0
    assert first_dkv_layer_index([]) == 0


def test_agrees_with_the_patch_loop_predicate():
    """The index must be a layer the patch loop actually wraps.

    This is the invariant that makes the whole thing model-agnostic: both use
    `hasattr(layer, "self_attn")`, so a layer set the loop skips can never be
    the one the gates fire on.
    """
    for pattern in ("A" * 8, "LLLA", "LALALALA", "LLLLAAAA",
                    "".join("A" if i % 4 == 3 else "L" for i in range(24))):
        layers = _layers(pattern)
        idx = first_dkv_layer_index(layers)
        wrapped = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]
        assert idx in wrapped, f"{pattern}: gates fire on {idx}, wrapped {wrapped}"
        assert idx == min(wrapped), (
            f"{pattern}: once-per-token work must run on the EARLIEST attended "
            f"layer so it happens before the others in the forward pass")


# ── Hybrid attention: sliding vs global, and cross-layer KV sharing ──────────
#
# Gemma 4 broke every assumption the tests above encode. Its layers ALL have
# self_attn, so "does this layer have attention" no longer decides what DKV can
# compress: sliding layers carry a different head_dim from the global ones (256
# vs 512), and a layer whose full-length K/V another layer consumes must stay
# native or the consumer raises KeyError.

from runtime.dkv_attention import (dkv_layer_indices, is_dkv_attention_layer,
                                   is_sliding_attention_layer, attn_owns_kv)


class _GAttn:
    """A Gemma-4-shaped attention module."""
    def __init__(self, layer_type="full_attention", k_proj="present",
                 is_kv_shared_layer=False, store_full_length_kv=False):
        self.layer_type = layer_type
        self.is_sliding = (layer_type == "sliding_attention")
        self.is_kv_shared_layer = is_kv_shared_layer
        self.store_full_length_kv = store_full_length_kv
        if k_proj != "absent":          # "absent" models a fused qkv_proj
            self.k_proj = None if k_proj is None else object()


class _GLayer:
    def __init__(self, **kw):
        self.self_attn = _GAttn(**kw)


def test_sliding_layers_are_never_compressed():
    # Their KV is already bounded by the window, and their head_dim differs
    # from the global layers' -- compressing them is what raised
    # "shape '[1, 1025, 1, 256]' is invalid for input of size 524800".
    layers = [_GLayer(layer_type="sliding_attention") for _ in range(4)]
    layers.insert(2, _GLayer(layer_type="full_attention"))
    assert dkv_layer_indices(layers) == [2]
    assert first_dkv_layer_index(layers) == 2
    assert not is_dkv_attention_layer(layers[0])
    assert is_sliding_attention_layer(layers[0])


def test_kv_sharing_producer_is_excluded_when_consumers_exist():
    # gemma-4-e2b-it: layer 14 publishes full-length K/V that layers 19..34
    # read out of shared_kv_states. Compressing 14 gives the consumers a
    # KeyError, so DKV must take 4 and 9 only.
    layers = [_GLayer(layer_type="sliding_attention") for _ in range(20)]
    layers[4] = _GLayer(layer_type="full_attention")
    layers[9] = _GLayer(layer_type="full_attention")
    layers[14] = _GLayer(layer_type="full_attention", store_full_length_kv=True)
    layers[19] = _GLayer(layer_type="full_attention", k_proj=None,
                         is_kv_shared_layer=True)
    assert dkv_layer_indices(layers) == [4, 9]


def test_producer_with_no_consumers_is_still_compressed():
    # gemma-4-12B-it sets store_full_length_kv on its last global layer but has
    # NO kv-shared layers, so all eight global layers stay compressible. The
    # exclusion must be conditional on a consumer actually existing.
    layers = [_GLayer(layer_type="sliding_attention") for _ in range(6)]
    layers[2] = _GLayer(layer_type="full_attention")
    layers[5] = _GLayer(layer_type="full_attention", store_full_length_kv=True)
    assert dkv_layer_indices(layers) == [2, 5]


def test_fused_qkv_layers_are_not_mistaken_for_kv_sharing():
    # A model with a packed qkv_proj has NO k_proj attribute. Treating that as
    # "shares KV" would silently compress nothing on such a model -- no error,
    # just DKV disabled. Only an explicit k_proj=None means sharing.
    fused = [_GLayer(layer_type="full_attention", k_proj="absent") for _ in range(4)]
    assert dkv_layer_indices(fused) == [0, 1, 2, 3]
    assert attn_owns_kv(fused[0])

    shared = _GLayer(layer_type="full_attention", k_proj=None)
    assert not attn_owns_kv(shared)


def test_plain_dense_model_is_unaffected_by_the_hybrid_rules():
    # Granite/Llama/Qwen: no layer_type, no sliding, no sharing -- every layer.
    assert dkv_layer_indices(_layers("A" * 12)) == list(range(12))
