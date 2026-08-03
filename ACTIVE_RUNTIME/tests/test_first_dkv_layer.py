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
