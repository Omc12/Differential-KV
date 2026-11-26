"""Slot codecs (no vLLM needed): round trip, size, and a DKV codec carried over."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                  # dkv_vllm
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ACTIVE_RUNTIME"))  # runtime.*

from dkv_vllm.slot_codecs import K8V4, make_slot_codec  # noqa: E402


@pytest.mark.parametrize("D", [64, 128])
def test_k8v4_round_trip_and_size(D):
    torch.manual_seed(0)
    k, v = torch.randn(50, 4, D) * 3, torch.randn(50, 4, D)
    c = K8V4(D)
    s = c.encode(k, v)
    assert s.dtype == torch.uint8 and s.shape == (50, 4, c.slot_bytes)
    assert c.slot_bytes % 16 == 0 and c.slot_bytes < 4 * D          # smaller than fp16 K+V
    k2, v2 = c.decode(s, torch.float32)
    assert (k2 - k).norm() / k.norm() < 0.01                         # 8-bit
    assert (v2 - v).norm() / v.norm() < 0.12                         # 4-bit
    # a row decodes the same alone or in a batch (pure per-token function)
    k1, v1 = c.decode(s[7:8], torch.float32)
    assert torch.equal(k1, k2[7:8]) and torch.equal(v1, v2[7:8])


def test_dkv_function_codec_runs_in_vllm_slots():
    c = make_slot_codec("examples.int8_codec:INT8", 4, 64)
    k, v = torch.randn(9, 4, 64), torch.randn(9, 4, 64)
    k2, v2 = c.decode(c.encode(k, v), torch.float32)
    assert (k2 - k).norm() / k.norm() < 0.01 and (v2 - v).norm() / v.norm() < 0.01


def test_per_channel_kivi_keys_are_refused_with_a_reason():
    with pytest.raises(ValueError, match="no smaller than 16-bit"):
        make_slot_codec("codes", 4, 64)
