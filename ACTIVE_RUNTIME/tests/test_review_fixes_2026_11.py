"""Unit checks for the 2026-11-24 review fixes (no model, no GPU needed).

  * residual rows are question-free by default (residual_query_enabled)
  * the elastic window compresses by token position, not free VRAM
  * router-owned key boxes: table bookkeeping and the upper-bound score
"""
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from native_core.compression.residual_capture import residual_query_enabled  # noqa: E402
from native_core.srl.key_box import KeyBoxTable, box_relevance               # noqa: E402
from native_core import streaming_sparse_ingest as ssi                       # noqa: E402


# ── residual ordering is question-free by default ────────────────────────────

@pytest.mark.parametrize("env,expected", [
    ({}, False),
    ({"DKV_RESIDUAL_QUERY": "1"}, True),
    ({"DKV_RESIDUAL_QUERY": "0"}, False),
    ({"DKV_RESIDUAL_NO_QUERY": "1"}, False),          # legacy spelling, off
    ({"DKV_RESIDUAL_NO_QUERY": "0"}, True),           # legacy spelling, on
    ({"DKV_RESIDUAL_QUERY": "0", "DKV_RESIDUAL_NO_QUERY": "0"}, False),  # new wins
])
def test_residual_query_default_off(monkeypatch, env, expected):
    for k in ("DKV_RESIDUAL_QUERY", "DKV_RESIDUAL_NO_QUERY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert residual_query_enabled() is expected


# ── elastic window by tokens ─────────────────────────────────────────────────

def _blk(anchor, n):
    return types.SimpleNamespace(anchor_idx=anchor, token_count=lambda n=n: n)


def test_elastic_due_counts_only_blocks_older_than_window(monkeypatch):
    monkeypatch.setenv("DKV_STREAM_EXACT_TOKENS", "4096")
    blocks = [_blk(i * 1024, 1024) for i in range(10)]   # ends 1024 .. 10240
    # seq 10240 -> cutoff 6144: blocks ending at 1024..6144 are due (6 of them)
    assert ssi._elastic_due(blocks, 10240) == 6
    assert ssi._elastic_due(blocks, 4096) == 0
    monkeypatch.setenv("DKV_STREAM_EXACT_TOKENS", "0")
    assert ssi._elastic_due(blocks, 10240) == 10


def test_elastic_hold_is_device_independent(monkeypatch):
    """Same blocks, same length -> same decision whatever _elastic_room says."""
    monkeypatch.setenv("DKV_STREAM_ELASTIC", "1")
    monkeypatch.setenv("DKV_STREAM_EXACT_TOKENS", "4096")
    monkeypatch.setenv("DKV_STREAM_ELASTIC_BATCH", "4")
    monkeypatch.delenv("DKV_STREAM_ELASTIC_POLICY", raising=False)

    class _Mgr:
        def update_metadata_state(self, *a):
            pass

    monkeypatch.setattr(ssi, "_elastic_room", lambda: True)   # a roomy card

    def run(seq):
        blocks = [types.SimpleNamespace(anchor_idx=i * 1024, state="SUBMITTED",
                                        token_count=lambda: 1024) for i in range(10)]
        out = ssi.StreamingSparseIngestManager._elastic_hold(_Mgr(), "s", 0, blocks, seq)
        return [b.anchor_idx for b in out]

    # The memory policy would compress NOTHING here (room); tokens compresses
    # what is older than the window, capped at the batch.
    assert run(10240) == [0, 1024, 2048, 3072]                # 6 due, batch 4
    assert run(5120) == [0]                                   # cutoff 1024
    monkeypatch.setenv("DKV_STREAM_ELASTIC_POLICY", "memory")
    assert run(10240) == []                                   # legacy: room -> hold


# ── key boxes ────────────────────────────────────────────────────────────────

def test_key_box_table_set_invalidate_gather():
    t = KeyBoxTable(torch.device("cpu"))
    assert t.gather(torch.tensor([0, 1])) is None             # nothing recorded
    mn = torch.zeros(2, 2, 4)
    mx = torch.ones(2, 2, 4)
    t.set([3, 70], mn, mx)                                    # grows past 64
    g_mn, g_mx, valid = t.gather(torch.tensor([3, 70, 5, 1000]))
    assert valid.tolist() == [True, True, False, False]       # 1000 out of range
    t.invalidate(3)
    t.invalidate(torch.tensor([70]))
    assert t.gather(torch.tensor([3, 70]))[2].tolist() == [False, False]


def test_box_score_is_an_upper_bound_and_finds_the_needle():
    torch.manual_seed(0)
    N, H_kv, gpk, T, D = 12, 2, 4, 64, 32
    keys = torch.randn(N, H_kv, T, D) * 0.5
    q = torch.randn(H_kv * gpk, 1, D)
    # plant a key in block 7 aligned with query head 5 (kv head 1)
    keys[7, 1, 10] = q[5, 0] * 3.0
    mn, mx = keys.amin(dim=2), keys.amax(dim=2)
    rel = box_relevance(q, mn, mx, torch.ones(N, dtype=torch.bool), scale=1.0)

    qg = q.reshape(H_kv, gpk, D)
    true = torch.einsum("kgd,nktd->nkgt", qg, keys).amax(dim=(1, 2, 3))   # [N]
    assert torch.all(rel >= true - 1e-4)                      # never under-scores
    assert int(rel.argmax()) == 7


def test_box_score_routes_unboxed_slots_first():
    mn = torch.zeros(3, 1, 4)
    mx = torch.ones(3, 1, 4)
    valid = torch.tensor([True, False, True])
    rel = box_relevance(torch.ones(2, 1, 4), mn, mx, valid, scale=1.0)
    assert torch.isinf(rel[1]) and torch.isfinite(rel[0])
