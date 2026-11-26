"""The block router combines heads by per-head softmax votes, not raw logits.

A head whose logits run on a much larger scale (Qwen2.5-7B: k_proj bias to
414, one head's logits spanning -4493..643) must not pick every routed block.
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import native_core.srl.query_router as QR


def _case():
    N, D = 40, 4
    anchors = torch.zeros(N, 2, D)
    # head 0: large logit scale, its best blocks are 16..39 (none of the answer)
    anchors[:, 0, 0] = torch.arange(N, dtype=torch.float32) / N
    # head 1: ordinary scale, sharp on blocks 0..15
    anchors[:16, 1, 1] = 5.0
    pool = SimpleNamespace(anchors_K=anchors.half(), routing_topk_default=16,
                           observed_block_span=1024, residual_K_values=None,
                           residual_K_positions=None)
    Q = torch.zeros(2, D)
    Q[0, 0] = 100.0
    Q[1, 1] = 1.0
    return Q, pool, torch.arange(N, dtype=torch.int32), torch.arange(N) * 1024


def _route(monkeypatch, norm):
    monkeypatch.setenv("DKV_TOPK_BLOCKS", "16")
    monkeypatch.setenv("DKV_EDGE_ROUTING", "0")
    monkeypatch.setattr(QR, "_ROUTE_HEADS", "vote" if norm else "max")
    Q, pool, idx, anc = _case()
    sel = QR.route_blocks_relevance(Q, pool, idx, anc, scale=1.0)
    return set(sel.tolist())


def test_raw_logits_let_the_large_scale_head_choose(monkeypatch):
    assert len(_route(monkeypatch, False) & set(range(16))) == 0


def test_vote_keeps_the_ordinary_heads_blocks(monkeypatch):
    sel = _route(monkeypatch, True)
    assert len(sel) == 16
    assert len(sel & set(range(16))) >= 14
    # the large-scale head still gets its own clear winner
    assert 39 in sel
