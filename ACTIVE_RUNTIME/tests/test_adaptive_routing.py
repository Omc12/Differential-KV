"""Adaptive block count (DKV_ROUTE_ADAPTIVE, 2026-11-24): estimate + selection."""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from native_core.srl.key_box import (  # noqa: E402
    KeyBoxTable, estimate_block_lse, select_adaptive)


def _blocks(N=12, n=64, H=2, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(N, n, H, D, generator=g)
    return K


def _stats(K):
    return (K.amin(1), K.amax(1), K.mean(1), K.var(1, unbiased=False),
            torch.full((K.shape[0],), math.log(K.shape[1])))


def test_estimate_respects_the_two_bounds_and_tracks_true_lse():
    K = _blocks()
    K[3] *= 3.0                                   # one high-variance block
    mn, mx, mu, var, logn = _stats(K)
    Q = torch.randn(4, 1, 16)                     # 4 query heads, 2 kv heads
    s = 0.25
    est = estimate_block_lse(Q, mn, mx, mu, var, logn, s)        # [4, N]
    kv = torch.arange(4) // 2
    true = torch.stack([torch.logsumexp(s * (K[:, :, kv[h]] @ Q[h, 0]), dim=1)
                        for h in range(4)])                     # [4, N]
    lo = torch.stack([logn + s * (mu[:, kv[h]] @ Q[h, 0]) for h in range(4)])
    assert (est >= lo - 1e-4).all()
    # Upper bound: log n + s * box bound >= true lse
    for h in range(4):
        q = Q[h, 0]
        ub = ((mx[:, kv[h]] + mn[:, kv[h]]) / 2 @ q + ((mx[:, kv[h]] - mn[:, kv[h]]) / 2) @ q.abs())
        assert (est[h] <= logn + s * ub + 1e-4).all()
        assert (true[h] <= logn + s * ub + 1e-4).all()
    # A useful estimate: ranks blocks like the truth
    r = torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1]
    assert r > 0.8


def test_floor_is_exactly_the_fixed_box_set_when_it_covers_p(monkeypatch):
    N = 40
    box = torch.arange(N, 0, -1).float()          # box ranks 0,1,2,... first
    est = torch.full((3, N), -5.0)
    est[:, :5] = 10.0                             # the mass sits inside box top-8
    ok = torch.ones(N, dtype=torch.bool)
    monkeypatch.setenv("DKV_ROUTE_ADAPTIVE_P", "0.9")
    sel = select_adaptive(est, box, ok, k_floor=8)
    assert sorted(sel.tolist()) == list(range(8))  # nothing added
    assert sel.tolist() == list(range(8))          # best first, box order


def test_grows_by_estimate_outside_the_floor_and_caps(monkeypatch):
    N = 40
    box = torch.arange(N, 0, -1).float()
    est = torch.full((2, N), -5.0)
    est[:, 30:34] = 10.0                          # mass the box ranks last
    ok = torch.ones(N, dtype=torch.bool)
    monkeypatch.setenv("DKV_ROUTE_ADAPTIVE_P", "0.9")
    monkeypatch.setenv("DKV_ROUTE_ADAPTIVE_KMAX", "64")
    sel = sel_l = select_adaptive(est, box, ok, k_floor=8).tolist()
    assert set(range(8)) <= set(sel) and set(range(30, 34)) <= set(sel)
    assert len(sel_l) == 12
    # Flat mass: coverage would need ~0.9 N; KMAX caps it.
    monkeypatch.setenv("DKV_ROUTE_ADAPTIVE_KMAX", "20")
    sel = select_adaptive(torch.zeros(2, N), box, ok, k_floor=8)
    assert sel.numel() == 20


def test_unrecorded_blocks_always_kept(monkeypatch):
    N = 30
    est = torch.zeros(2, N)
    est[:, 0] = 20.0
    box = torch.zeros(N)
    box[0] = 9.0
    box[23] = float("inf")                        # no box recorded
    ok = torch.ones(N, dtype=torch.bool)
    ok[11] = False                                # no moments recorded
    monkeypatch.setenv("DKV_ROUTE_ADAPTIVE_P", "0.5")
    sel = select_adaptive(est, box, ok, k_floor=2).tolist()
    assert {0, 23, 11} <= set(sel)
    assert len(sel) == len(set(sel))


def test_table_moments_follow_slot_validity():
    t = KeyBoxTable(torch.device("cpu"))
    mn = torch.zeros(2, 2, 4)
    mom = (torch.ones(2, 2, 4), torch.ones(2, 2, 4) * 2, torch.full((2,), 3.0))
    t.set([3, 5], mn, mn + 1, mom)
    mu, var, logn, ok = t.gather_moments(torch.tensor([3, 5, 7]))
    assert ok.tolist() == [True, True, False]
    assert var[0, 0, 0].item() == 2.0 and logn[1].item() == 3.0
    t.invalidate(5)
    assert t.gather_moments(torch.tensor([3, 5]))[3].tolist() == [True, False]
    t.set([3], mn[:1], mn[:1] + 1)                # box without moments
    assert t.gather_moments(torch.tensor([3]))[3].tolist() == [False]
