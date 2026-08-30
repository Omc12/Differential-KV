"""Top-p block selection: the guard rails, and that it stays OFF by default.

WHY THIS EXISTS
---------------
`select_by_mass` is the adaptive alternative to a fixed top-K -- take blocks
until they cover a share of the softmaxed relevance mass. It is opt-in and
measured to be no better than fixed K on quality (see the function's docstring),
so the single most important property to pin is that it does not run unless
someone asks for it. The rest is the clamping, which is what makes an adaptive
rule safe: a rule with no floor drops to one block on an over-confident score,
and nothing downstream can recover the context it skipped.

These are pure-tensor tests -- no GPU, no model, no pool.
"""
import importlib
import math
import os
import sys

import pytest

# Same preamble as the other tests in this directory (test_basis_group.py:15).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

QR = importlib.import_module("native_core.srl.query_router")


def _rel(*vals):
    return torch.tensor(vals, dtype=torch.float32)


def test_disabled_by_default():
    """The env knobs must default to off. This is the whole safety story: the
    measurement says top-p is not better, so it must not engage silently."""
    assert QR._ROUTE_TOPP == 0.0, \
        f"DKV_ROUTE_TOPP defaulted to {QR._ROUTE_TOPP}, not 0 (off)"
    assert QR._ROUTE_SCORE_MODE == "max", \
        f"DKV_ROUTE_SCORE defaulted to {QR._ROUTE_SCORE_MODE!r}, not 'max'"
    assert os.environ.get("DKV_ROUTE_TOPP") in (None, "", "0")


def test_spiky_relevance_is_held_up_by_the_floor():
    """One dominant block would otherwise select K=1. The floor is the only
    thing standing between an over-confident score and a starved hard query."""
    rel = torch.full((27,), -5.0)
    rel[7] = 10.0
    sel = QR.select_by_mass(rel, topp=0.99, k_min=4, k_floor=1)
    assert len(sel) == 4
    assert sel[0].item() == 7, "the dominant block must still be picked first"


def test_flat_relevance_takes_everything():
    """A diffuse query is the case top-p exists for: it must be able to grow."""
    rel = torch.zeros(27)
    sel = QR.select_by_mass(rel, topp=0.999999, k_min=4, k_floor=1)
    assert len(sel) == 27


def test_k_floor_is_respected():
    """DKV_TOPK_FRAC raises k_eff; top-p must not quietly undo that."""
    rel = torch.linspace(3.0, 0.0, 27)
    sel = QR.select_by_mass(rel, topp=0.5, k_min=4, k_floor=20)
    assert len(sel) == 20


def test_never_exceeds_the_block_count():
    """The ceiling is N -- asking for more blocks than exist would index off
    the end of the pool."""
    rel = torch.zeros(5)
    sel = QR.select_by_mass(rel, topp=0.999999, k_min=64, k_floor=99)
    assert len(sel) == 5
    assert sel.max().item() < 5


def test_indices_are_ordered_by_descending_relevance():
    rel = _rel(0.0, 5.0, 1.0, 4.0, 2.0)
    sel = QR.select_by_mass(rel, topp=0.99, k_min=3, k_floor=1)
    got = [i.item() for i in sel]
    assert got[:3] == [1, 3, 4], got


def test_higher_threshold_never_selects_fewer_blocks():
    """Monotonic in the threshold. A rule that shrank as you asked for more mass
    would be incoherent, and the sweep that tuned the threshold assumes this."""
    rel = torch.linspace(4.0, 0.0, 27)
    sizes = [len(QR.select_by_mass(rel, topp=t, k_min=1, k_floor=1))
             for t in (0.5, 0.75, 0.9, 0.95, 0.99)]
    assert sizes == sorted(sizes), sizes


def test_lse_can_reorder_blocks_but_only_within_log_R():
    """logsumexp re-ranks a diffuse block above a spiky one -- but barely.

    This is the quantitative reason DKV_ROUTE_SCORE=lse did not rescue top-p.
    For R keys, log-sum-exp exceeds the max by AT MOST log(R):

        max <= logsumexp <= max + log(R)

    so it can only reorder blocks whose best logits already sit within ~log(R)
    of each other -- 1.39 nats at R=4, 3.47 at R=32. Any wider gap is untouchable
    no matter how much mass the weaker block holds. Measured consequence: with
    the LSE score the median top-1 block still held 0.577 of the softmaxed mass
    (against 0.673 under max), and a 0.99 threshold still never asked for more
    than 15 of 27 blocks. Fixing the aggregation is not enough to make an
    adaptive rule grow; the score would have to see beyond the top-R keys.
    """
    # A gap SMALLER than log(R): logsumexp flips the order.
    near = torch.tensor([[[[[5.5, 0.0, 0.0, 0.0],        # spiky, best = 5.5
                            [5.0, 5.0, 5.0, 5.0]]]]])    # diffuse, best = 5.0
    mx = near.max(dim=-1).values
    lse = torch.logsumexp(near.float(), dim=-1)
    assert mx[..., 0] > mx[..., 1], "max must prefer the spiky block"
    assert lse[..., 0] < lse[..., 1], "logsumexp must prefer the massive block"

    # A gap WIDER than log(R): logsumexp cannot flip it, however diffuse.
    far = torch.tensor([[[[[9.0, 0.0, 0.0, 0.0],
                           [5.0, 5.0, 5.0, 5.0]]]]])
    lse_far = torch.logsumexp(far.float(), dim=-1)
    assert lse_far[..., 0] > lse_far[..., 1], \
        "a gap beyond log(R) must survive the change of aggregation"

    # The bound itself, which is what limits the whole approach.
    R = 4
    for _ in range(20):
        v = torch.randn(R) * 3.0
        assert v.max() <= torch.logsumexp(v, 0) <= v.max() + math.log(R) + 1e-5
