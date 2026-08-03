"""The trace's row resolution must not claim a token its block does not own.

`anchor_indices` holds only the ROUTED blocks. The first version of this bounded
containment by the PADDED block width (1+S), so when the traced token's own block
was NOT routed, an earlier routed block whose anchor happened to land within that
width would absorb the offset and the trace reported a real-but-wrong row.

That failure is invisible in the output: a wrong row is still a live token and
still gets a small share, which reads exactly like "the needle is routed but
scores badly" -- the single reading the mass trace exists to produce. Bounding by
seq_lens instead makes containment exact, and a token in no routed block resolves
to None so the caller can say so out loud.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.dkv_attention import _resolve_trace_row      # noqa: E402

S1 = 258                      # padded width: 1 anchor row + 257 active rows


def test_row_zero_is_the_anchor_token_itself():
    anc = torch.tensor([1000, 2000])
    sl = torch.tensor([257, 257])
    assert _resolve_trace_row(anc, sl, S1, 1000) == (0, 0, 0)
    assert _resolve_trace_row(anc, sl, S1, 2000) == (1, 0, S1)


def test_active_token_j_sits_at_row_one_plus_j():
    """Active token j is at anchor+1+j and in row 1+j -- so off IS the row."""
    anc = torch.tensor([1000, 2000])
    sl = torch.tensor([257, 257])
    n, off, row = _resolve_trace_row(anc, sl, S1, 1005)
    assert (n, off, row) == (0, 5, 5)
    n, off, row = _resolve_trace_row(anc, sl, S1, 2232)
    assert (n, off, row) == (1, 232, S1 + 232)


def test_short_routed_block_does_not_claim_a_later_token():
    """THE BUG. Block 0 is routed but holds only 60 tokens; the traced token is
    200 past its anchor, inside a block that was NOT routed. Bounding by the
    padded width would hand it to block 0."""
    anc = torch.tensor([1000])
    sl = torch.tensor([60])
    assert _resolve_trace_row(anc, sl, S1, 1200) is None, \
        "a block claimed a token beyond its seq_len -- padded-width containment"


def test_last_live_token_is_inclusive():
    """Rows 0..seq_len are live (anchor + seq_len active), so off == seq_len is
    the block's final real token, not one past the end."""
    anc = torch.tensor([1000])
    sl = torch.tensor([60])
    assert _resolve_trace_row(anc, sl, S1, 1060) == (0, 60, 60)
    assert _resolve_trace_row(anc, sl, S1, 1061) is None


def test_token_before_every_routed_block_resolves_to_none():
    anc = torch.tensor([5000, 6000])
    sl = torch.tensor([257, 257])
    assert _resolve_trace_row(anc, sl, S1, 100) is None


def test_picks_the_containing_block_not_merely_the_last_candidate():
    """With several routed blocks, the one that OWNS the token must win even
    when a later-indexed block also precedes it in absolute position."""
    anc = torch.tensor([1000, 2000, 9000])
    sl = torch.tensor([257, 257, 257])
    n, off, row = _resolve_trace_row(anc, sl, S1, 2100)
    assert n == 1 and off == 100 and row == S1 + 100


def test_unsorted_anchors_still_resolve_to_the_true_owner():
    """Routing order is not guaranteed to be ascending by position."""
    anc = torch.tensor([9000, 1000, 2000])
    sl = torch.tensor([257, 257, 257])
    n, off, row = _resolve_trace_row(anc, sl, S1, 1005)
    assert n == 1 and off == 5 and row == S1 + 5
