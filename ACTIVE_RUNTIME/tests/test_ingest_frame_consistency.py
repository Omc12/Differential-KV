"""Every prefill capture site must store K in the frame the decoder expects.

`pool_stores_rotated_k()` tells the decode gather whether to re-rotate. Prefill
must therefore store POST-RoPE keys when that predicate is true and PRE-RoPE
keys when it is false -- and `_ingest_k(rot_k, unrot_k)` is the single helper
that decides, existing (per its own docstring) so "the two sides cannot
silently disagree again".

A site that passes `unrot_key` unconditionally writes the pool in the OPPOSITE
frame from every other site. It fails silently: RoPE is orthogonal, so the
stored norms are identical and only the ANGLES are wrong. The original
occurrence of this bug measured cos 1.0000 against unrotated ground truth and
0.84-0.98 against rotated, with a depth gradient that made it look like a
retrieval-quality problem rather than a missing rotation.

Four sites still bypassed the helper after the first fix -- the dense/bypass
path, the chunked incremental-prefill path, and two fed from one `curr_unrot_k`
variable. None are exercised by first-turn NIAH, which is why the suite stayed
green; they are the short-context, second-turn and chunked paths.

This is a source-level test on purpose. The failure has no runtime signature to
assert on -- nothing raises, shapes match, norms match -- so the only place to
catch it is where the decision is made.
"""

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _capture_call_args():
    """Yield (line_no, arg_text) for the K argument of every
    capture_prefill_kv(...) call in dkv_attention."""
    import runtime.dkv_attention as DA

    src = inspect.getsource(DA)
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "capture_prefill_kv(" not in line or line.strip().startswith("#"):
            continue
        # sid/layer on the next line, K on the one after -- the shape every
        # call site in this file uses.
        for j in (i + 1, i + 2, i + 3):
            if j >= len(lines):
                break
            t = lines[j].strip()
            if t.startswith(("sid", "_sid")) or not t:
                continue
            yield i + 1, t
            break


def test_no_capture_site_passes_a_raw_unrotated_key():
    """The regression itself: an argument that names an unrotated key without
    going through _ingest_k."""
    offenders = []
    for ln, arg in _capture_call_args():
        if "_ingest_k" in arg:
            continue
        # A bare identifier is fine only if it was assigned from _ingest_k;
        # what must never appear is unrot_* used directly here.
        if re.search(r"\bunrot[_a-z]*\b", arg):
            offenders.append((ln, arg))
    assert not offenders, (
        "capture_prefill_kv called with a raw unrotated key; route it through "
        f"_ingest_k(rot, unrot): {offenders}")


def test_every_intermediate_unrot_variable_is_ingest_k_sourced():
    """The subtler half: a site can pass `chunk_unrot_k` -- which looks clean at
    the call -- while that variable was sliced from a raw unrotated tensor two
    hundred lines earlier. Check the ASSIGNMENTS, not just the calls."""
    import runtime.dkv_attention as DA

    src = inspect.getsource(DA)
    bad = []
    # Names that are KEY TENSORS: contain "unrot" and end in a k-ish suffix.
    # A bare `\w*unrot\w*` also matches booleans like `_unrotate`, which are
    # flags and not frames.
    for m in re.finditer(r"^\s*(\w*unrot\w*?_?k(?:ey)?(?:_states)?)\s*=\s*(.+)$",
                         src, re.M):
        name, rhs = m.group(1), m.group(2)
        if name in ("unrot_key", "unrot_key_states", "unrot_query_states"):
            continue                      # the raw inputs themselves
        if "_ingest_k" in rhs or "None" in rhs:
            continue
        # Slicing another already-ingested variable is fine.
        if re.match(r"^\s*curr_unrot_k\b", rhs) or re.match(r"^\s*chunk_unrot_k\b", rhs):
            continue
        bad.append((name, rhs.strip()[:80]))
    assert not bad, f"unrotated-key variables not sourced from _ingest_k: {bad}"


def test_ingest_k_selects_on_the_pool_predicate():
    """_ingest_k must ask the same question the decode gather asks, or the two
    can disagree while both look self-consistent."""
    import runtime.dkv_attention as DA

    src = inspect.getsource(DA._ingest_k)
    assert "_pool_rotated_k()" in src
    assert "rot_k if" in src and "else unrot_k" in src


def test_ingest_k_round_trips_both_ways(monkeypatch):
    import torch

    import runtime.dkv_attention as DA

    rot = torch.ones(2, 2)
    unrot = torch.zeros(2, 2)
    monkeypatch.setattr(DA, "_pool_rotated_k", lambda: True)
    assert torch.equal(DA._ingest_k(rot, unrot), rot)
    monkeypatch.setattr(DA, "_pool_rotated_k", lambda: False)
    assert torch.equal(DA._ingest_k(rot, unrot), unrot)
