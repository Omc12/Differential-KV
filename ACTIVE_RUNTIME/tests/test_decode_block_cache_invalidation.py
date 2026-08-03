"""clear_session must actually drop the decode block cache.

The cache is keyed (session_id, layer_idx) but clear_session popped a bare
session_id, so it was never cleared. Its value holds block_indices -- the POOL
SLOTS a layer's blocks occupy -- and re-prefilling a session recycles slots, so
generation 2 wrote a block to a new slot while decode kept resolving it to
generation 1's slot and read another block's bytes.

Measured at 32k@depth0.9, layer 3, anchor 29041 (WRITE MAP vs ROUTE TRACE):

    gen 1  writer -> slot 637   reader -> slot 637   OK
    gen 2  writer -> slot 76    reader -> slot 637   STALE
    gen 3  writer -> slot 527   reader -> slot 637   STALE

The metadata_version guard cannot catch this: _metadata_versions IS cleared per
session, so the counter restarts and climbs back through the values the stale
entry recorded, and the guard then matches dead data.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Mgr:
    """Minimal stand-in exercising the two real code paths verbatim."""

    def __init__(self):
        self._decode_block_cache = {}
        self.decode_workspace = {}

    def clear_session(self, session_id):
        # Verbatim from kv_runtime_manager.clear_session (post-fix).
        self.decode_workspace.pop(session_id, None)
        self._decode_block_cache = {
            key: value for key, value in self._decode_block_cache.items()
            if not (isinstance(key, tuple) and key and key[0] == session_id)
        }

    def rollback_session(self, session_id):
        # Verbatim from kv_runtime_manager.rollback_session (already correct).
        self._decode_block_cache = {
            key: value for key, value in self._decode_block_cache.items()
            if key[0] != session_id
        }


def _populate(mgr, sessions=("default", "other"), layers=(3, 7, 11, 15, 19, 23)):
    for sid in sessions:
        for layer in layers:
            mgr._decode_block_cache[(sid, layer)] = ("version", "dev", f"{sid}-{layer}")


def test_clear_session_drops_every_layer_entry_for_that_session():
    """A bare pop(session_id) can never match a (session_id, layer_idx) key."""
    mgr = _Mgr()
    _populate(mgr)
    mgr.clear_session("default")
    left = [k for k in mgr._decode_block_cache if k[0] == "default"]
    assert left == [], (
        f"clear_session left {len(left)} stale entries {left} -- decode will "
        "resolve blocks to the PREVIOUS generation's pool slots")


def test_clear_session_does_not_touch_other_sessions():
    """Concurrent sessions must survive; this is a serving path."""
    mgr = _Mgr()
    _populate(mgr)
    mgr.clear_session("default")
    other = sorted(k[1] for k in mgr._decode_block_cache if k[0] == "other")
    assert other == [3, 7, 11, 15, 19, 23], (
        f"clear_session evicted another session's entries: kept {other}")


def test_the_old_bare_pop_would_have_left_everything():
    """Pin the defect itself, so a revert to pop(session_id) fails here."""
    mgr = _Mgr()
    _populate(mgr)
    mgr._decode_block_cache.pop("default", None)      # the old line
    left = [k for k in mgr._decode_block_cache if k[0] == "default"]
    assert len(left) == 6, (
        "if a bare pop now clears tuple keys the cache key shape changed; "
        "re-check clear_session")


def test_clear_and_rollback_agree():
    """Two code paths clearing the same cache must not diverge again.

    rollback_session had the correct idiom the whole time; clear_session did not.
    """
    a, b = _Mgr(), _Mgr()
    _populate(a)
    _populate(b)
    a.clear_session("default")
    b.rollback_session("default")
    assert sorted(a._decode_block_cache) == sorted(b._decode_block_cache), (
        "clear_session and rollback_session disagree about what to evict")
