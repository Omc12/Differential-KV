"""The question-span pin, which never worked on CUDA.

`current_query_tokens` is read as a LEXICAL QUERY by query_router's lexical
lookup and by the decode-time query_toks set. MLX sets it from the extracted
question span (`_pending_query`); CUDA fell back to the whole prompt on every
request, so on a single-turn call it named every token in an 8k document as
part of the question -- IDF ~uniform, discriminating nothing.

Two independent faults, both silent:

  1. `KVRuntimeManager` never declared `_pending_query`, so the wrapper's write
     raised AttributeError into a bare `except Exception: pass`.
  2. The write was gated on `_factual_enabled` (off by default), so even with
     the attribute present the default path would not have filled it.

Both are source-level regressions -- nothing errors when they come back, the
lexical signal just goes uniformly wrong -- so they are pinned here rather than
left to a benchmark to notice.
"""

import inspect
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_manager_declares_pending_query():
    """Fault 1. The wrapper writes manager._pending_query[sid]; without the
    attribute that is an AttributeError swallowed by a bare except."""
    from native_core.kv_runtime_manager import KVRuntimeManager

    src = inspect.getsource(KVRuntimeManager.__init__)
    assert "self._pending_query" in src, (
        "KVRuntimeManager must declare _pending_query in __init__; "
        "hf_dkv_wrapper writes to it inside a bare except")


def test_finalize_prefers_the_pinned_question_over_the_whole_prompt():
    """The pin must WIN over the fallback, and the fallback must survive."""
    from native_core import kv_runtime_manager as KVM

    src = inspect.getsource(KVM)
    i = src.index("Query tokens: the QUESTION span")
    seg = src[i:i + 2000]
    assert '_pending_query", {}).pop(session_id' in seg, "must consume the pin"
    assert "if _pq:" in seg and "current_query_tokens = list(_pq)" in seg
    # ...and still fall back when nothing was pinned (multi-turn, or a caller
    # that supplies no question).
    assert "elif token_ids_cpu is not None:" in seg


def test_population_is_not_gated_on_the_factual_store():
    """Fault 2. current_query_tokens feeds ROUTING, not just the factual store,
    so filling it must not depend on DKV_FACTUAL_STORE."""
    import serving.hf_dkv_wrapper as W

    src = inspect.getsource(W)
    i = src.index("_pending_query[session_id] = _q_ids")
    # Walk back to the enclosing guard and check it is not the factual gate.
    head = src[:i]
    guard = head.rsplit("\n        if ", 1)[-1].splitlines()[0]
    assert "_factual_enabled" not in guard, (
        f"query-span population is gated on the factual store: {guard!r}")


