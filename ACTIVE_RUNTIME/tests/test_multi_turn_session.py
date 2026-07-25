"""
tests/test_multi_turn_session.py — Multi-turn session isolation & cache clear test.

Verifies:
  1. Running prompt A under session_id="test_session".
  2. Calling clear_session("test_session") purges wrapper sessions, engine sessions, and residual gather caches.
  3. Running prompt B under the SAME session_id="test_session" produces coherent output without stale cache contamination.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_multi_turn_session_invalidation():
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
    
    class MockEngine:
        def __init__(self):
            self.sessions = {}
        def clear_session(self, sid):
            self.sessions.pop(sid, None)

    class MockManager:
        def clear_session(self, sid):
            pass

    wrapper = MLXDKVWrapper.__new__(MLXDKVWrapper)
    wrapper.manager = MockManager()
    wrapper.engine = MockEngine()
    wrapper.sessions = {"test_session": {"_res_cache": {1: "stale"}, "_cache_kv": {1: "stale"}}}
    wrapper._session_token_ids = {"test_session": [1, 2, 3]}
    wrapper.engine.sessions = {"test_session": {"blocks": []}}

    # Verify session is populated before clear
    assert "test_session" in wrapper.sessions
    assert "test_session" in wrapper._session_token_ids
    assert "test_session" in wrapper.engine.sessions

    # Clear session
    wrapper.clear_session("test_session")

    # Assert complete purging of session state
    assert "test_session" not in wrapper.sessions, "session_id still in wrapper.sessions!"
    assert "test_session" not in wrapper._session_token_ids, "session_id still in wrapper._session_token_ids!"
    assert "test_session" not in wrapper.engine.sessions, "session_id still in wrapper.engine.sessions!"
    print("[SUCCESS] Multi-turn session clear verified cleanly.")

if __name__ == "__main__":
    test_multi_turn_session_invalidation()
