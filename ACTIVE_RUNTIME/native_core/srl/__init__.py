"""
native_core/srl/__init__.py

Semantic Routing Layer (SRL) — public API.

SRL sits between the decoded query and the NativeBlockPool.
Given a query Q, it selects ~40–80 relevant pool slots from the full set
(~781 for a 25K-token context), reducing first-token latency from ~10–15s
to under 1s for simple queries.

Usage (called by KVRuntimeManager):
    from native_core.srl import build_srl_for_session, SessionSRLState

Usage (called by diffkv_attention.py decode path):
    from native_core.srl import route_query
"""

from native_core.srl.session_srl_state import SessionSRLState
from native_core.srl.semantic_index import SemanticIndex, build_semantic_index
from native_core.srl.chunk_graph import ChunkGraph, build_chunk_graph
from native_core.srl.inverted_index import InvertedTokenIndex, build_inverted_index
from native_core.srl.query_router import route_query, route_query_fixed_k, two_level_gate, adaptive_k

__all__ = [
    "SessionSRLState",
    "SemanticIndex",
    "build_semantic_index",
    "ChunkGraph",
    "build_chunk_graph",
    "InvertedTokenIndex",
    "build_inverted_index",
    "route_query",
    "route_query_fixed_k",
    "two_level_gate",
    "adaptive_k",
]
