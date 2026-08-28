"""Tests for Solution A (Selective Global-Only DKV Patching) on MLX for Gemma 4 E2B
and homogeneous regression tests.
"""
import os
import sys
import types
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.mlx_dkv_wrapper import (
    MLXDKVWrapper,
    MLXKVBlockManager,
    MLXQwenModel,
    attention_forward,
    _resolve_attn_dims,
)


class MockRoPE:
    def __init__(self, dims: int, base: float = 10000.0, scale: float = 1.0):
        self.dims = dims
        self.dim = dims
        self.base = base
        self.scale = scale
        self.traditional = False

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


class MockSlidingAttention(nn.Module):
    """Sliding-window bounded attention (head_dim=256, sliding_window=512)."""
    def __init__(self, layer_idx: int, hidden_size: int = 2048, n_heads: int = 8, head_dim: int = 256):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = True
        self.layer_type = "sliding_attention"
        self.sliding_window = 512
        self.n_heads = n_heads
        self.n_kv_heads = 1
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.rope = MockRoPE(dims=head_dim)
        self.call_count = 0

    def __call__(self, x: mx.array, mask: mx.array = None, cache: tuple = None) -> mx.array:
        self.call_count += 1
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None and hasattr(cache, "update_and_fetch"):
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MockGlobalAttention(nn.Module):
    """Global full-attention layer (head_dim=512, full context)."""
    def __init__(self, layer_idx: int, hidden_size: int = 2048, n_heads: int = 8, head_dim: int = 512, is_kv_shared: bool = False, target_layer_idx: int = 14):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = False
        self.layer_type = "full_attention"
        self.is_kv_shared_layer = is_kv_shared
        self.target_layer_idx = target_layer_idx if is_kv_shared else layer_idx
        self.n_heads = n_heads
        self.n_kv_heads = 1
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        if not is_kv_shared:
            self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.rope = MockRoPE(dims=head_dim)
        self.call_count = 0

    def __call__(self, x: mx.array, mask: mx.array = None, cache: tuple = None) -> mx.array:
        self.call_count += 1
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        if not self.is_kv_shared_layer:
            k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        else:
            k, v = None, None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MockDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, attn: nn.Module):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = attn

    def __call__(self, x: mx.array, mask: mx.array = None, cache: tuple = None) -> mx.array:
        return self.self_attn(x, mask=mask, cache=cache)


class MockBackbone(nn.Module):
    def __init__(self, layers: list, hidden_size: int = 2048):
        super().__init__()
        self.layers = layers
        self.embed_tokens = nn.Embedding(32000, hidden_size)

    def __call__(self, x: mx.array, cache: list = None) -> mx.array:
        if x.ndim == 2:
            h = self.embed_tokens(x)
        else:
            h = x
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None and i < len(cache) else None
            h = layer(h, cache=c)
        return h


class MockGemma4Model(nn.Module):
    """35-layer Gemma 4 E2B architecture:
    - 28 Sliding-Window Layers (head_dim=256)
    - 7 Global Full-Attention Layers (head_dim=512):
      - 3 Non-shared: [4, 9, 14]
      - 4 Shared: [19, 24, 29, 34] -> target layer 14
    """
    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.global_layers = [4, 9, 14, 19, 24, 29, 34]
        self.non_shared_global = [4, 9, 14]
        self.shared_global = [19, 24, 29, 34]
        
        layers = []
        for i in range(35):
            if i in self.non_shared_global:
                attn = MockGlobalAttention(i, hidden_size=hidden_size, head_dim=512, is_kv_shared=False)
            elif i in self.shared_global:
                attn = MockGlobalAttention(i, hidden_size=hidden_size, head_dim=512, is_kv_shared=True, target_layer_idx=14)
            else:
                attn = MockSlidingAttention(i, hidden_size=hidden_size, head_dim=256)
            layers.append(MockDecoderLayer(i, attn))

        self.model = MockBackbone(layers, hidden_size=hidden_size)
        self.layers = self.model.layers
        self.lm_head = nn.Linear(hidden_size, 32000, bias=False)

    def __call__(self, x: mx.array, cache: list = None) -> mx.array:
        hidden = self.model(x, cache=cache)
        return self.lm_head(hidden)


class MockHomogeneousModel(nn.Module):
    """Standard 28-layer model with uniform head_dim=128 across all layers (e.g. Qwen2.5 / Llama)."""
    def __init__(self, hidden_size: int = 2048, num_layers: int = 28, head_dim: int = 128):
        super().__init__()
        layers = []
        for i in range(num_layers):
            attn = MockGlobalAttention(i, hidden_size=hidden_size, head_dim=head_dim, is_kv_shared=False)
            layers.append(MockDecoderLayer(i, attn))
        self.model = MockBackbone(layers, hidden_size=hidden_size)
        self.layers = self.model.layers
        self.lm_head = nn.Linear(hidden_size, 32000, bias=False)

    def __call__(self, x: mx.array, cache: list = None) -> mx.array:
        hidden = self.model(x, cache=cache)
        return self.lm_head(hidden)


class MockRotatingCache:
    """Mock MLX RotatingKVCache for sliding layers."""
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)[:, :, -self.max_size:, :]
            self.values = mx.concatenate([self.values, values], axis=2)[:, :, -self.max_size:, :]
        self.offset += keys.shape[2]
        return self.keys, self.values


# ─────────────────────────────────────────────────────────────────────────────
# UNIT & ARCHITECTURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_gemma4_patching_and_layer_mapping():
    """Verify that Gemma 4 E2B is correctly classified and mapped into DKV."""
    model = MockGemma4Model()
    manager = MLXKVBlockManager(
        num_layers=len(model.layers),
        heads=8,
        kv_heads=1,
        head_dim=512,
        rank=16,
        block_size=256,
    )

    wrapper = object.__new__(MLXDKVWrapper)
    wrapper.manager = manager
    wrapper.config = {}
    wrapper.base_rank = 16
    wrapper.block_size = 256
    wrapper.layer_adaptive_rank = False
    wrapper.stop_token_ids = set()

    wrapper._patch_attention_layers(model)

    # 1. Check attended layers
    assert manager._attended_layers == [4, 9, 14], f"Expected [4, 9, 14], got {manager._attended_layers}"
    assert len(manager._attended_layers) == 3

    # 2. Check sliding layers: not in attended, is_sliding is True
    for i in range(35):
        attn = model.layers[i].self_attn
        if i in [4, 9, 14, 19, 24, 29, 34]:
            assert not attn.is_sliding
        else:
            assert attn.is_sliding

    # 3. Check non-shared global layers: slots 0, 1, 2
    assert model.layers[4].self_attn.kv_slot == 0
    assert model.layers[4].self_attn.kv_layer_idx == 4
    assert not model.layers[4].self_attn.is_kv_shared_layer

    assert model.layers[9].self_attn.kv_slot == 1
    assert model.layers[9].self_attn.kv_layer_idx == 9
    assert not model.layers[9].self_attn.is_kv_shared_layer

    assert model.layers[14].self_attn.kv_slot == 2
    assert model.layers[14].self_attn.kv_layer_idx == 14
    assert not model.layers[14].self_attn.is_kv_shared_layer

    # 4. Check shared global layers: map directly to slot 2 and target layer 14
    for i in [19, 24, 29, 34]:
        attn = model.layers[i].self_attn
        assert attn.is_kv_shared_layer, f"Layer {i} should be kv shared"
        assert attn.kv_slot == 2, f"Layer {i} kv_slot should be 2, got {attn.kv_slot}"
        assert attn.kv_layer_idx == 14, f"Layer {i} kv_layer_idx should be 14, got {attn.kv_layer_idx}"

    # 5. Check RoPE parameters resolved to head_dim=512
    assert manager._rope_dims == 512


def test_gemma4_pool_memory_allocation_sizing():
    """Verify that MLX session pool only allocates memory for the 3 unique global layers (>90% savings)."""
    model = MockGemma4Model()
    manager = MLXKVBlockManager(
        num_layers=35,
        heads=8,
        kv_heads=1,
        head_dim=512,
        rank=16,
        block_size=256,
    )
    wrapper = object.__new__(MLXDKVWrapper)
    wrapper.manager = manager
    wrapper.config = {}
    wrapper.base_rank = 16
    wrapper.block_size = 256
    wrapper.layer_adaptive_rank = False
    wrapper.stop_token_ids = set()

    wrapper._patch_attention_layers(model)

    manager.init_session("test_sess", prefill_len=512)
    session = manager.sessions["test_sess"]

    # comp_anc_k has length 35 (one entry per layer index)
    anc_k = session["comp_anc_k"]
    assert len(anc_k) == 35

    # Exactly layers 4, 9, 14 have non-zero row shape
    allocated_layers = []
    zero_row_layers = []
    for l_idx in range(35):
        shape = anc_k[l_idx].shape
        if shape[0] > 0:
            allocated_layers.append(l_idx)
            assert shape[1] == 1      # kv_heads
            assert shape[2] == 512    # head_dim
        else:
            zero_row_layers.append(l_idx)
            assert shape[0] == 0      # 0-row empty slab

    assert allocated_layers == [4, 9, 14]
    assert len(zero_row_layers) == 32


def test_gemma4_decode_and_prefill_execution():
    """Verify that prefill and decode execute without error on Gemma 4 and shared layers reuse layer 14."""
    model = MockGemma4Model(hidden_size=512)
    manager = MLXKVBlockManager(
        num_layers=35,
        heads=8,
        kv_heads=1,
        head_dim=512,
        rank=16,
        block_size=256,
    )
    wrapper = object.__new__(MLXDKVWrapper)
    wrapper.manager = manager
    wrapper.config = {}
    wrapper.base_rank = 16
    wrapper.block_size = 256
    wrapper.layer_adaptive_rank = False
    wrapper.stop_token_ids = set()

    wrapper._patch_attention_layers(model)
    qwen_model = MLXQwenModel(model, manager)
    manager.patched_model = qwen_model

    # 1. Prefill step (sequence length 16)
    manager.init_session("default", prefill_len=16)
    manager.active_session_ids = ["default"]
    manager.position_ids = np.zeros((1, 16), dtype=np.int64)
    for p in range(16):
        manager.position_ids[0, p] = p

    x_prefill = mx.random.normal((1, 16, 512))
    # Run through the layers directly
    out_prefill = model(x_prefill)
    assert out_prefill.shape == (1, 16, 32000)

    # Verify that layer 4, 9, 14 captured prefill KV into dense window
    sess = manager.sessions["default"]
    assert sess["dense_lens"][4] == 16
    assert sess["dense_lens"][9] == 16
    assert sess["dense_lens"][14] == 16
    # Shared layer 19 should NOT have duplicate captured KV
    assert sess["dense_lens"][19] == 0

    # 2. Decode step (L=1)
    manager.position_ids = np.array([[16]], dtype=np.int64)
    x_decode = mx.random.normal((1, 1, 512))
    out_decode = model(x_decode)
    assert out_decode.shape == (1, 1, 32000)


def test_homogeneous_model_no_regression():
    """Verify standard homogeneous model (e.g. Qwen2.5-style 28 layers) runs with zero regressions."""
    model = MockHomogeneousModel(hidden_size=256, num_layers=28, head_dim=128)
    manager = MLXKVBlockManager(
        num_layers=28,
        heads=8,
        kv_heads=1,
        head_dim=128,
        rank=16,
        block_size=256,
    )
    wrapper = object.__new__(MLXDKVWrapper)
    wrapper.manager = manager
    wrapper.config = {}
    wrapper.base_rank = 16
    wrapper.block_size = 256
    wrapper.layer_adaptive_rank = False
    wrapper.stop_token_ids = set()

    wrapper._patch_attention_layers(model)
    qwen_model = MLXQwenModel(model, manager)
    manager.patched_model = qwen_model

    # All 28 layers should be attended
    assert manager._attended_layers == list(range(28))
    assert len(manager._attended_layers) == 28

    manager.init_session("default", prefill_len=8)
    manager.active_session_ids = ["default"]
    manager.position_ids = np.arange(8, dtype=np.int64).reshape(1, 8)

    # Prefill
    x = mx.random.normal((1, 8, 256))
    out = model(x)
    assert out.shape == (1, 8, 32000)

    # Decode
    manager.position_ids = np.array([[8]], dtype=np.int64)
    x_dec = mx.random.normal((1, 1, 256))
    out_dec = model(x_dec)
    assert out_dec.shape == (1, 1, 32000)


def test_gemma4_multi_step_decode_with_cache_retention():
    """Verify multi-step generation maintains sliding cache while nulling global caches."""
    model = MockGemma4Model(hidden_size=512)
    manager = MLXKVBlockManager(
        num_layers=35,
        heads=8,
        kv_heads=1,
        head_dim=512,
        rank=16,
        block_size=256,
    )
    wrapper = object.__new__(MLXDKVWrapper)
    wrapper.manager = manager
    wrapper.config = {}
    wrapper.base_rank = 16
    wrapper.block_size = 256
    wrapper.layer_adaptive_rank = False
    wrapper.stop_token_ids = set()

    wrapper._patch_attention_layers(model)
    qwen_model = MLXQwenModel(model, manager)
    manager.patched_model = qwen_model

    # Initialize prefill caches (sliding layers have MockRotatingCache, global layers have KVCache)
    session_id = "test_gen_sess"
    qwen_model._dkv_session_ids = [session_id]
    manager.init_session(session_id, prefill_len=8)
    manager.active_session_ids = [session_id]

    caches = []
    for i in range(35):
        if i in [4, 9, 14, 19, 24, 29, 34]:
            caches.append(MockRotatingCache(max_size=2048))  # Global
        else:
            caches.append(MockRotatingCache(max_size=512))   # Sliding

    cache_key = (session_id,)
    qwen_model._prefill_caches[cache_key] = caches

    # 1. Prefill step via qwen_model.__call__
    input_ids = torch.randint(0, 1000, (1, 8))
    pos_prefill = torch.arange(8).unsqueeze(0)
    out_prefill = qwen_model(input_ids, pos_prefill)
    assert out_prefill.logits.shape == (1, 1, 32000)

    # 2. Decode step 1: sliding caches must be preserved, global caches nulled out
    decode_ids_1 = torch.tensor([[100]])
    pos_dec_1 = torch.tensor([[8]])
    out_dec_1 = qwen_model(decode_ids_1, pos_dec_1)
    assert out_dec_1.logits.shape == (1, 1, 32000)

    cached_list = qwen_model._prefill_caches.get(cache_key)
    assert cached_list is not None
    for i in range(35):
        if i in [4, 9, 14, 19, 24, 29, 34]:
            assert cached_list[i] is None, f"Global layer {i} cache should be None in decode"
        else:
            assert cached_list[i] is not None, f"Sliding layer {i} cache must be preserved"

    # 3. Decode step 2
    decode_ids_2 = torch.tensor([[101]])
    pos_dec_2 = torch.tensor([[9]])
    out_dec_2 = qwen_model(decode_ids_2, pos_dec_2)
    assert out_dec_2.logits.shape == (1, 1, 32000)
