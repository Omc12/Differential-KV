"""Pluggable KV codecs (runtime/kv_codecs.py).

Two things are pinned here: (1) moving the built-in stores behind the codec
interface changed no number (bit-identical to the quantizers they wrap, and
the prefill tile reads a codec exactly as it read the old keywords), and
(2) a codec written outside DKV -- two plain functions in examples/ -- plugs in
with one setting and runs the whole store path: compress, pool growth,
residual keys for the router, tiered paging and the prefill tile.
"""
import importlib
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PLUGIN = "examples.int8_codec:INT8"


def _nbp(monkeypatch, **env):
    for v in ("DKV_STORE", "DKV_LOWRANK", "DKV_KEY_QUANT", "DKV_VALUE_QUANT",
              "DKV_QUANT_BITS"):
        monkeypatch.delenv(v, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DKV_DISABLE_CUDA_GRAPH", "1")
    import runtime.native_block_pool as nbp
    importlib.reload(nbp)
    return nbp


def _pool(nbp, H=2, D=64, T=256, n=16):
    return nbp.NativeBlockPool(max_blocks=4 * n, num_kv_heads=H, head_dim=D, rank=32,
                               max_seq_len=T, device=DEV, dtype=torch.float16,
                               initial_blocks=n, num_layers=1, lazy=False,
                               max_residual_tokens=16, residual_quant="int8")


# ── the built-ins are the old quantizers, bit for bit ───────────────────────
@pytest.mark.parametrize("bits", [2, 4])
def test_kivi_codec_is_bit_identical_to_the_quantizers(bits):
    from runtime import kv_codecs as KC
    from runtime.native_block_pool import (quantize_keys_per_channel, dequantize_keys_per_channel,
                                           quantize_values_per_token, dequantize_values_per_token)
    T, H, D = 96, 2, 64
    c = KC.KiviCodec(T, H, D, DEV, bits=bits)
    st = {k: torch.zeros((3,) + s, dtype=dt, device=DEV) for k, (s, dt) in c.state_spec().items()}
    k = torch.randn(3, T, H, D, device=DEV) * 2
    v = torch.randn(3, T, H, D, device=DEV)
    c.encode(st, torch.arange(3, device=DEV), k, v, T)
    kq, ks, kz = quantize_keys_per_channel(k, bits, 32)
    vq, vs, vz = quantize_values_per_token(v, bits, 32)
    ref_k = dequantize_keys_per_channel(kq, ks, kz, bits, 32, torch.float16)
    ref_v = dequantize_values_per_token(vq, vs, vz, bits, 32, torch.float16)
    assert torch.equal(c.decode_keys(st, torch.float16), ref_k)
    assert torch.equal(c.decode_values(st, torch.float16), ref_v)
    pos = torch.tensor([[0, 5, 95], [1, 2, 3], [64, 65, 31]], device=DEV)
    rows = c.decode_key_rows(st, pos, torch.float32)
    full = c.decode_keys(st, torch.float32)
    assert torch.allclose(rows, torch.gather(full, 1, pos[..., None, None].expand(-1, -1, H, D)))


def test_switches_select_the_same_codecs_as_before(monkeypatch):
    cases = [({}, "codes", True, True), ({"DKV_LOWRANK": "1"}, "lowrank", False, False),
             ({"DKV_STORE": "hybrid"}, "hybrid", True, False),
             ({"DKV_KEY_QUANT": "pc4"}, "hybrid", True, False),
             ({"DKV_STORE": "codes", "DKV_QUANT_BITS": "2"}, "codes", True, True)]
    for env, name, hk, hv in cases:
        nbp = _nbp(monkeypatch, **env)
        p = _pool(nbp)
        assert (p.codec.name, p.codec.holds_keys, p.codec.holds_values) == (name, hk, hv), env
    nbp = _nbp(monkeypatch, DKV_STORE="codes", DKV_QUANT_BITS="2")
    assert _pool(nbp).key_quant_bits == 2


# ── a codec written outside DKV: two functions, one setting ─────────────────
def _blocks(N, T, H, D, seed=0):
    out, orig = [], []
    for i in range(N):
        g = torch.Generator().manual_seed(seed + i)
        b = types.SimpleNamespace(
            active_k=torch.randn(1, H, T, D, generator=g),
            active_v=torch.randn(1, H, T, D, generator=g),
            anchor_kv=torch.randn(1, 2, H, D, generator=g), pool_idx=None,
            session_id="s", layer_idx=0, anchor_idx=i * (T + 1), token_indices=[],
            state="SUBMITTED", dirty=False, micro_block_size=T)
        b.token_count = lambda T=T: T
        orig.append((b.active_k[0].permute(1, 0, 2).clone(), b.active_v[0].permute(1, 0, 2).clone(),
                     b.anchor_kv[0, 0].clone(), b.anchor_kv[0, 1].clone()))
        out.append(b)
    return out, orig


@pytest.mark.parametrize("how", ["env", "register", "object"])
def test_plugin_codec_runs_the_whole_store(monkeypatch, how):
    from runtime import kv_codecs as KC
    import examples.int8_codec as E
    if how == "env":
        store = PLUGIN
    elif how == "register":
        store = KC.register_codec("int8_test", encode=E.encode, decode=E.decode)
    else:                                       # what config={"codec": obj} does
        store = KC.register_codec("user_obj_test", E.INT8)
    nbp = _nbp(monkeypatch, DKV_STORE=store, DKV_MAX_RESIDUAL_TOKENS="16")
    import native_core.compression.lowrank as LR
    N, T, H, D = 3, 255, 2, 64
    pool = _pool(nbp, H, D, 256, 4)
    assert pool.codec.holds_keys and pool.codec.holds_values and pool.rank == 1
    mgr = types.SimpleNamespace(native_pool=pool, rank=32, tokenizer=None,
                                _streaming_mgr=None, device=DEV)
    blocks, orig = _blocks(N, T, H, D)
    assert LR.compress_layer_blocks_gpu(blocks, 32, manager=mgr) is True
    for b, (k0, v0, ak, av) in zip(blocks, orig):
        idx = torch.tensor([b.pool_idx], device=DEV)
        kd = pool.get_key_deltas(idx)[0, :T].float().cpu()
        vd = pool.get_value_deltas(idx)[0, :T].float().cpu()
        assert ((kd + ak) - k0).norm() / (k0 - ak).norm() < 0.02      # int8
        assert ((vd + av) - v0).norm() / (v0 - av).norm() < 0.02
    # residual keys for the router come from the codec, at the stored rows
    idx = torch.tensor([blocks[0].pool_idx], device=DEV)
    rk = pool.get_residual_k(idx)[0].float().cpu()
    pos = pool.residual_K_positions[idx][0].long().cpu()
    full = pool.get_key_deltas(idx)[0].float().cpu()
    for j, p in enumerate(pos.tolist()):
        if p >= 0:
            assert torch.allclose(rk[j], full[p], atol=1e-2)
    # growth keeps every slot's codec state
    before = {k: v[:pool.current_blocks].clone() for k, v in pool.codec_state.items()}
    pool._grow_pool(pool.current_blocks * 2)
    for k, v in before.items():
        assert torch.equal(pool.codec_state[k][:v.shape[0]], v)
    monkeypatch.delenv("DKV_STORE", raising=False)
    importlib.reload(nbp)


def test_tiered_store_pages_any_codec(monkeypatch):
    nbp = _nbp(monkeypatch, DKV_STORE=PLUGIN)
    pool = _pool(nbp, n=4)
    from native_core.paging.tiered_block_store import TieredBlockStore
    k = torch.randn(1, 256, 2, 64, device=DEV)
    pool.codec.encode(pool.codec_state, torch.tensor([1], device=DEV), k, k, 256)
    before = {n: t[1].clone() for n, t in pool.codec_state.items()}
    ts = TieredBlockStore(pool, pager=None, device=DEV)
    assert ts.evict_slot(1)
    for t in pool.codec_state.values():
        t[1].zero_()                              # the slot is reused meanwhile
    assert ts.restore_slot(1, blocking=True)
    for n, t in pool.codec_state.items():
        assert torch.equal(t[1], before[n]), n
    monkeypatch.delenv("DKV_STORE", raising=False)
    importlib.reload(nbp)


# ── the prefill tile reads a codec exactly as it read the old keywords ──────
def test_history_tile_codec_path_equals_keyword_path(monkeypatch):
    nbp = _nbp(monkeypatch, DKV_STORE="codes")
    from native_core.sparse_decode.triton_fused_decode import history_attend_sdpa_tile
    N, S, H, D, Hq, Q = 3, 64, 2, 64, 4, 5
    pool = _pool(nbp, H, D, S, 4)
    idx = torch.arange(N, device=DEV)
    pool.codec.encode(pool.codec_state, idx, torch.randn(N, S, H, D, device=DEV),
                      torch.randn(N, S, H, D, device=DEV), S)
    base = dict(U=torch.zeros(N, S, 1, device=DEV, dtype=torch.float16),
                V_K=torch.zeros(N, 1, Hq, D, device=DEV, dtype=torch.float16),
                V_V=torch.zeros(N, 1, Hq, D, device=DEV, dtype=torch.float16),
                anchors_K=torch.randn(N, Hq, D, device=DEV, dtype=torch.float16),
                anchors_V=torch.randn(N, Hq, D, device=DEV, dtype=torch.float16),
                scales=torch.ones(N, device=DEV, dtype=torch.float16),
                cos_sliced=torch.ones(N, 1 + S, 1, D, device=DEV, dtype=torch.float16),
                sin_sliced=torch.zeros(N, 1 + S, 1, D, device=DEV, dtype=torch.float16),
                seq_lens=torch.full((N,), S, device=DEV, dtype=torch.int32),
                inv_scale=D ** -0.5,
                residual_K_positions=None, residual_K_values=None,
                residual_V_positions=None, residual_V_values=None)
    q = torch.randn(1, Hq, Q, D, device=DEV, dtype=torch.float16)
    old = dict(base, **{n: t[idx] for n, t in pool.codec_state.items()})
    new = dict(base, **{"codec:" + n: t for n, t in pool.codec_rows(idx).items()})
    a = history_attend_sdpa_tile(q, old, Hq // H, 4, 32, 4, 32)
    b = history_attend_sdpa_tile(q, new, Hq // H, codec=pool.codec)
    assert torch.equal(a, b)
    monkeypatch.delenv("DKV_STORE", raising=False)
    importlib.reload(nbp)


# ── what is refused, and why ────────────────────────────────────────────────
def test_values_without_keys_is_refused(monkeypatch):
    from runtime import kv_codecs as KC
    import examples.int8_codec as E
    name = KC.register_codec("v_only_test", encode=E.encode, decode=E.decode, keys=False)
    nbp = _nbp(monkeypatch, DKV_STORE=name)
    with pytest.raises(ValueError, match="must hold keys"):
        _pool(nbp)
    monkeypatch.delenv("DKV_STORE", raising=False)
    importlib.reload(nbp)


def test_keys_only_function_codec_keeps_values_in_the_factor(monkeypatch):
    from runtime import kv_codecs as KC
    import examples.int8_codec as E
    name = KC.register_codec("k_only_test", encode=E.encode, decode=E.decode, values=False)
    nbp = _nbp(monkeypatch, DKV_STORE=name)
    p = _pool(nbp)
    assert p.codec.holds_keys and not p.codec.holds_values and p.rank == 32
    assert p._vkv_halves == 1
    monkeypatch.delenv("DKV_STORE", raising=False)
    importlib.reload(nbp)


def test_builtin_names_are_reserved_and_unknown_names_fail():
    from runtime import kv_codecs as KC
    import examples.int8_codec as E
    with pytest.raises(ValueError, match="built-in"):
        KC.register_codec("codes", encode=E.encode, decode=E.decode)
    with pytest.raises(KeyError):
        KC.resolve_factory("no_such_codec")


def test_encode_must_return_block_first_tensors():
    from runtime import kv_codecs as KC
    with pytest.raises(TypeError, match="dict"):
        KC.FunctionCodec(8, 1, 4, "cpu", encode=lambda x: x, decode=lambda s: s)
    with pytest.raises(TypeError, match="block axis"):
        KC.FunctionCodec(8, 1, 4, "cpu", encode=lambda x: {"q": x[0]}, decode=lambda s: s)
