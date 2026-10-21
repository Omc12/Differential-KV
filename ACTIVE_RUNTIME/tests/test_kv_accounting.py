"""dkv_kv_bytes (benchmarks/run_longbench_cuda.py), accounting v2: only layers
that hold KV count. CPU-only, mock manager.

On a hybrid layout (attention every 4th layer, as Qwen3.5-4B) the dense
equivalent must be (attention layers) x tokens x KV bytes -- what the baselines
report -- not every decoder layer; on an all-attention layout nothing changes.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))


def _mgr(num_blocks, res_per_block, L, Hkv=4, D=256, B=1024, rank=32):
    sessions = {"s": {
        "num_blocks": num_blocks,
        "dense_lens": [512] * L,
        "comp_res_n": [[res_per_block] * n for n in num_blocks],
    }}
    cfg = types.SimpleNamespace(max_active_dense_tokens=1536, max_residual_tokens=128,
                                preset="mid")
    pool = types.SimpleNamespace(rank=rank, _vkv_halves=2, key_quant_bits=0)
    return types.SimpleNamespace(num_layers=L, kv_heads=Hkv, head_dim=D,
                                 micro_block_size=B, native_pool=pool, rank=rank,
                                 sessions=sessions, config=cfg)


def test_hybrid_layout_counts_attention_layers_only():
    from run_longbench_cuda import dkv_kv_bytes
    L, seq = 32, 32768
    nb = [31 if l % 4 == 3 else 0 for l in range(L)]     # 8 attention layers
    r = dkv_kv_bytes(_mgr(nb, 100, L), seq, "s")
    kv_tok = 4 * 256 * 2 * 2
    assert r["kv_layers"] == 8 and r["kv_acct"] == 2
    assert abs(r["kv_dense_equiv_gb"] - 8 * seq * kv_tok / 1e9) < 1e-9
    assert r["blocks_layer0"] == 31                      # first KV layer, not layer 0


def test_all_attention_layout_unchanged():
    from run_longbench_cuda import dkv_kv_bytes
    L, seq = 40, 16384
    r = dkv_kv_bytes(_mgr([15] * L, 128, L, Hkv=8, D=128), seq, "s")
    assert r["kv_layers"] == 40
    assert abs(r["kv_dense_equiv_gb"] - 40 * seq * 8 * 128 * 4 / 1e9) < 1e-9
    assert r["kv_compression_x"] > 1


def test_ratio_independent_of_linear_layers():
    """Adding non-KV layers must not change the store or the ratio."""
    from run_longbench_cuda import dkv_kv_bytes
    a = dkv_kv_bytes(_mgr([31] * 8, 100, 8), 32768, "s")
    b = dkv_kv_bytes(_mgr([0, 0, 0, 31] * 8, 100, 32), 32768, "s")
    assert abs(a["kv_physical_gb"] - b["kv_physical_gb"]) < 1e-12
    assert abs(a["kv_compression_x"] - b["kv_compression_x"]) < 1e-9


def test_int8_residual_rows_cost_their_stored_width():
    """Residual rows are charged at what the pool stores: int8 codes packed in
    32-bit words plus an fp16 scale and offset per 64 channels, not 16 bits."""
    from run_longbench_cuda import dkv_kv_bytes
    L, seq, H, D = 40, 16384, 8, 128
    m16 = _mgr([15] * L, 128, L, Hkv=H, D=D)
    m8 = _mgr([15] * L, 128, L, Hkv=H, D=D)
    m8.native_pool.residual_quant = "int8"
    m8.native_pool.residual_quant_bits = 8
    m8.native_pool.residual_quant_group_size = 64
    a = dkv_kv_bytes(m16, seq, "s")["kv_physical_gb"]
    b = dkv_kv_bytes(m8, seq, "s")["kv_physical_gb"]
    rows = L * 15 * 128
    row16, row8 = 2 * H * D * 2, 2 * H * (D + (D // 64) * 4)
    assert abs((a - b) - rows * (row16 - row8) / 1e9) < 1e-12
