"""CPU tests for the shared content-aware residual capture
(native_core/compression/residual_capture.py — the torch/CUDA-path port of
the MLX/lowrank.cpp boost machinery; CUDA_TRITON_AUDIT.md C10).

Runs on CPU: no GPU, no HF tokenizer required."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.compression.residual_capture import (
    _detect_table_rows,
    compute_boost_multipliers,
)


PROSE = [' The', ' history', ' of', ' artificial', ' intelligence', ' is',
         ' long', '.', ' Deep', ' learning', ' transformed', ' AI', '.\n']
TABLE_ROW = ['|', ' 7', 'x', '7', ' |', ' 8', '3', '.', '2', ' |\n']
MATH_PARA = ([' the', ' kernel'] * 40 + [' k', '(', 'x', ',', ' y', ')',
             ' max', ' ', '0', ',', ' ', '1', ' −', ' |', 'x', '−', 'y',
             ' |', ' δ'] + [' on', ' a', ' segment'] * 20 + ['.\n'])


def _ids(toks):
    return list(range(len(toks)))


class TestDetectTableRows:
    def test_markdown_rows_marked(self):
        toks = TABLE_ROW * 3
        marked = _detect_table_rows(toks)
        assert all(marked), marked

    def test_prose_unmarked(self):
        assert not any(_detect_table_rows(PROSE))

    def test_inline_math_paragraph_unmarked(self):
        # A paragraph-long 'line' with |x−y| norm bars has standalone pipes
        # but fails the density/line-start shape guard (the false positive
        # that marked whole blocks of the random-features paper).
        assert not any(_detect_table_rows(MATH_PARA))

    def test_latex_ampersand_row_marked(self):
        toks = [' Swin', '-T', ' &', ' 81', '.', '3', ' &', ' 755', ' \\\\\n']
        assert any(_detect_table_rows(toks))

    def test_header_and_units_cells_marked(self):
        toks = ['|', ' Kernel', ' size', ' |', ' Through', 'put', ' (',
                'imgs', '/sec', ')', ' |\n']
        marked = _detect_table_rows(toks)
        # the alpha header/unit cells are exactly the tokens the core
        # classifier misses — table capture must include them
        assert marked[7] and marked[8], marked


class TestComputeBoostMultipliers:
    def test_prose_only_core_segments_boosted(self):
        boost, n = compute_boost_multipliers(PROSE, _ids(PROSE), {}, 8192)
        # 'AI' is an all-caps core token; its segment + window is boosted,
        # ordinary prose stays 1.0
        assert n > 0
        assert boost[0] == 1.0

    def test_table_rows_outrank_plain_core(self):
        # keep the plain digit outside the table's W=2 window glue
        toks = PROSE + TABLE_ROW + PROSE + [' 42', ' apples', '.\n']
        boost, _ = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        t0 = len(PROSE)
        table_min = min(boost[t0:t0 + len(TABLE_ROW)])
        plain_digit = boost[len(PROSE) * 2 + len(TABLE_ROW)]
        assert plain_digit > 1.0
        assert table_min > plain_digit, (table_min, plain_digit)

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("DIFFKV_RESIDUAL_TOKEN_BOOST", "0")
        boost, n = compute_boost_multipliers(PROSE, _ids(PROSE), {}, 8192)
        assert boost is None and n == 0

    def test_table_capture_off_leaves_core_boost(self, monkeypatch):
        monkeypatch.setenv("DIFFKV_RESIDUAL_TABLE_CAPTURE", "0")
        toks = TABLE_ROW * 3
        boost, n = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        # digit cells are still is_core (layer-1 boost), but no priority
        monkeypatch.setenv("DIFFKV_RESIDUAL_TABLE_CAPTURE", "1")
        boost_on, n_on = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        assert n_on >= n
        assert max(boost_on) > max(boost)

    def test_owner_capture_boosts_entity_name(self):
        toks = [' The', ' Meridian', ' facility', ' processes', ' 43', '82',
                ' samples', ' per', ' day', '.\n']
        boost, _ = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        assert boost[1] > 1.0   # ' Meridian' captured as owner


@pytest.mark.skipif(
    os.environ.get("DIFFKV_RUN_TORCH_TESTS", "1") != "1",
    reason="torch not requested")
class TestTorchPathIntegration:
    """compress_layer_blocks_gpu on CPU tensors: boosted table rows must win
    the residual slots ahead of structure-blind high-error rows."""

    def test_table_rows_selected_as_residuals(self):
        import torch
        from native_core.compression import lowrank

        T, H, D = 64, 2, 16
        table_toks = (['|', ' 7', 'x', '7', ' |', ' 8', '3', '.', '2', ' |\n'] *
                      3)                     # 30 tokens of table rows
        prose_toks = [' the', ' quick', ' brown', ' fox', ' jumps', '.',
                      ' over', ' lazy', ' dogs', ' again'] * 4  # > owner_dist
        toks = (prose_toks[:T - len(table_toks)] + table_toks)[:T]
        tids = list(range(T))

        class FakeTok:
            def decode(self, ids):
                return "".join(toks[i] for i in ids)

        class FakeMgr:
            tokenizer = FakeTok()
            _session_token_ids = {"s": torch.arange(T)}
            native_pool = None
            _streaming_mgr = None

        class FakeBlock:
            session_id = "s"
            token_indices = list(range(T))
            anchor_kv = torch.zeros(1, 2, H * D)
            layer_idx = 0
            pool_idx = None
            pool = None

            def __init__(self):
                g = torch.Generator().manual_seed(0)
                self.active_k = torch.randn(1, H, T, D, generator=g)
                self.active_v = torch.randn(1, H, T, D, generator=g)

        blk = FakeBlock()
        ok = lowrank.compress_layer_blocks_gpu([blk], rank=8, manager=FakeMgr())
        assert ok
        assert blk.residual_K_positions is not None
        sel = set(blk.residual_K_positions.tolist())
        table_positions = set(range(T - len(table_toks), T))
        frac = len(sel & table_positions) / max(1, len(sel))
        # boosted table rows should dominate the kept residual set
        assert frac >= 0.7, (frac, sorted(sel))
