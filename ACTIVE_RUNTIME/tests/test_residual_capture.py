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
    atomic_runs,
    compute_boost_multipliers,
    rank_runs_by_query,
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

    def test_pdf_aligned_rows_marked(self):
        # PDF copy-paste: whitespace-aligned columns, x as the multiplication
        # glyph, NO pipes — the columnar rule must fire on consecutive rows
        # and pull in the header line above.
        header = [' Kernel', ' size', '   Top', '-', '1', ' (%)',
                  '   Through', 'put', '\n']
        row1 = ['3', '×', '3', '           ', '79', '.', '1', '       ',
                '151', '2', '\n']
        row2 = ['5', '×', '5', '           ', '80', '.', '3', '       ',
                '137', '7', '\n']
        toks = header + row1 + row2
        marked = _detect_table_rows(toks)
        assert all(marked[len(header):]), marked          # both data rows
        assert all(marked[:len(header)]), marked          # header joins

    def test_single_numeric_prose_line_unmarked(self):
        # one prose sentence ending in a number, neighbors are plain prose —
        # no consecutive numeric line, so the columnar rule must NOT fire
        toks = [' The', ' model', ' achieves', ' 83', '.', '2', ' accuracy',
                ' at', ' 15', '12', ' images', '\n',
                ' which', ' is', ' a', ' strong', ' result', '.\n',
                ' The', ' training', ' recipe', ' is', ' standard', '.\n']
        assert not any(_detect_table_rows(toks))


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
        monkeypatch.setenv("DKV_RESIDUAL_TOKEN_BOOST", "0")
        boost, n = compute_boost_multipliers(PROSE, _ids(PROSE), {}, 8192)
        assert boost is None and n == 0

    def test_table_capture_off_leaves_core_boost(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_TABLE_CAPTURE", "0")
        toks = TABLE_ROW * 3
        boost, n = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        # digit cells are still is_core (layer-1 boost), but no priority
        monkeypatch.setenv("DKV_RESIDUAL_TABLE_CAPTURE", "1")
        boost_on, n_on = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        assert n_on >= n
        assert max(boost_on) > max(boost)

    def test_owner_capture_boosts_entity_name(self):
        toks = [' The', ' Meridian', ' facility', ' processes', ' 43', '82',
                ' samples', ' per', ' day', '.\n']
        boost, _ = compute_boost_multipliers(toks, _ids(toks), {}, 8192)
        assert boost[1] > 1.0   # ' Meridian' captured as owner


@pytest.mark.skipif(
    os.environ.get("DKV_RUN_TORCH_TESTS", "1") != "1",
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
        # ASSERT COVERAGE, NOT SHARE. The invariant that protects a table is
        # "every table row got an exact slot" -- a row left to the low-rank
        # reconstruction is real data loss. The old check divided by len(sel), so
        # it also failed whenever the residual set grew for an unrelated good
        # reason, penalising capturing MORE. Rarity capture does exactly that on
        # this input (all 64 tokens are unique, so every one scores maximum IDF):
        # selection went 36 -> 64 rows and the share fell 0.83 -> 0.47 while
        # in-table coverage stayed at 30/30. Coverage is the property worth
        # pinning; share is an artefact of the budget.
        missed = sorted(table_positions - sel)
        assert not missed, (
            f"{len(missed)} table rows left to low-rank reconstruction: {missed}")


# Qwen2.5 splits the validator's needle into exactly these eleven tokens.
CODE = [' Falcon', '-', '9', '4', '2', '7', '-', '6', '1', '8', '3']
SENT = [' The', ' secret', ' passcode', ' is'] + CODE + ['.', chr(10)]


class TestAtomicRuns:
    """A code is worth nothing captured half-way, so it has to be ONE span."""

    def test_code_and_its_owner_are_one_run(self):
        runs = atomic_runs(SENT)
        lo, hi = next(r for r in runs if r[0] <= 4 and 14 <= r[1])
        assert (lo, hi) == (4, 15), (lo, hi, runs)   # ' Falcon' .. '3'

    def test_owner_word_is_included(self):
        # 'Falcon' is prose by shape -- alphabetic and title-case -- so only the
        # owner walk-back pulls it in, and a code without the word naming it is
        # as useless as the word without the code.
        lo, _hi = atomic_runs(SENT)[0]
        assert SENT[lo] == ' Falcon'

    def test_merge_never_destroys_the_run_it_should_protect(self):
        # REGRESSION. Merging across a one-token gap used to be unconditional, so
        # in dense numeric prose the code chained into the following figure's
        # numbers, the combined span blew past max_len, and the length filter
        # dropped it whole -- leaving NO run over the code. Measured at 8k depth
        # 0.58: capture 1-4 of 11 at every layer, answer 'Falcon-942.'.
        figure = [' 56', ' x', ' 112', ' x', ' 224', ' x', ' 448', ' x',
                  ' 896', ' x', ' 1792', ' x', ' 3584', ' x', ' 7168'] * 3
        toks = SENT + figure
        runs = atomic_runs(toks)
        assert any(lo <= 4 and 15 <= hi for lo, hi in runs), runs

    def test_sentence_break_is_not_inside_a_run(self):
        figure = [' 56', ' x', ' 112', ' x', ' 224']
        runs = atomic_runs(SENT + figure)
        code = next(r for r in runs if r[0] <= 4 < r[1])
        assert code[1] <= len(SENT), (code, runs)

    def test_overlong_single_segment_is_dropped_not_truncated(self):
        # A whole table row is not an atomic unit: taking it all-or-nothing would
        # spend the entire budget on one line. Those fall back to per-token.
        row = [str(i) for i in range(60)]
        assert atomic_runs(row) == []

    def test_plain_prose_has_no_runs(self):
        assert atomic_runs(PROSE[:8]) == []


@pytest.mark.skipif(
    os.environ.get("DKV_RUN_TORCH_TESTS", "1") != "1",
    reason="torch not requested")
class TestRunAtomicSelection:
    """The same slot count, spent on a COMPLETE set instead of a truncated one."""

    @pytest.fixture(autouse=True)
    def _reserve_all(self, monkeypatch):
        # These cases hand in runs with NO query priority. The shipped default
        # (`query`) deliberately reserves nothing for those, so they pin the
        # `all` mode; the default's own behaviour is covered below.
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "all")

    def _fixture(self):
        import torch
        filler = [' w%s' % chr(97 + i % 26) for i in range(40)]
        toks = filler + CODE
        scores = torch.cat([
            torch.linspace(9.0, 2.0, 40),            # filler out-errors the code
            torch.tensor([8.6, 8.4, 8.2, 8.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        ])
        return toks, scores, atomic_runs(toks), set(range(40, 51))

    def test_whole_run_survives_the_pool_truncation(self):
        from native_core.compression.lowrank import (
            _select_residual_rows, _topk_with_coverage)
        _toks, scores, runs, code = self._fixture()
        cap = 40
        base = _topk_with_coverage(scores, cap, 0.0)
        sel = _select_residual_rows(scores, cap, 0.0, runs=runs, res_cap=cap)
        kept_base = len(code & set(base.indices.tolist()[:cap]))
        kept_run = len(code & set(sel.indices.tolist()[:cap]))
        assert kept_base < len(code), kept_base      # per-token truncates it
        assert kept_run == len(code), (kept_run, sel.indices.tolist()[:cap])

    def test_budget_is_not_raised(self):
        from native_core.compression.lowrank import (
            _select_residual_rows, _topk_with_coverage)
        _toks, scores, runs, _code = self._fixture()
        base = _topk_with_coverage(scores, 40, 0.0)
        sel = _select_residual_rows(scores, 40, 0.0, runs=runs, res_cap=40)
        assert sel.indices.numel() == base.indices.numel()

    def test_values_match_their_indices(self):
        # _reorder_runs_first rebuilds `values` for the new order; a mismatch here
        # would silently corrupt the caller's `values > threshold` mask.
        import torch
        from native_core.compression.lowrank import _select_residual_rows
        _toks, scores, runs, _code = self._fixture()
        sel = _select_residual_rows(scores, 40, 0.0, runs=runs, res_cap=40)
        assert torch.allclose(sel.values, scores[sel.indices.long()])

    def test_disabled_reproduces_the_plain_ranking(self, monkeypatch):
        from native_core.compression.lowrank import (
            _select_residual_rows, _topk_with_coverage)
        monkeypatch.setenv("DKV_RESIDUAL_RUN_ATOMIC", "0")
        _toks, scores, runs, _code = self._fixture()
        off = _select_residual_rows(scores, 40, 0.0, runs=runs, res_cap=40)
        assert off.indices.tolist() == _topk_with_coverage(
            scores, 40, 0.0).indices.tolist()

    def test_run_too_large_for_the_remainder_reserves_nothing(self):
        # ALL-OR-NOTHING. A run that cannot fit must not be taken in PART, so
        # with a budget below the run length nothing is reserved and the result
        # is exactly the per-token ranking.
        #
        # That is not the same as "no row of the code is selected": the leftover
        # slots are still filled by the ordinary ranking, which is free to pick a
        # high-error row that happens to sit inside the run. Suppressing that
        # would be a regression, not the guarantee.
        from native_core.compression.lowrank import (
            _select_residual_rows, _topk_with_coverage)
        _toks, scores, runs, _code = self._fixture()
        assert min(hi - lo for lo, hi in runs) > 5
        sel = _select_residual_rows(scores, 5, 0.0, runs=runs, res_cap=5)
        assert sel.indices.tolist() == _topk_with_coverage(
            scores, 5, 0.0).indices.tolist()


def test_both_residual_producers_agree_on_the_boost_cache_arity():
    """manager._res_capture_boost_rows is written by the batched GPU compressor
    and read by the deferred one. They are the SAME dict, so a producer storing
    a different tuple width breaks the other -- and the reader's
    `except Exception: pass` would swallow it, silently disabling the boost and
    the residual budget floor with it."""
    import io
    import re
    root = os.path.join(os.path.dirname(__file__), "..")
    lr = io.open(os.path.join(root, "native_core", "compression", "lowrank.py"),
                 encoding="utf-8").read()
    km = io.open(os.path.join(root, "native_core", "kv_runtime_manager.py"),
                 encoding="utf-8").read()
    unpacks = (re.findall(r"= _cached_boost", lr)
               + re.findall(r"= _cached_boost", km))
    assert len(unpacks) == 2, unpacks
    for src, name in ((lr, "lowrank"), (km, "kv_runtime_manager")):
        line = next(l for l in src.splitlines() if "= _cached_boost" in l)
        assert line.count(",") == 2, (name, line)


class TestQueryRunPriority:
    """WHICH whole run wins the slots -- the one question error cannot answer."""

    # A block shaped like the one that actually failed: the answer's run plus
    # the paper front-matter (authors, a URL) that was evicting it at 4 of 28
    # layers, all of it genuinely hard to reconstruct and none of it asked for.
    BLOCK = ([' The', ' secret', ' passcode', ' is'] + CODE + ['.', chr(10)]
             + [' filler'] * 40
             + [' Hassani', '1', ',', ' Walton', '2', chr(10)]
             + [' filler'] * 40
             + [' https', '://', 'github', '.', 'com', '/', 'SHI', '-', 'Labs',
                chr(10)])

    def _ids(self, toks):
        return list(range(500, 500 + len(toks)))

    def test_answer_run_is_prioritised_over_irrelevant_runs(self):
        ids = self._ids(self.BLOCK)
        runs = atomic_runs(self.BLOCK)
        q = [ids[1], ids[2]]                       # ' secret', ' passcode'
        out = rank_runs_by_query(self.BLOCK, ids, q, runs)
        assert any(len(r) > 2 for r in out), out
        code = next(r for r in out if r[0] <= 4 < r[1])
        others = [r for r in out if r is not code]
        assert code[2] == 1
        assert all(r[2] == 0 for r in others), out

    def test_no_query_leaves_the_error_order_untouched(self):
        runs = atomic_runs(self.BLOCK)
        assert rank_runs_by_query(
            self.BLOCK, self._ids(self.BLOCK), None, runs) == runs

    def test_a_query_matching_everything_has_its_MARKS_dropped(self):
        # `_pending_query` falls back to the WHOLE PROMPT when no question span
        # can be extracted. That would mark every run relevant, which carries no
        # ranking information and would silently randomise the order.
        #
        # The marks go, the ANNOTATION stays. "Matched too much to discriminate"
        # is still a real query, and the selector must not read it as "there was
        # no query" -- that fallback reserves by error, which is exactly what
        # cost document synthesis 13.3 -> 6.7.
        ids = self._ids(self.BLOCK)
        runs = atomic_runs(self.BLOCK)
        out = rank_runs_by_query(self.BLOCK, ids, ids, runs)
        assert all(len(r) == 3 and r[2] == 0 for r in out), out

    def test_priority_beats_a_higher_error_run_in_the_greedy(self):
        import torch
        from native_core.compression.lowrank import _select_residual_rows
        toks = ([' alpha'] * 8 + CODE + [' beta'] * 8
                + [' Hassani', '1', ',', ' Walton', '2'])
        runs = atomic_runs(toks)
        code = next(i for i, r in enumerate(runs) if r[0] <= 8 < r[1])
        # The irrelevant run out-errors the answer by a wide margin.
        scores = torch.ones(len(toks))
        for lo, hi in runs:
            for r in range(lo, hi):
                scores[r] = 9.0
        for r in range(runs[code][0], runs[code][1]):
            scores[r] = 2.0
        prio = [(lo, hi, 1 if i == code else 0) for i, (lo, hi) in enumerate(runs)]
        cap = 11
        plain = _select_residual_rows(scores, cap, 0.0, runs=runs, res_cap=cap)
        withp = _select_residual_rows(scores, cap, 0.0, runs=prio, res_cap=cap)
        rows = set(range(runs[code][0], runs[code][1]))
        assert not rows <= set(plain.indices.tolist()[:cap])
        assert rows <= set(withp.indices.tolist()[:cap])


@pytest.mark.skipif(
    os.environ.get("DKV_RUN_TORCH_TESTS", "1") != "1",
    reason="torch not requested")
class TestQueryScopedReservation:
    """The shipped default: reserve slots ONLY for runs the query points at."""

    def _fixture(self):
        import torch
        toks = [' alpha'] * 8 + CODE + [' beta'] * 8 + [' Hassani', '1', ',']
        runs = atomic_runs(toks)
        code_i = next(i for i, r in enumerate(runs) if r[0] <= 8 < r[1])
        scores = torch.ones(len(toks))
        for lo, hi in runs:                      # every run out-errors the prose
            for r in range(lo, hi):
                scores[r] = 9.0
        for r in range(runs[code_i][0], runs[code_i][1]):
            scores[r] = 2.0                      # the ANSWER reconstructs well
        return toks, scores, runs, code_i

    def test_unasked_runs_do_not_claim_slots(self, monkeypatch):
        # 4f bought needle recall by reserving every run, and that is what cost
        # document synthesis its scattered rare-prose rows (6.7 against 13.3).
        from native_core.compression.lowrank import (
            _select_residual_rows, _topk_with_coverage)
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "query")
        _toks, scores, runs, _ci = self._fixture()
        plain = _topk_with_coverage(scores, 12, 0.0)
        sel = _select_residual_rows(scores, 12, 0.0, runs=runs, res_cap=12)
        assert sel.indices.tolist() == plain.indices.tolist()

    def test_the_asked_for_run_still_gets_its_guarantee(self, monkeypatch):
        from native_core.compression.lowrank import _select_residual_rows
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "query")
        _toks, scores, runs, code_i = self._fixture()
        prio = [(lo, hi, 1 if i == code_i else 0)
                for i, (lo, hi) in enumerate(runs)]
        cap = 12
        sel = _select_residual_rows(scores, cap, 0.0, runs=prio, res_cap=cap)
        rows = set(range(runs[code_i][0], runs[code_i][1]))
        assert rows <= set(sel.indices.tolist()[:cap])

    def test_only_the_marked_run_is_reserved(self, monkeypatch):
        # The other runs out-error the answer, so if they were reserved too the
        # answer would be evicted -- which is exactly the 4f behaviour.
        from native_core.compression.lowrank import _select_residual_rows
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "query")
        _toks, scores, runs, code_i = self._fixture()
        prio = [(lo, hi, 1 if i == code_i else 0)
                for i, (lo, hi) in enumerate(runs)]
        sel = _select_residual_rows(scores, 12, 0.0, runs=prio, res_cap=12)
        front = sel.indices.tolist()[:len(range(*runs[code_i]))]
        assert sorted(front) == list(range(*runs[code_i]))


@pytest.mark.skipif(
    os.environ.get("DKV_RUN_TORCH_TESTS", "1") != "1",
    reason="torch not requested")
class TestNoQueryIsNotTheSameAsQuerySaidNo:
    """The shipped default has to tell those two apart.

    `_pending_query` is best-effort -- hf_dkv_wrapper fills it from an explicit
    query_text, else from the chat messages, else not at all, all under a bare
    except. If "no query" were treated as "the query points nowhere", every block
    would reserve nothing and the whole of 4f-4h would silently switch off:
    measured at 3/12 on the 8k natural sweep with the signal neutered.
    """

    def _fixture(self):
        import torch
        toks = [' alpha'] * 8 + CODE + [' beta'] * 8
        runs = atomic_runs(toks)
        ci = next(i for i, r in enumerate(runs) if r[0] <= 8 < r[1])
        # The filler out-errors the answer, so a per-token top-k truncates the
        # run and the reservation is the only thing that keeps it whole.
        scores = torch.full((len(toks),), 9.0)
        for lo, hi in runs:
            for r in range(lo, hi):
                scores[r] = 2.0
        return toks, scores, runs, ci

    def test_no_query_falls_back_to_reserving_by_error(self, monkeypatch):
        from native_core.compression.lowrank import (
            _select_residual_rows, _topk_with_coverage)
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "query_first")
        _toks, scores, runs, ci = self._fixture()
        assert all(len(r) == 2 for r in runs)          # bare spans == no query
        cap = 11
        sel = _select_residual_rows(scores, cap, 0.0, runs=runs, res_cap=cap)
        rows = set(range(*runs[ci]))
        # The reservation still happens, so a whole run survives the truncation.
        assert rows <= set(sel.indices.tolist()[:cap])
        assert sel.indices.tolist() != _topk_with_coverage(
            scores, cap, 0.0).indices.tolist()

    def test_query_pointing_nowhere_still_reserves_ONE_run(self, monkeypatch):
        # The relevance signal is LEXICAL, so "the query marked nothing here" is
        # not proof the block is irrelevant -- the answer may simply be worded
        # differently from the question. Reserving nothing for unmarked blocks
        # scores 2/12 on the 8k natural sweep with a reworded needle, against
        # 12/12 when the wording matches.
        #
        # So an unmarked block reserves its BEST run and no more: enough to keep
        # a buried code (typically its block's worst-reconstructed span) whole,
        # bounded so the rest of the budget still goes to the error ranking.
        from native_core.compression.lowrank import _select_residual_rows
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "query_first")
        monkeypatch.setenv("DKV_RESIDUAL_RUN_UNMARKED", "1")
        _toks, scores, runs, ci = self._fixture()
        annotated = [(lo, hi, 0) for lo, hi in runs]   # consulted, said no
        cap = 11
        sel = _select_residual_rows(scores, cap, 0.0, runs=annotated, res_cap=cap)
        rows = set(range(*runs[ci]))
        assert rows <= set(sel.indices.tolist()[:cap]), sel.indices.tolist()

    def test_unmarked_reservation_is_bounded_to_one_run(self, monkeypatch):
        import torch
        from native_core.compression.lowrank import _select_residual_rows
        monkeypatch.setenv("DKV_RESIDUAL_RUN_RESERVE", "query_first")
        monkeypatch.setenv("DKV_RESIDUAL_RUN_UNMARKED", "1")
        # Three separate 4-token codes; unmarked, so only the best may reserve.
        toks = ([' Falcon', '-', '9', '4'] + [' alpha', ' beta', ' gamma']) * 3
        runs = atomic_runs(toks)
        assert len(runs) == 3, runs
        scores = torch.tensor([9.0 if (i % 7) < 4 else 1.0 for i in range(len(toks))])
        annotated = [(lo, hi, 0) for lo, hi in runs]
        sel = _select_residual_rows(scores, 20, 0.0, runs=annotated, res_cap=20)
        front = sel.indices.tolist()[:4]
        lo = min(front)
        assert sorted(front) == list(range(lo, lo + 4)), front
        # and only ONE run was reserved, not all three
        starts = {r[0] for r in runs}
        assert lo in starts

    def test_rank_runs_by_query_annotates_even_when_nothing_matches(self):
        toks = [' alpha'] * 8 + CODE + [' beta'] * 8
        ids = list(range(700, 700 + len(toks)))
        runs = atomic_runs(toks)
        out = rank_runs_by_query(toks, ids, [999999], runs)   # no hit anywhere
        assert all(len(r) == 3 and r[2] == 0 for r in out), out

    def test_absent_query_returns_bare_spans(self):
        toks = [' alpha'] * 8 + CODE + [' beta'] * 8
        runs = atomic_runs(toks)
        out = rank_runs_by_query(toks, list(range(len(toks))), None, runs)
        assert all(len(r) == 2 for r in out), out
