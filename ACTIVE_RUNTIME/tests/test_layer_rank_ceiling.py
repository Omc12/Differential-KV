"""`base_rank` is a CEILING, and making it one changed no delivered rank.

Until 2026-08-31 the per-layer schedule's middle band returned `1.5 * base_rank`,
so the configured number was not a ceiling and nothing delivered what it
declared: `mid` asked for 64 and stored 32-96, `high` asked for 128 and stored
64-192 -- it had never once delivered its stated rank.

The fix was a REPARAMETERISATION, not a fidelity change: every multiplier was
divided by 1.5 (0.75/1.50/0.50 -> 0.50/1.00/0.333) and every preset rank
multiplied by 1.5 (low 32->48, mid/ultra 64->96, high 128->192). The delivered
per-layer ranks are therefore bit-identical to the old schedule.

This test pins BOTH halves of that claim, because either one failing silently is
a fidelity change nobody asked for:

  1. equivalence -- new(1.5 * b) == old(b), layer by layer, for every base and
     model depth that matters;
  2. the ceiling property itself -- no band may ever exceed base_rank.

It also guards the direction of travel: if someone "restores" a 1.5x band, (2)
fails immediately rather than three sessions later when a pool banner looks odd.
"""
import pytest

from native_core.kv_runtime_manager import get_layer_rank


def old_schedule(layer_idx: int, num_layers: int, base_rank: int) -> int:
    """The pre-2026-08-31 schedule, verbatim, as the equivalence reference."""
    ratio = layer_idx / max(num_layers, 1)
    if ratio < 0.25:
        return max(8, round(0.75 * base_rank))
    elif ratio < 0.75:
        return round(1.5 * base_rank)
    return max(8, round(0.50 * base_rank))


# (old base, new base) -- the presets, plus small/odd bases for the floors.
BASES = [(32, 48), (64, 96), (128, 192), (16, 24), (8, 12), (96, 144)]
DEPTHS = [12, 24, 28, 32, 48, 64]


@pytest.mark.parametrize("old_base,new_base", BASES)
@pytest.mark.parametrize("num_layers", DEPTHS)
def test_rescale_preserved_every_delivered_rank(old_base, new_base, num_layers):
    """new(1.5*b) must equal old(b) at every layer -- no fidelity change."""
    for layer in range(num_layers):
        got = get_layer_rank(layer, num_layers, new_base)
        want = old_schedule(layer, num_layers, old_base)
        assert got == want, (
            f"layer {layer}/{num_layers}: base {new_base} gives {got}, "
            f"but the old schedule at base {old_base} gave {want}. "
            f"The rescale was supposed to be behaviour-preserving."
        )


@pytest.mark.parametrize("base_rank", [8, 12, 16, 24, 32, 48, 64, 96, 128, 192])
@pytest.mark.parametrize("num_layers", DEPTHS)
def test_base_rank_is_a_real_ceiling(base_rank, num_layers):
    """No band may return more than base_rank. This is the whole point."""
    for layer in range(num_layers):
        got = get_layer_rank(layer, num_layers, base_rank)
        # The floor (max(8, ...)) can legitimately lift a tiny base above itself;
        # exempt only that case, which is a minimum-fidelity guard, not a boost.
        if base_rank >= 8:
            assert got <= base_rank, (
                f"layer {layer}/{num_layers} at base {base_rank} returned {got}"
                f" -- above the ceiling. Did a 1.5x band come back?"
            )


@pytest.mark.parametrize("num_layers", DEPTHS)
def test_middle_band_uses_the_whole_ceiling(num_layers):
    """The middle band should reach base_rank exactly -- not less, not more.

    Guards against a "fix" that makes rank a ceiling by simply shrinking every
    band, which would be a silent fidelity cut rather than a reparameterisation.
    """
    base = 96
    delivered = [get_layer_rank(l, num_layers, base) for l in range(num_layers)]
    assert max(delivered) == base, (
        f"max delivered rank {max(delivered)} != base {base}; the middle band "
        f"should consume the full ceiling."
    )


def test_adaptive_off_returns_base_rank(monkeypatch):
    """With the schedule disabled, rank has always meant rank. Keep it that way."""
    monkeypatch.setenv("DKV_LAYER_ADAPTIVE_RANK", "0")
    for base in (32, 48, 64, 96):
        assert get_layer_rank(5, 28, base) == base


def test_presets_declare_the_rescaled_ceiling():
    """The preset values must move WITH the multipliers or delivery changes."""
    from native_core.config import DKVConfig
    expected = {"low": 48, "mid": 96, "ultra": 96, "high": 192}
    for preset, want in expected.items():
        cfg = DKVConfig({"preset": preset})
        assert cfg.rank == want, (
            f"preset {preset} declares rank {cfg.rank}, expected {want}. "
            f"Preset ranks and schedule multipliers must move together: "
            f"multipliers were divided by 1.5, so presets were scaled by 1.5."
        )


# ── The coupling that actually broke ─────────────────────────────────────────
# The r_proj cap defaults on only for presets at or below a base-rank threshold
# (`high` is excluded: capping rank 192 to 32 is a 6x cut nobody measured).
# That threshold is compared against the PRESET RANK -- so the 2026-08-31
# rescale, which multiplied every preset rank by 1.5, silently pushed mid and
# ultra past a threshold still written as 64 and uncapped the default path.
# Caught in review, but only by chance. These pin the relationship.
CAP_MAX_BASE_RANK = 96          # must equal the literal in BOTH files below


def _cap_thresholds_in_source():
    """Read the guard literal out of both files that implement it."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    found = {}
    for rel in ("native_core/compression/lowrank.py",
                "native_core/kv_runtime_manager.py"):
        src = (root / rel).read_text(encoding="utf-8")
        m = re.findall(r"_(?:max_rproj|pool_rproj_cap) = 32 if \("
                       r"isinstance\([^)]*\) and [\w.]+ <= (\d+)\)", src)
        assert m, f"could not find the cap guard in {rel}"
        found[rel] = {int(x) for x in m}
    return found


def test_cap_guard_threshold_matches_in_both_files():
    """Compress and pool sites must agree, or the pool is sized for a rank the
    compressor never produces."""
    found = _cap_thresholds_in_source()
    values = set().union(*found.values())
    assert values == {CAP_MAX_BASE_RANK}, (
        f"cap guard thresholds disagree or drifted: {found}; "
        f"expected all == {CAP_MAX_BASE_RANK}"
    )


def test_default_cap_covers_low_mid_ultra_but_not_high():
    """The threshold must track the preset ranks. If presets are rescaled again
    without moving it, this fails instead of silently uncapping production."""
    from native_core.config import DKVConfig
    expected = {"low": True, "mid": True, "ultra": True, "high": False}
    for preset, should_cap in expected.items():
        rank = DKVConfig({"preset": preset}).rank
        assert (rank <= CAP_MAX_BASE_RANK) is should_cap, (
            f"preset {preset} has rank {rank}; expected cap-by-default="
            f"{should_cap} against threshold {CAP_MAX_BASE_RANK}"
        )
