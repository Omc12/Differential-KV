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
