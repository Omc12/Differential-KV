"""runtime_summary() must report the GPU compress path's block count.

`total_compressions` is incremented only in _postprocess_compressed_block,
which is on the CPU compress path. config.gpu_compress defaults to
`not is_macos`, so on CUDA every block goes through compress_layer_blocks_gpu
instead and that counter stays 0 -- runtime_summary() reported "0 compressions,
0.0 MB saved" for a fully compressed session, and an external benchmark harness
read that as DKV never engaging.

Measured end-to-end on an RTX 4070 SUPER (Qwen2.5-0.5B-Instruct, 9015 tokens,
preset mid) before the fix: streaming stats total_blocks_created=216,
total_compressed=192, 192 blocks in state COMPRESSED -- while
runtime_summary()["total_compressions"] read 0.

This is the same streaming-store-vs-manager-store split the `sessions` property
docstring records; runtime_summary was the remaining unfixed instance.
"""
import sys, os, types

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

from native_core.kv_runtime_manager import KVRuntimeManager


def _fake(total_compressions, streaming_total):
    """Minimal stand-in exposing only what runtime_summary() reads."""
    f = types.SimpleNamespace()
    f.pager = types.SimpleNamespace(summary=lambda: {})
    f._compressor = types.SimpleNamespace(summary=lambda: {})
    f.total_cosine_sim = 0.99 * total_compressions
    f.total_norm_drift = 0.01 * total_compressions
    f.total_compressions = total_compressions
    f.vram_saved_bytes = 0
    f.rank = 64
    f.rank_histogram = {}
    f.session_blocks = {"s": {}}
    f._streaming_mgr = (
        None if streaming_total is None
        else types.SimpleNamespace(stats={"total_compressed": streaming_total})
    )
    return f


def test_gpu_path_count_is_reported():
    # CUDA shape: GPU path compressed 192, CPU counter never fired.
    s = KVRuntimeManager.runtime_summary(_fake(0, 192))
    assert s["blocks_compressed"] == 192, s["blocks_compressed"]
    assert s["total_compressions"] == 0      # unchanged, still CPU-path only
    assert s["quality_sampled"] == 0
    assert s["vram_saved_measured"] is False


def test_cpu_fallback_is_not_double_counted():
    # stats["total_compressed"] is bumped by the GPU branch AND both CPU
    # fallbacks, so it is the whole total. Summing it with total_compressions
    # would count every fallback block twice.
    s = KVRuntimeManager.runtime_summary(_fake(30, 192))
    assert s["blocks_compressed"] == 192, s["blocks_compressed"]
    assert s["quality_sampled"] == 30


def test_non_streaming_path_unchanged():
    # MPS/CPU: no streaming manager, total_compressions is the only count.
    s = KVRuntimeManager.runtime_summary(_fake(45, None))
    assert s["blocks_compressed"] == 45
    assert s["quality_sampled"] == 45
    assert s["vram_saved_measured"] is False


def test_quality_averages_keep_the_cpu_denominator():
    # Folding GPU blocks into the denominator would drag avg_cosine_sim toward
    # 0 and read as a fidelity collapse -- a plausible wrong number is worse
    # than a visible zero.
    s = KVRuntimeManager.runtime_summary(_fake(30, 192))
    assert abs(s["avg_cosine_sim"] - 0.99) < 1e-6, s["avg_cosine_sim"]
