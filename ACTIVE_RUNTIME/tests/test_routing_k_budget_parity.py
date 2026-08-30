"""K is a routed-TOKEN budget, and every site that derives it must agree.

WHAT THIS PINS
--------------
`max(16, 4096 // block_size)` appeared in three places -- CUDA's
KVRuntimeManager, CUDA's query_router, and the MLX wrapper -- and two separate
things were wrong with it.

1. THE DIVISOR (fixed on both runtimes). CUDA divided by `block_size + 1`, the
   payload PLUS the anchor row, so 4096//257 = 15 at the 256-token block. The
   `max(16, ...)` existed only to round that 15 back up. Dividing by the token
   payload gives 16 directly, so the floor no longer papers over an off-by-one.

2. THE FLOOR VALUE (runtime-specific, and deliberately NOT unified -- see below).
   The floor only binds when the division returns < 16, i.e. only for block sizes
   ABOVE 256, so it clamped exactly where the shipped default had just moved to:

       block   4096//block   max(16,..)   routed tokens
        256        16            16          4096   ok
       1024         4            16         16384   4x the budget

WHY THE TWO RUNTIMES USE DIFFERENT FLOORS
-----------------------------------------
This asymmetry is measured, not an oversight, and this file exists partly to
stop someone "tidying" it away. K trades synthesis quality against the DECODE
CACHE it sizes, so a smaller K only pays where that cache is actually allocated:

  * MLX -- decode cache ON. K=16 -> 4 at block 1024 left linkbench at 24/24
    (= dense) and cut the decode cache 377.9 -> 151.1 MB. Floor 2.
  * CUDA -- decode cache OFF by default. `DKV_DECODE_CACHE_CUDA` defaults to
    "0" (runtime/dkv_attention.py); the `DKV_DECODE_CACHE=1` that
    serving/decode_config.py sets is read only by the MLX wrapper. So on CUDA a
    smaller K pays the quality cost and collects nothing. Floor 16.

CUDA measurement, 2026-08-30, RTX 4070 SUPER, Qwen3.5-2B, micro_block_size=1024,
observed_block_span=1024:

    metric                          K=4        K=16      verdict
    peak VRAM @16k                  4213.0 MB  4213.0 MB identical
    decode ms/tok @32k (paired)     61.06      63.10     CI contains 0
    needle recall, 3 ctx x 3 depth  9/9        9/9       unchanged
    synthesis (paired, n=8)         27.9       44.2      -16.2, CI [-29.1,-3.4]

No memory saved, no time saved, 16 synthesis points lost. K=8 was also measured
(38.8; +10.8 over K=4 resolved, -5.4 vs K=16 NOT resolvable at n=8) and is the
knee to use if the CUDA decode cache is ever enabled.

WHY SOURCE-LEVEL ASSERTIONS
---------------------------
The MLX site cannot be imported without `mlx`, which does not exist on the CUDA
box, and the CUDA sites compute their K inside objects that need a GPU and a
model to construct. Reading the literals out of the source needs neither, so this
runs in ordinary CI -- which is where a drift between the sites would otherwise
sit unnoticed until it showed up as an unexplained number in someone's benchmark.

Same shape as tests/test_quant_alias_parity.py, and the same reason: the defect
was entry points carrying their own copy of one constant.
"""
import ast
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ACTIVE, ".."))

KV_RUNTIME = os.path.join(ACTIVE, "native_core", "kv_runtime_manager.py")
QUERY_ROUTER = os.path.join(ACTIVE, "native_core", "srl", "query_router.py")
MLX_WRAPPER = os.path.join(ACTIVE, "serving", "mlx_dkv_wrapper.py")
MAIN_CPP = os.path.join(REPO, "dkv_native", "src", "main.cpp")
BATCH_ENGINE = os.path.join(REPO, "dkv_native", "serving", "batch_engine.cpp")

# K * block_size = the context attention sees per decode step. 4096 = 16 * 256,
# i.e. the K=16 every pre-1024 measurement was taken at. SHARED by both runtimes.
ROUTED_TOKEN_BUDGET = 4096

# Floors differ on purpose -- see the module docstring for the measurement.
CUDA_K_FLOOR = 16
MLX_K_FLOOR = 2


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(src):
    """Live code only. All these files DESCRIBE old formulas in prose, and that
    prose is the record of why they went; matching it would be a false positive."""
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())


def _floor_and_budget(path, pattern):
    """Pull (floor, budget) out of a `max(FLOOR, BUDGET // ...)` site."""
    m = re.search(pattern, _read(path))
    assert m, f"K expression not found in {os.path.basename(path)}; if it was " \
              f"refactored, update this test rather than deleting it"
    return int(m.group(1)), int(m.group(2))


def test_cuda_runtime_manager_floor_and_budget():
    got = _floor_and_budget(
        KV_RUNTIME,
        r"routing_topk_default\s*=\s*max\(\s*(\d+)\s*,\s*(\d+)\s*//")
    assert got == (CUDA_K_FLOOR, ROUTED_TOKEN_BUDGET)


def test_cuda_query_router_floor_and_budget():
    got = _floor_and_budget(
        QUERY_ROUTER,
        r"_pool_default\s*=\s*max\(\s*(\d+)\s*,\s*(\d+)\s*//")
    assert got == (CUDA_K_FLOOR, ROUTED_TOKEN_BUDGET)


def test_the_two_cuda_sites_agree():
    """Two consumers, one quantity. They have diverged before."""
    a = _floor_and_budget(
        KV_RUNTIME,
        r"routing_topk_default\s*=\s*max\(\s*(\d+)\s*,\s*(\d+)\s*//")
    b = _floor_and_budget(
        QUERY_ROUTER,
        r"_pool_default\s*=\s*max\(\s*(\d+)\s*,\s*(\d+)\s*//")
    assert a == b, (f"kv_runtime_manager derives K as {a} but query_router as "
                    f"{b}; the pool default and the router that reads it must "
                    f"agree or K depends on which one ran last.")


def test_cuda_router_divides_by_the_payload_not_the_anchor():
    """The divisor is the block's token PAYLOAD, not payload + anchor row.

    This half of the fix is invisible in the floor/budget literals. Dividing by
    `_span + 1` gives 4096//257 = 15 at the 256-token block, and the floor then
    silently means "round 15 up" rather than "never route fewer than 16 blocks".
    """
    m = re.search(r"_pool_default\s*=\s*max\(\s*\d+\s*,\s*\d+\s*//\s*([^)\n]+)\)",
                  _read(QUERY_ROUTER))
    assert m, "_pool_default expression not found in query_router"
    divisor = m.group(1).strip()
    assert divisor == "_span", (
        f"query_router divides the token budget by {divisor!r}; it must be the "
        f"raw span. `_span + 1` reintroduces the off-by-the-anchor-row that the "
        f"floor was compensating for, which is what made the floor look load-"
        f"bearing for the wrong reason.")
    assert ROUTED_TOKEN_BUDGET // 257 == 15      # what made the floor look needed
    assert ROUTED_TOKEN_BUDGET // 256 == 16      # what the payload gives directly


def test_mlx_wrapper_floor_and_budget():
    """MLX names its two constants; assert the names carry the right values."""
    found = {}
    for node in ast.walk(ast.parse(_read(MLX_WRAPPER))):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("_ROUTED_TOKEN_BUDGET", "_K_MIN")
                and isinstance(node.value, ast.Constant)):
            found[node.targets[0].id] = node.value.value
    assert found.get("_ROUTED_TOKEN_BUDGET") == ROUTED_TOKEN_BUDGET, found
    assert found.get("_K_MIN") == MLX_K_FLOOR, found


def test_runtimes_share_the_budget_even_though_floors_differ():
    """The BUDGET is the cross-runtime invariant; only the floor is per-runtime.

    If someone changes 4096 on one side to chase a number, the two runtimes stop
    routing the same amount of context from the same block size, which is the
    class of divergence this whole file exists to catch.
    """
    cuda = _floor_and_budget(
        KV_RUNTIME,
        r"routing_topk_default\s*=\s*max\(\s*(\d+)\s*,\s*(\d+)\s*//")[1]
    mlx = None
    for node in ast.walk(ast.parse(_read(MLX_WRAPPER))):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_ROUTED_TOKEN_BUDGET"
                and isinstance(node.value, ast.Constant)):
            mlx = node.value.value
    assert cuda == mlx == ROUTED_TOKEN_BUDGET, (cuda, mlx)


def test_block_256_is_unchanged_on_both_runtimes():
    """Every pre-existing measurement was taken at block 256. Both runtimes must
    still produce K=16 there, or that whole body of results is invalidated."""
    assert max(CUDA_K_FLOOR, ROUTED_TOKEN_BUDGET // 256) == 16
    assert max(MLX_K_FLOOR, ROUTED_TOKEN_BUDGET // 256) == 16


def test_floors_bind_where_the_measurement_says_they_should():
    """At the shipped block_size=1024 the two runtimes deliberately diverge."""
    assert max(CUDA_K_FLOOR, ROUTED_TOKEN_BUDGET // 1024) == 16, \
        "CUDA must stay at 16: a smaller K there costs 16.2 synthesis points " \
        "and saves 0 MB, because DKV_DECODE_CACHE_CUDA defaults to 0"
    assert max(MLX_K_FLOOR, ROUTED_TOKEN_BUDGET // 1024) == 4, \
        "MLX must take the budget K: its decode cache is real, and K=4 cut it " \
        "377.9 -> 151.1 MB at no measured retrieval cost"


def test_both_runtimes_ceil_the_adaptive_fraction():
    """DKV_TOPK_FRAC must round the SAME way on both runtimes.

    It used int() on both, so frac=0.3 with nb=13 gave 3 where ceil gives 4.
    They were equal then and are equal now; the risk is one being changed alone,
    which would make one env var route different block counts per runtime.
    """
    cuda = re.search(r"k_eff\s*=\s*max\(topk,\s*([\w.]+)\(topk_frac\s*\*\s*N\)\)",
                     _read(QUERY_ROUTER))
    assert cuda and cuda.group(1) == "math.ceil", \
        f"query_router k_eff rounding is {cuda and cuda.group(1)!r}, not math.ceil"

    mlx = re.search(
        r"return max\(self\.topk_blocks,\s*([\w.]+)\(nb\s*\*\s*self\.topk_frac\)\)",
        _read(MLX_WRAPPER))
    assert mlx and mlx.group(1) == "math.ceil", \
        f"mlx _route_k rounding is {mlx and mlx.group(1)!r}, not math.ceil"

    # And the rounding they agree on is the one that motivated the change.
    assert math.ceil(0.3 * 13) == 4 and int(0.3 * 13) == 3


def test_native_entry_points_agree_on_micro_block_size():
    """main.cpp and batch_engine.cpp each carry their own default.

    They diverged -- 1024 vs 256 -- so every BATCH-SERVED native session ran at
    the block size linkbench scores 9/24 on, against 24/24 = dense at 1024,
    while single-stream ran 1024. Nothing reconciled them.
    """
    def declared(path):
        m = re.search(r"int\s+micro_block_size\s*=\s*(\d+)\s*;",
                      _strip_comments(_read(path)))
        assert m, f"micro_block_size declaration not found in {path}"
        return int(m.group(1))

    main_mbs = declared(MAIN_CPP)
    batch_mbs = declared(BATCH_ENGINE)
    assert main_mbs == batch_mbs, (
        f"main.cpp defaults micro_block_size to {main_mbs} but "
        f"batch_engine.cpp to {batch_mbs}. These are two entry points into the "
        f"same engine; a divergence here is invisible at runtime.")
    assert main_mbs == 1024


def test_python_runtime_manager_default_matches_native():
    """KVRuntimeManager's signature default is the same 1024.

    It was still 256 -- so any caller that did not pass one explicitly got the
    bad block size. The production caller passes 1024, which is exactly why it
    went unnoticed.
    """
    for node in ast.walk(ast.parse(_read(KV_RUNTIME))):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args
            names = [a.arg for a in args.args]
            if "micro_block_size" not in names:
                continue
            offset = len(names) - len(args.defaults)   # defaults align to the tail
            default = args.defaults[names.index("micro_block_size") - offset]
            assert isinstance(default, ast.Constant) and default.value == 1024, \
                f"KVRuntimeManager micro_block_size default is {ast.dump(default)}"
            return
    raise AssertionError("KVRuntimeManager.__init__ with micro_block_size not found")
