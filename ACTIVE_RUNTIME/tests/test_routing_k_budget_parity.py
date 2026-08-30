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


def test_native_k_is_clamped_to_the_blocks_that_exist():
    """Both native entry points clamp srl_k_keep AFTER their raise-only gates.

    THE DEFECT. main.cpp and batch_engine.cpp each carry three gates that can raise
    srl_k_keep, and every one of them is raise-only. The "n_slots <= 32" gate keys
    off the context WINDOW (n_ctx / micro_block_size) rather than the blocks the
    prompt produced, and n_slots is only ever grown to fit a long prompt, never
    shrunk for a short one -- so at the shipped micro_block_size = 1024 that gate
    fires on essentially every request and pins K at n_slots. At L=8192 that was
    K=32 for 7 blocks, and in main.cpp srl_k_keep sizes a real allocation
    (cache_routed_cap = srl_k_keep * (micro_block_size + 1)).

    WHY A CLAMP AND NOT A RE-KEYED GATE. Re-keying the n_slots gate off the block
    count fixes only the no-preset case. Under DKV_PRESET, n_slots is already <= 16
    so that gate never fired, and the binding overshoot is instead
    adaptive_k_min = max(20, ...) -- which can exceed n_slots outright, i.e. keep
    more blocks than the pool has slots for. One clamp after all three covers both,
    which is why the assertion below also requires the n_slots term.

    WHY BOTH FILES. These are two entry points into the same engine and they have
    diverged before: batch_engine.cpp defaulted micro_block_size to 256 while
    main.cpp used 1024, so every batch-served session silently ran the block size
    linkbench scores 9/24 on. Nothing at runtime reconciles them, so the pin is here.

    THE GROWTH TERM MUST COME FROM THE GENERATION BUDGET, not from the pool's
    headroom_slots. headroom_slots looks like the right quantity and is not: it is
    capped at 512 tokens and does not bound how many blocks generation can create
    (generation runs until active_slot >= n_slots). At the default DKV_MAX_TOKENS
    of 2048 with micro_block_size=1024 that is 2 blocks against a headroom of 1, so
    K would fall one block short mid-answer. It was written that way first and
    caught by reading the clamp's own log line -- the NIAH suite sets
    DKV_MAX_TOKENS=40, where both forms give 1 and therefore agree.
    """
    for path in (MAIN_CPP, BATCH_ENGINE):
        name = os.path.basename(path)
        src = _strip_comments(_read(path))

        m = re.search(r"k_ceiling\s*=\s*std::min\(([^;]+)\);", src)
        assert m, (f"{name} has no k_ceiling clamp. srl_k_keep must be bounded by "
                   f"the blocks that can exist; without it the raise-only gates "
                   f"pin K at the context window.")
        expr = " ".join(m.group(1).split())
        assert "n_slots" in expr, (
            f"{name} clamps K as std::min({expr}) with no n_slots term. "
            f"adaptive_k_min = max(20, ...) can exceed the pool's slot count, so "
            f"the block-count term alone still permits K > pool size.")
        assert "n_comp_blocks" in expr, (
            f"{name} clamps K as std::min({expr}) without the compressed-block "
            f"count -- that count is the whole point of the clamp.")

        gm = re.search(r"int\s+growth\s*=\s*\(([^;]+)\)\s*/", src)
        assert gm, f"{name} has no growth term feeding the clamp"
        growth = " ".join(gm.group(1).split())
        assert "headroom_slots" not in growth, (
            f"{name} derives the clamp's growth from headroom_slots ({growth!r}). "
            f"That is capped at 512 tokens and does not bound blocks created during "
            f"generation, so K falls short mid-answer whenever the generation budget "
            f"exceeds one block. Derive it from the generation budget instead.")
        assert ("max_generate" in growth or "max_tokens" in growth), (
            f"{name} derives the clamp's growth from {growth!r}; it must come from "
            f"the request's generation budget.")

        assert re.search(r"srl_k_keep\s*=\s*k_ceiling\s*;", src), \
            f"{name} computes k_ceiling but never assigns it to srl_k_keep"

        # ORDER IS THE WHOLE FIX. A clamp placed before the gates is dead code:
        # they would simply raise K again afterwards, and nothing would fail.
        last_raise = max(mm.start() for mm in
                         re.finditer(r"srl_k_keep\s*=\s*(?:target_k|adaptive_k)\s*;", src))
        clamp_at = re.search(r"srl_k_keep\s*=\s*k_ceiling\s*;", src).start()
        assert clamp_at > last_raise, (
            f"{name} clamps srl_k_keep BEFORE its last raise-only gate, so the "
            f"gate re-raises K afterwards and the clamp does nothing.")


def test_native_clamp_lands_before_every_allocation_that_reads_k():
    """main.cpp only: the clamp must precede the buffers srl_k_keep sizes.

    srl_k_keep sizes cache_routed_cap, native_dup_tri [K,K] and native_attn_slots
    [K]. A clamp that ran after any of them would leave a buffer sized off the
    pre-clamp value while the decode loop indexed with the post-clamp one -- a
    disagreement far worse than the over-allocation it was meant to fix.
    """
    src = _strip_comments(_read(MAIN_CPP))
    clamp_at = re.search(r"srl_k_keep\s*=\s*k_ceiling\s*;", src)
    assert clamp_at, "main.cpp has no k_ceiling clamp"
    clamp_at = clamp_at.start()
    for consumer in (r"cache_routed_cap\s*=", r"native_dup_tri\s*=\s*ggml_new_tensor_2d",
                     r"native_attn_slots\s*=\s*ggml_new_tensor_1d"):
        m = re.search(consumer, src)
        assert m, f"consumer {consumer!r} not found in main.cpp"
        assert m.start() > clamp_at, (
            f"main.cpp sizes {consumer!r} at offset {m.start()} but clamps "
            f"srl_k_keep later, at {clamp_at}. The buffer would be sized from the "
            f"unclamped K.")


def test_host_slots_tensor_is_sized_after_the_gates_mutate_k_host():
    """Both native entry points must create host_slots_decode AFTER the gates.

    THE DEFECT (batch_engine.cpp, fixed alongside the K clamp). The gates raise
    srl_k_semantic to 3x srl_k_keep and then recompute
    srl_k_host = 1 + recency + lexical + semantic + graph. host_slots_decode is a
    tensor whose LENGTH is srl_k_host, and batch_engine.cpp created it before that
    block while main.cpp created it after -- so on the batch-served path the tensor
    was sized with the pre-gate value and written with the post-gate one.

    Concretely at micro_block_size = 1024, n_ctx = 32768: n_slots = 32, so
    srl_k_semantic starts at 32 and srl_k_host at 57. The "n_slots <= 32" gate then
    raises srl_k_keep to 32, sem_floor2 to 96 and srl_k_host to 121, and the decode
    loop's ggml_backend_tensor_set writes `srl_k_host * sizeof(int32_t)` -- 121
    int32s into a 57-int32 tensor. It is an overflow of the backend buffer, not a
    merely wasteful size, which is why this is pinned rather than left to review.

    The length is exact and not a bound, so a "write less than capacity" reading
    does not save it: route_decode_slots both pads and caps its result to
    srl_k_host (kv_runtime_manager.cpp ~290-297).

    NOT caught by the existing decode-loop path: that loop re-creates the tensor
    when the pool grows, using the post-gate value, so the bug only appears when
    the pool does NOT grow -- the ordinary case rather than the rare one.
    """
    for path in (MAIN_CPP, BATCH_ENGINE):
        name = os.path.basename(path)
        src = _strip_comments(_read(path))

        creates = [m.start() for m in re.finditer(
            r"host_slots_decode\s*=\s*ggml_new_tensor_1d\([^;]*srl_k_host\s*\)", src)]
        assert creates, f"{name} never sizes host_slots_decode from srl_k_host"

        mutations = [m.start() for m in re.finditer(
            r"srl_k_host\s*=\s*1\s*\+\s*srl_k_recency", src)]
        # The first assignment is the initial definition, not a gate mutation; the
        # ones inside the gate block are what move the value out from under the
        # tensor. If a file ever has only the initial one, there is nothing to order.
        gate_mutations = [o for o in mutations if o != min(mutations)]
        if not gate_mutations:
            continue

        first_create = min(creates)
        last_mutation = max(gate_mutations)
        assert first_create > last_mutation, (
            f"{name} creates host_slots_decode at offset {first_create} but "
            f"srl_k_host is still raised afterwards at {last_mutation}. The tensor "
            f"is then sized with a stale, SMALLER length than the "
            f"ggml_backend_tensor_set that fills it, which overflows the backend "
            f"buffer. Create it after the gate block, as src/main.cpp does.")


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
