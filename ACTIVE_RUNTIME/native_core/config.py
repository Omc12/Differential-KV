import os
from typing import Dict, Any

class DKVConfig:
    def __init__(self, config_dict: Dict[str, Any] = None):
        config_dict = config_dict or {}

        # 1. Detect preset
        # Preset can be specified in config_dict['preset'] or environment variable DKV_PRESET.
        # Default is "mid".
        preset = config_dict.get("preset", os.environ.get("DKV_PRESET", "mid")).lower()
        if preset not in ("low", "mid", "high", "ultra"):
            preset = "mid"
        self.preset = preset

        import sys
        is_macos = (sys.platform == "darwin")

        # Base defaults for residual quantization & rarity selection
        self.residual_quant = "none"
        self.residual_quant_group_size = 64
        self.residual_quant_bits = 4
        self.rarity_capture = True
        self.rarity_weight = 1.5
        self.rarity_min_idf = 2.0
        self.boost_digits = 20.0
        self.boost_owner = 14.6
        self.boost_rare = 7.3

        # Apply preset defaults
        # NOTE on CUDA prefill_chunk_size: ingest_chunk creates full blocks of exactly
        # (1 + micro_block_size) tokens — 1 anchor + micro_block_size active keys.
        # micro_block_size defaults to 1024, so block_capacity = 1025.
        # prefill_chunk_size MUST accommodate at least one full block so that at least one
        # full block is produced per inner chunk.  On macOS/MPS the chunk size is already
        # small (256) because MLX handles compression differently; on CUDA we need larger
        # chunks or every chunk produces only a partial block and nothing is ever compressed.
        if self.preset == "low":
            self.decode_cache_enabled = False
            self.decode_cache_max_tokens = 0
            # CUDA: 1024 ensures ≥3 full blocks (3×257=771 < 1024) per inner chunk.
            # macOS/MLX keeps 256 (handled post-forward by compress_deferred_prefill_blocks).
            self.prefill_chunk_size = 256 if is_macos else 1024
            self.srl_threshold = 30
            self.async_svd = False if is_macos else True
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0  # MLX parity: pure relevance, no recency bias (see override note)
            self.kv_quant = "q4_0"
            self.max_active_dense_tokens = 1024
            # Residual budget per block — see the "mid" branch for the full
            # rationale.  `low` is memory-priority.
            #
            # CAUTION: the old justification here ("40 already covers the
            # adaptive prose cap int(0.15*256)=38, so it loses essentially
            # nothing"), and the 13.4K A/B that "confirmed res40 matched res128
            # quality", are both INVALID.  They were measured while the budget
            # was bugged to start from int(0.15*n)=38 (handoff §10j), which
            # capped 40 and 128 to the same 38 exact tokens — of course they
            # matched.  With that cap removed this ladder is real for the first
            # time, and 40 now genuinely stores ~3x fewer exact tokens per block
            # than `high`.
            #
            # RE-MEASURED, on a metric that can actually resolve it. linkbench at
            # 32k over 24 seeds, `low` with max_residual 40 vs 128: 18/24 BOTH
            # WAYS. So the residual budget is not what limits `low` -- 40 stays,
            # and it is now measured rather than assumed.
            #
            # CONFIRMED AGAIN on `mid`, at 48 seeds: 21/48 at residual 128 and
            # 21/48 at 40, on Qwen3.5-2B at 32k. Three independent measurements
            # now say the exact-token budget is inert on prose, so `mid` is
            # paying for 128 of them. The untested case is the one this ladder
            # was written for -- table- and digit-dense documents, where the
            # residuals carry exactly what the SVD reconstructs worst -- so
            # lowering `mid` wants that measured first, not just these two.
            #
            # What DOES limit `low` is its energy target: 0.999 gives a realised
            # mean per-block rank of 35 against mid's 53. That is the memory-for-
            # fidelity trade the preset exists to make, working as intended.
            #
            # EVERY linkbench SCORE IN THIS FILE IS `QMODE=direct`. Write the
            # mode down next to the number; it was not, and that cost an
            # afternoon.
            #
            # linkbench has two question modes and `chain` (multi-hop) is the
            # DEFAULT. `direct` names the intermediate entity outright, which
            # collapses the chain to one lookup over the same context. Re-run in
            # `chain` every arm roughly halves:
            #
            #                   direct   chain
            #   rotated          40/48   21/48
            #   unrotated        47/48   23/48
            #   dense            47/48   23/48
            #
            # The DENSE arm halves too, and dense shares no DKV code, so this was
            # briefly read as an environment shift with a transformers/torch
            # update as the suspect. It was not: packages are unchanged since
            # 2026-08-10 (before these were recorded), the transformers 5.14.1
            # bump predates them by three weeks, and the harness has not been
            # touched since before them. It was two different benchmarks.
            #
            # The ladder's ordering holds in BOTH modes. Run a dense control
            # alongside every time -- it is what distinguished "DKV regressed"
            # from "these are different tasks".
            #
            # 40 -> 64 (2026-08-31), the `low` rung of the ladder f5f96e13
            # announced and only ever applied to MLX. Byte-justified, not
            # quality-justified: see the `mid` branch for the arithmetic and for
            # the four A/Bs that still say this dial is inert. 64 int4 slots cost
            # 4,608 B per (block, kv-head) against the 10,240 B this preset paid
            # at 40 fp16 slots, so `low` stays the memory-priority rung.
            self.max_residual_tokens = 64
            # Spectral energy a block's low-rank form must retain, and the rank
            # ceiling that serves it. This -- not `rank` -- is what actually sets
            # a block's rank: the compressor keeps the smallest k carrying this
            # fraction of the energy, so raising the ceiling alone does nothing
            # (asking for rank 32 vs 128 moved the real median rank only 24->34).
            #
            # The claim above is CONFIRMED by direct instrumentation: at
            # configured rank 216/224/232 the realised per-block rank is 52-137
            # with mean ~67 in all three cases, i.e. the ceiling binds for 0.0%
            # of blocks. Energy is the dial; see the `ultra` branch for the
            # realised-rank-vs-energy table.
            #
            # These synthesis numbers, however, are SINGLE SEED and inside the
            # +-15-point RSVD-seed band, so they do not establish an ordering:
            #   0.999   / rank 32   -> 30.0   peak_alloc 5.07 GB
            #   0.9999  / rank 64   -> 43.3   peak_alloc 5.16 GB
            #   0.99999 / rank 128  -> 46.7   peak_alloc 5.41 GB
            # The VRAM column is real and deterministic. TTFT is flat across all
            # three (10.45-10.77 s), so the cost is VRAM, not latency, and
            # distractor retrieval stays 24/24 at every setting.
            self.svd_energy = 0.999
            self.rank = 48  # ceiling; was 32 before the 2026-08-31 rescale (schedule multipliers /1.5, preset ranks *1.5 -> delivered ranks unchanged)
        elif self.preset == "high":
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 16384
            self.prefill_chunk_size = 2048
            self.srl_threshold = 100
            self.async_svd = False if is_macos else True  # Disable background async SVD on macOS for MPS stability
            self.mps_watermark = 0.0
            self.torch_compile = False if is_macos else True
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0  # MLX parity: pure relevance, no recency bias (see override note)
            self.kv_quant = "f16"
            self.max_active_dense_tokens = 4096
            # `high` = max fidelity: full 128-residual ceiling (paper config of
            # record, MLX default) for table/factual-dense docs, accepting the
            # larger pool.  See the "mid" branch for the ladder rationale.
            #
            # 128 -> 256 (2026-08-31), the `high` rung of f5f96e13's ladder. This
            # is the one rung that costs more than its pre-int4 self: 256 int4
            # slots = 18,432 B per (block, kv-head) against 128 fp16 slots'
            # 32,768 B, so it is still a saving there, but it is 6.4x the
            # 40-at-int4 floor. `high` is the rung defined as "accept the pool
            # size for fidelity", which is why the extra headroom goes here and
            # not on `mid`. Still no measurement showing it recovers anything.
            self.max_residual_tokens = 256
            # UNROTATED too, from 2026-08-17. `high` is the MAX-FIDELITY rung, and
            # it made no sense for it to be the one preset that still lost 42% of
            # exact digit recall -- the "table/factual-dense docs" this branch is
            # written for are precisely the content the rotated pool damages, and
            # its bigger rank and residual ceiling do not recover any of it
            # (measured: residual 40 vs 128 is 14/24 either way while unrotated is
            # 24/24). A fidelity preset that is beaten by the default on the
            # metric it exists for is a mis-labelled preset.
            #
            # This leaves `low` as the only rotated preset, which is the right
            # shape: rotated is now a SPEED choice, not a fidelity one, so it
            # belongs on the memory/speed rung and on the DKV_ROTATED_POOL=1
            # escape hatch rather than scattered across the ladder.
            self.rotated_pool = False
            # Quality end: keep more of the spectrum. The supporting note here
            # used to be that DKV_RANK_BOOST=auto and DKV_REMAT_CACHE=0 "both
            # leave synthesis at 30.0"; those are single-seed numbers inside the
            # +-15-point RSVD-seed band and are not evidence either way. What
            # survives is the structural half: at this block size K already
            # routes every block, so no routing change can add information the
            # model is not already being shown -- that is visible in the routing
            # code, not inferred from a score.
            #
            # `high` is the top of the ladder for cost-sensitive quality work.
            # `ultra` keeps one more energy rung; see its branch, which also
            # carries the retraction of the rank-sweep numbers this ladder was
            # originally justified with.
            self.svd_energy = 0.99999
            self.rank = 192  # ceiling; was 128 before the 2026-08-31 rescale (schedule multipliers /1.5, preset ranks *1.5 -> delivered ranks unchanged)
        elif self.preset == "ultra":
            # `ultra` is MID's settings with a higher energy target.
            #
            # It was originally "mid with rank 192", justified by a rank sweep
            # that is now retracted (see the correction further down). The
            # settings are still mid's rather than high's, because that is the
            # configuration everything here was measured on -- but the preset now
            # differs from `high` on `svd_energy`, which is the parameter that
            # provably changes what gets stored.
            #
            # WHICH mid setting, bisected one at a time against high's value:
            # prefill_chunk_size, and ONLY that -- srl_threshold 100, kv_quant
            # f16, max_active_dense_tokens 4096 and decode_cache_max_tokens 16384
            # each left the score untouched, while prefill_chunk_size 2048 took
            # it from 60.0 to 33.3.
            #
            # TREAT THAT AS UNPROVEN. It is a single-seed comparison and 60.0 ->
            # 33.3 is inside the +-15-point RSVD-seed band measured in the
            # `ultra` branch. The block-formation mechanism below is real and
            # visible in the code; the SCORE attributed to it is not established,
            # and needs re-running across seeds before it is trusted.
            #
            # The mechanism is block formation, not chunking as such. The wrapper
            # rounds the chunk UP to a multiple of block capacity, so with
            # micro_block_size 1024 (capacity 1025) a 1024 chunk becomes 1025 --
            # exactly ONE block per chunk -- and 2048 becomes 2050, two. Forming
            # two blocks per chunk is what costs the synthesis.
            #
            # It is not a rule that smaller is better, and it interacts with rank:
            # at rank 128, chunk 2048 scores 50.0 and chunk 1024 scores 46.7 --
            # the opposite direction. Do not carry "chunk 1024 is better" over to
            # another rank without re-measuring.
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 4096
            self.prefill_chunk_size = 512 if is_macos else 1024
            self.srl_threshold = 50
            self.async_svd = False if is_macos else True
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0
            self.kv_quant = "q8_0"
            self.max_active_dense_tokens = 2048
            # 128 -> 40, tracking `mid`, which is what this preset is defined as
            # plus the unrotated pool -- see the `mid` branch for the four
            # measurements that showed the residual budget inert. Keeping ultra
            # above mid here would reintroduce exactly the kind of unmeasured
            # extra that 69393023 removed from this preset.
            #
            # 40 -> 128 (2026-08-31), still TRACKING MID rather than taking the
            # 256 that f5f96e13's message assigned `ultra`. That number would
            # break the one-line definition of this preset -- `ultra` is `mid`
            # plus the unrotated pool, nothing else -- and put it above `mid` on
            # a dial measured inert four times, which is the unmeasured extra the
            # paragraph above exists to prevent. If `ultra` should own residual
            # headroom of its own, that needs a measurement first, not a commit
            # message. Deliberate divergence from the announced ladder.
            self.max_residual_tokens = 128
            # ENERGY IS THE DIAL. RANK IS A CEILING THAT USUALLY DOES NOT BIND.
            #
            # This block previously claimed the opposite ("rank is the driver,
            # not energy") off a rank sweep whose every number is now known to be
            # noise. Both halves of that claim were wrong; the corrected version:
            #
            # 1. `rank` is a CEILING on the per-block SVD rank, not the rank.
            #    The compressor keeps the smallest k reaching `svd_energy` and
            #    then clamps to `rank`. Instrumented on Qwen3.5-2B at 16k, the
            #    realised per-block rank at ranks 216/224/232 is 52-137 with mean
            #    ~67 in ALL THREE cases -- the ceiling binds for 0.0% of blocks.
            #    Configuring 216 vs 232 changes nothing about what is stored.
            #
            # 2. `svd_energy` sets it ONLY WHEN THE CONTENT IS LOW-RANK. Which
            #    of the two binds depends on the SPECTRAL RICHNESS OF THE INPUT,
            #    and the numbers below are from repetitive filler:
            #        0.999     -> 35        0.999999   -> 94
            #        0.9999    -> 53        0.9999999  -> 180
            #        0.99999   -> 67
            #    Those were measured on "The archive records a long sequence of
            #    unremarkable events." repeated -- a nearly rank-deficient input,
            #    where the energy target is met far below the ceiling.
            #
            #    ON REAL PROSE THE OPPOSITE HOLDS. Re-measured on the Random
            #    Features paper at 16k, realised MEAN per-block rank tracks the
            #    CEILING and barely moves with energy:
            #        ceiling  64 -> 66.5 (energy 0.999) .. 66.7 (0.999999)
            #        ceiling 128 -> 130.3 .. 133.3
            #        ceiling 224 -> 215.3 .. 233.3
            #    (slightly above the ceiling because get_layer_rank boosts early
            #    layers.) A real document's spectrum does not decay fast enough to
            #    reach the target under the cap, so the CAP binds and `rank` is
            #    the dial.
            #
            #    This is why an energy A/B on the paper corpus produced BYTE-
            #    IDENTICAL generated text at 0.9999 / 0.99999 / 0.999999 while
            #    pool.U differed at every setting: the stored bytes changed a
            #    little, the rank did not, and the answer did not.
            #
            #    PRACTICAL CONSEQUENCE: on the workloads this system is for, the
            #    preset ladder works through `rank`, not through `svd_energy`.
            #    Do not quote the filler table as though it described documents.
            #
            # 3. What a different `rank` DOES change is r_proj = rank + 5, the
            #    width of the randomised-SVD projection -- so it redraws Omega
            #    and produces a different approximate basis at the same realised
            #    rank. The "rank landscape" was that redraw, not fidelity.
            #
            # THE SYNTHESIS BENCHMARK CANNOT RESOLVE ANY OF THIS. Holding the
            # config fixed at rank 224 and changing ONLY DKV_RSVD_SEED:
            #        seed 0 -> 63.3      seed 1 -> 33.3      seed 2 -> 50.0
            # a 30-point spread from the random draw alone, which is the entire
            # range the rank sweep "found". Over three seeds:
            #        mid   (rank 64)  50.0 / 56.7 / 53.3   mean 53.3
            #        ultra (rank 224) 63.3 / 33.3 / 50.0   mean 48.9
            #        dense                                      60.0
            # so rank 224 is WORSE on average than rank 64 and far less stable,
            # and DKV does not beat dense on synthesis at either. The earlier
            # "ultra beats dense 63.3 vs 60.0" was one lucky seed.
            #
            # MEASURED PROPERLY, AND `ultra` DOES REACH DENSE. Using
            # colab/synthesis_power.py -- replicated over document windows AND
            # SVD seeds, paired against dense, 4 replicates, Qwen3.5-2B at 16k:
            #
            #     dense   mean 61.7  sd 1.9
            #     ultra   mean 63.3  sd 4.7   paired diff +1.67,
            #                                 95% CI [-7.5, +10.9]
            #                                 -> no difference resolvable
            #     mid     mean 45.0  sd 6.4   paired diff vs dense -16.7,
            #                                 95% CI [-28.1, -5.2] -> behind
            #
            # So ultra is at PARITY with dense on synthesis, and mid is genuinely
            # behind it. This is NOT the retracted claim returning: that one was a
            # single seed on a fixed document window, this is replicated, paired
            # and interval-bounded.
            #
            # NEVER quote a single-seed multifact number again. At temperature 0
            # a repeat run is deterministic and proves nothing; the seed is the
            # axis that has to be varied, and a difference under ~15 points is
            # not a difference.
            #
            # So `ultra` is defined on the dial that demonstrably moves the
            # stored representation -- one energy rung above `high` -- and its
            # rank is set to 224 only so the ceiling does not clip that target
            # (realised max at this energy is 205). It is NOT claimed to beat
            # `high` on any benchmark; it is claimed to store more of the
            # spectrum, which is measured and deterministic.
            # `ultra` IS MID PLUS AN UNROTATED POOL, AND NOTHING ELSE.
            #
            # It used to also carry rank 224 and svd_energy 0.999999. Both are
            # removed, because measured against the version without them they
            # bought nothing and cost a great deal. Qwen3.5-2B at 32k,
            # interleaved arms:
            #
            #     with rank 224 / energy 1e-6    8.16 / 8.23 tok/s, 9.22 GB
            #     mid settings + unrotated pool 10.05 / 10.07 tok/s, 6.28 GB
            #
            # 22% of decode and 2.9 GB of device memory, for NO difference on
            # anything measurable: linkbench 47/48 either way, needle sweep clean
            # either way, and synthesis cannot resolve it at all (+-15-point
            # RSVD-seed band).
            #
            # The rank-224 choice came from a sweep later retracted as
            # randomised-SVD projection noise, and the energy rung is nearly
            # inert on real prose because the rank ceiling binds there. Both were
            # carrying cost on evidence that no longer stands.
            #
            # What DOES stand is the unrotated pool, on the one accuracy metric
            # with real power: linkbench at 32k over 48 seeds, 40/48 rotated
            # against 47/48 unrotated -- exactly dense's 47/48. That single
            # change is the whole preset.
            self.svd_energy = 0.9999
            self.rank = 96  # ceiling; was 64 before the 2026-08-31 rescale (schedule multipliers /1.5, preset ranks *1.5 -> delivered ranks unchanged)
            # Store keys UNROTATED. This is the one change in the project that
            # measurably closes a gap to dense on a metric that can actually
            # resolve it -- see the rotated_pool resolution below.
            self.rotated_pool = False
            # Dual-scale is implemented and OFF here too -- see the dual_scale
            # resolution below for the three policies measured and why none of
            # them earns its place. `ultra` was the intended home for it.
            self.dual_scale = False
        else:  # "mid" (Default)
            self.decode_cache_enabled = True
            self.decode_cache_max_tokens = 4096
            # CUDA: 1024 ensures ≥3 full blocks per inner chunk.
            self.prefill_chunk_size = 512 if is_macos else 1024
            self.srl_threshold = 50
            self.async_svd = False if is_macos else True  # Disable background async SVD on macOS for MPS stability
            self.mps_watermark = 0.0
            self.torch_compile = False
            self.approximate_attn = True if is_macos else False
            self.srl_age_penalty = 0.0  # MLX parity: pure relevance, no recency bias (see override note)
            self.kv_quant = "q8_0"
            # Middle of the fidelity ladder -- see the `low` branch for the
            # measurements. 0.9999/64 recovers most of the synthesis that 0.999
            # gives up (30.0 -> 43.3 of the 46.7 that `high` reaches) for a third
            # of its VRAM cost (+0.09 GB against +0.34).
            self.svd_energy = 0.9999
            self.rank = 96  # ceiling; was 64 before the 2026-08-31 rescale (schedule multipliers /1.5, preset ranks *1.5 -> delivered ranks unchanged)
            self.max_active_dense_tokens = 2048
            # Residual budget per block: how many exact (uncompressed) tokens
            # correct the lossy SVD.  This is the main quality dial.
            #
            # It used to be described here as NOT a flat knob, on the grounds
            # that "the compressor caps actual usage at int(0.15*T_active) (=38
            # for T=256) ... so prose blocks never use more than ~38 regardless
            # of this value".  That cap was a BUG, not a design (handoff §10j):
            # the budget started from int(0.15*n) instead of max_residual, so a
            # block could never exceed 38 exact tokens and raising this setting
            # to 128 did literally nothing.  Both compress paths now start from
            # the pool value, and the separate 0.08 error floor that was
            # discarding the budget entirely on ordinary prose is gone (§10k,
            # MLX picks residuals by pure top-k).  The value below is now the
            # real per-block ceiling.
            #
            # The pool allocates this many slots UNIFORMLY per block, so the
            # physical VRAM cost is paid on every block even though most sit
            # mostly empty.  A/B at 13.4K: res128 pool = 2.8 GB, res40 = 1.5 GB,
            # identical output quality on prose synthesis -- which is why `mid`
            # used to be 64.
            #
            # ladder it: `mid` = 64 (covers the prose cap plus boost headroom),
            # `high` = 128 (full table/factual fidelity, accepts the VRAM), and
            # `low` = 40 (memory-priority).  Override with
            # DKV_MAX_RESIDUAL_TOKENS.
            #
            # RAISED TO 128 (2026-07-28) to match MLX, which uses 128 FLAT at
            # every preset (mlx_dkv_wrapper.py: DKV_MAX_RESIDUAL default "128");
            # the 40/64/128 ladder is a CUDA-only invention.
            #
            # Doing so initially produced GARBAGE output, which turned out to be
            # two CUDA-side limits this value reaches and 64 did not:
            #   * the decode kernel's residual scratch was 64 wide while its READ
            #     loops ran to max_residual -- out-of-bounds reads above 64
            #     (fixed: DKV_MAX_RESIDUAL_SHARED in dkv_decode.metal);
            #   * pool sizing divided the budget by a per-slot cost that EXCLUDED
            #     the residual arrays, so the over-allocation grew from 2.2x to
            #     3.3x (fixed in KVRuntimeManager).
            # Neither was a reason to keep 64 -- both were bugs 64 happened to
            # hide. See handoff §9u.
            #
            # LOWERED 128 -> 40 (2026-08-17). Measured inert FOUR times, the last
            # on the content this dial was designed for:
            #
            #   linkbench @32k, 24 seeds, `low`, 40 vs 128      18/24 both
            #   linkbench @32k, 48 seeds, `mid`, 40 vs 128      21/48 both
            #   prose synthesis @13.4k,          40 vs 128      unchanged
            #   digit-table @32k, 24 seeds, `mid`, 40 vs 128    14/24 both
            #
            # The first three were all PROSE, and prose is not what residuals are
            # for -- a low-rank approximation of flowing text is a good one, so
            # the dial was only ever measured where it could not matter.
            # colab/tablebench_cuda.py tests the opposite (a ledger of unrelated
            # 4-digit codes, near full-rank blocks, exact-match scored) and 40
            # and 128 score identically there too.
            #
            # The pool allocates these slots UNIFORMLY on every block, so this is
            # a saving on every block in the session (13.4k A/B on record: pool
            # 2.8 GB -> 1.5 GB) for no measured quality cost.
            #
            # Diverges from MLX's flat 128 deliberately; `high` keeps 128.
            #
            # RAISED 40 -> 128 (2026-08-31), completing the ladder f5f96e13's
            # message announced (low 64 / mid 128 / high 256 / ultra 256) but
            # never actually applied on this side: that commit's config.py hunk
            # only added the residual_quant getters, so the ladder landed in
            # mlx_dkv_wrapper.py alone and CUDA kept 40/40/128/40 while claiming
            # otherwise.
            #
            # BE CLEAR ABOUT WHAT THIS BUYS: nothing measured. The four A/Bs
            # above still stand -- 40 and 128 score identically on all of them,
            # including the digit-table bench built to make residuals matter.
            # This is a CAPACITY RESTORATION justified on bytes, not a quality
            # change. int4 residuals cost 72 B per (slot, kv-head) at head_dim
            # 128 against fp16's 256 B, so:
            #
            #   40 slots @ fp16  = 10,240 B   <- what this preset cost pre-int4
            #   128 slots @ int4 =  9,216 B   <- what it costs now
            #   40 slots @ int4  =  2,880 B   <- what it cost between the two
            #
            # i.e. 128 int4 slots fit inside the budget 40 fp16 slots already
            # had. Against the 40-at-int4 state this is still 3.2x the residual
            # bytes, so if a VRAM regression shows up at 32k+, this is the first
            # dial to put back -- DKV_MAX_RESIDUAL_TOKENS=40 restores it with no
            # expected accuracy cost.
            self.max_residual_tokens = 128
            # `mid` KEEPS ROTATED KEYS. Decided 2026-08-17 with both halves of
            # the trade finally measured, after trying the other way and
            # reverting it.
            #
            # The unrotated pool is the only change that has ever moved accuracy
            # in this project, and it reaches dense EXACTLY on both metrics with
            # the power to resolve anything, Qwen3.5-2B at 32k:
            #
            #                       rotated   UNROTATED   dense
            #   digit-table  24 sd    14/24     24/24     24/24
            #   linkbench    48 sd    21/48     23/48     23/48
            #
            # The cost USED to be 43% / 137%, and most of it turned out to be a
            # bug rather than a price: _remat_attend declined outright on an
            # unrotated pool, which disabled the remat cache for the whole
            # session. Rotating the dense window there instead (see that site)
            # cut it to:
            #
            #   Qwen3.5-2B   ( 6 of 24 attended)  38.63 -> 43.02 ms/tok   +11.5%
            #                                     CI [+2.643, +6.217]
            #   Qwen2.5-1.5B (28 of 28 attended)  51.28 -> 65.10 ms/tok   +26.5%
            #                                     CI [+9.360, +17.865]
            #
            # What remains IS the rotation, and it still scales with attended-
            # layer count because it is applied at read on every attended layer.
            # 11-27% is a real price but no longer a disqualifying one, so
            # whether `mid` should take it is now a live question rather than a
            # settled no -- revisit with a decode-throughput target in hand.
            #
            # DECIDED 2026-08-17, on the post-fix numbers: `mid` TAKES IT.
            #
            # This was declined once, at 43%/137%, and that decline was right for
            # those numbers. It is not right for 11%/27%. What changed is that
            # most of the old cost was _remat_attend refusing to serve an
            # unrotated pool at all rather than the rotation itself; rotating the
            # dense window there fixed it.
            #
            # The trade `mid` now makes: ~11% of decode on a hybrid model, ~27%
            # on a dense-attention one, for EXACT dense parity on both metrics
            # that can resolve anything -- digit-table 24/24 against dense's
            # 24/24 where rotated scores 14/24, and linkbench 23/48 against
            # dense's 23/48. A 42% loss of exact digit recall is not an exotic
            # failure: it is invoices, logs, IDs, any table.
            #
            # `low` and `high` stay rotated, so a rotated pool is still one flag
            # or one preset away for anyone whose decode budget cannot take it:
            # DKV_ROTATED_POOL=1, or --preset low / high.
            #
            # `ultra` is now `mid` in every respect. It is kept because it is a
            # documented name people have in scripts, not because it differs.
            self.rotated_pool = False

        # 2. Individual options overrides (dict or env variables)
        self.decode_cache_enabled = self._get_bool(
            "decode_cache_enabled", "DKV_DECODE_CACHE_ENABLED", self.decode_cache_enabled, config_dict
        )
        self.decode_cache_max_tokens = self._get_int(
            "decode_cache_max_tokens", "DKV_DECODE_CACHE_MAX_TOKENS", self.decode_cache_max_tokens, config_dict
        )
        self.prefill_chunk_size = self._get_int(
            "prefill_chunk_size", "DKV_PREFILL_CHUNK_SIZE", self.prefill_chunk_size, config_dict
        )
        # Safety guard: prefill_chunk_size must accommodate at least ONE full streaming
        # block (1 anchor + micro_block_size active tokens = 1025 tokens at the
        # default micro_block_size=1024).  If the resolved value is too small,
        # ingest_chunk produces zero full blocks → all blocks stay ACCUMULATING →
        # dense window overflows at decode → model collapse.  We clamp upward on
        # non-macOS (CUDA) only; on macOS MLX compression runs post-forward so chunks
        # can be small without this constraint.
        import sys as _sys
        if _sys.platform != "darwin":
            # Derive the floor from the ACTUAL block size. This was hardcoded to
            # 2 * 257 = 514, which assumed micro_block_size=256; the default is
            # now 1024 (capacity 1025), so the constant had stopped protecting
            # anything -- a 600-token chunk would have passed it while producing
            # zero full blocks, which is the collapse the guard exists to prevent.
            #
            # The floor is ONE block, not two. The wrapper rounds the chunk up to
            # a multiple of block capacity anyway, and one block per chunk is the
            # measured-best setting for synthesis, not a degenerate case -- two
            # blocks per chunk scores 33.3 against 63.3 (see the `ultra` branch).
            # KVRuntimeManager now seeds config_dict["micro_block_size"] from its
            # kwarg, so this reads the REAL block size. It previously could not:
            # the kwarg never reached config_dict and DKVConfig has no
            # micro_block_size attribute, so both terms were None and the floor
            # silently came from the stale literal below.
            _mbs = (config_dict.get("micro_block_size")
                    or getattr(self, "micro_block_size", None) or 1024)
            _min_chunk = _mbs + 1
            if self.prefill_chunk_size < _min_chunk:
                self.prefill_chunk_size = _min_chunk
        self.srl_threshold = self._get_int(
            "srl_threshold", "DKV_SRL_THRESHOLD", self.srl_threshold, config_dict
        )
        self.async_svd = self._get_bool(
            "async_svd", "DKV_ASYNC_SVD", self.async_svd, config_dict
        )
        self.mps_watermark = self._get_float(
            "mps_watermark", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", self.mps_watermark, config_dict
        )
        self.torch_compile = self._get_bool(
            "torch_compile", "DKV_USE_TORCH_COMPILE", self.torch_compile, config_dict
        )
        self.approximate_attn = self._get_bool(
            "approximate_attn", "DKV_MPS_APPROXIMATE_ATTN", self.approximate_attn, config_dict
        )
        # srl_age_penalty: subtracts age*penalty from each block's relevance in
        # two_level_gate, biasing selection toward RECENT blocks.  Default moved
        # from 0.01 to 0.0 to match the MLX router, which ranks blocks purely by
        # q·k relevance with no recency term — a recency bias actively drops
        # early-document content on whole-document synthesis (the likely cause
        # of the routed-decode degradation observed at 13.4K).  Re-enable with
        # DKV_SRL_AGE_PENALTY>0 for multi-turn chat, where damping stale
        # concepts from earlier turns can help.
        self.srl_age_penalty = self._get_float(
            "srl_age_penalty", "DKV_SRL_AGE_PENALTY", self.srl_age_penalty, config_dict
        )
        self.kv_quant = self._get_str(
            "kv_quant", "DKV_KV_QUANT", self.kv_quant, config_dict
        )
        # How many recent tokens stay DENSE (uncompressed and exact).
        #
        # This is also the largest decode knob measured: halving mid/ultra's 2048
        # to 1024 is worth ~5% on Qwen3.5-2B at 32k -- 20.99 vs 19.73 and 18.57
        # vs 17.83 tok/s, the smaller window ahead in both interleaved rounds --
        # because it is rows removed from every layer's softmax on every token.
        # (A first unpaired reading said 14%; that was variance. Interleave.)
        #
        # Every accuracy gate held at 1024: synthesis 63.3, needle sweep 9/9
        # INCLUDING depth 0.9, which is the case that actually probes recent
        # context, and linkbench 20/24. TTFT and VRAM unchanged.
        #
        # NOT changed in any preset regardless. 5% of decode is not worth halving
        # the exact-token window in `ultra`, whose whole purpose is fidelity, and
        # the benchmarks that held are not a proof that nothing depends on those
        # tokens. `low` already runs 1024. Set DKV_MAX_ACTIVE_DENSE_TOKENS=1024
        # if you want the trade on a preset that does not default to it.
        self.max_active_dense_tokens = self._get_int(
            "max_active_dense_tokens", "DKV_MAX_ACTIVE_DENSE_TOKENS", self.max_active_dense_tokens, config_dict
        )
        # Issue 10: max_residual_tokens — configurable upper bound on correction slots per block.
        # NativeBlockPool reads DKV_MAX_RESIDUAL_TOKENS directly for backward-compat;
        # DKVConfig surfaces it here for callers that pass config objects.
        self.max_residual_tokens = self._get_int(
            "max_residual_tokens", "DKV_MAX_RESIDUAL_TOKENS", self.max_residual_tokens, config_dict,
            alias_env="DKV_MAX_RESIDUAL",   # MLX's name for the same knob
        )
        # Residual storage format (Production-Grade INT4 / INT8 Residual Buffers).
        # Resolved HERE and nowhere else: KVRuntimeManager forwards all three to
        # NativeBlockPool, which used to read the environment itself with a
        # "none" default and ignore this object entirely.
        #
        # int4 -> int8 (2026-08-31). THIS DEFAULT IS LOAD-BEARING AND ONLY BECAME
        # SO IN e38f3cd1. Before that commit the pool ignored this object and fell
        # back to DKV_RESIDUAL_QUANT with a "none" default, so a caller that did
        # not go through serving.decode_config.apply_best_decode_defaults() got
        # fp16 residuals no matter what this line said. Now the value is forwarded
        # into the constructor, so this line -- not the env -- is what every
        # direct-construction path allocates: serving/hf_dkv_wrapper.py, and
        # therefore colab/run_nat_eval.py, which declines apply_best_decode_defaults
        # on purpose (:243, for reasons about SPARSE_BIAS and the fused Triton
        # gate, nothing to do with residual format). Leaving "int4" here would have
        # silently moved every one of those paths off fp16 and onto the format
        # that measures WORSE than fp16 on both backends (see below), without a
        # line of the diff mentioning them.
        #
        # WHY int8. MEASURED HERE, on CUDA, not inherited from the MLX result.
        # colab/residual_format_niah_cuda.py, Qwen2.5-1.5B-Instruct, 4 verbatim
        # codes at randomised codes AND depths, 12 trials x 16k and 32k = 48
        # needles per cell, greedy, exact-string scored. Arms share prompts, so
        # the informative statistic is the DISCORDANT needles, not the rates:
        #
        #     arm      16k         32k         discordant vs fp16 (96 needles)
        #     fp16    36/48 75.0%  42/48 87.5%   --
        #     fp16_aa 36/48 75.0%  42/48 87.5%    0   (A/A floor, ties exactly)
        #     int8    36/48 75.0%  42/48 87.5%    0   sign test p = 1.0
        #     int4    32/48 66.7%  40/48 83.3%    6, all 6 losses  p = 0.031
        #
        # int8 did not merely match fp16's RATE -- it returned the identical
        # per-needle result on all 96 paired needles. That is the strongest form
        # this instrument can express, and it is why int8 is the default: it is
        # free on quality and halves the residual buffer against fp16.
        #
        # BE HONEST ABOUT WHAT CUDA DID *NOT* REPRODUCE. On MLX the same shape of
        # test at ctx=20000 read int4 0/48, int8 41/48, fp16 42/48 -- int4 lost
        # ALL long-context recall. CUDA does not show that. Here int4 is
        # consistently but mildly worse: 6 lost needles out of 96 and not one
        # gained, which is real (a 6-0 split is p=0.031) and small. So the CUDA
        # case for int8 rests on "identical to fp16 at half the bytes", NOT on
        # rescuing a catastrophe.
        #
        # WHY THE BACKENDS DIVERGE: STILL OPEN, BUT FOUR HYPOTHESES ARE DEAD.
        # Measured 2026-08-31 with colab/residual_format_niah_cuda.py against the
        # MLX session running the mirror experiments. Every arm below is 96
        # needles (12 trials x 16k/32k) unless marked, prompts byte-identical
        # across runs (sha1-checked), A/A tied exactly in every run, and each
        # arm's live allocator fingerprint recorded so no arm can be inert.
        #
        # 1. ROTATION ORIENTATION -- REAL BUT FAR TOO SMALL. MLX stores residual
        #    keys already rotated; mid/high/ultra store pre-RoPE and rotate at
        #    read (`low` already ships MLX's orientation). Forcing rot=1 here:
        #
        #                        fp16    int8    int4   int4's paired deficit
        #        rot=0          78/96   78/96   72/96     6/96, all losses
        #        rot=1          85/96   86/96   73/96    12/96, all losses
        #
        #    Rotation is worth +7 needles to fp16 (p=0.065) and +8 to int8
        #    (p=0.022) and +1 to int4 (p=1.0) -- int4 captures none of the gain.
        #    Its deficit doubles in the predicted direction, but Fisher on 6/96
        #    vs 12/96 is p=0.21. AND IT SETTLES NOTHING: in MLX's own
        #    orientation CUDA int4 still scores 73/96 = 76% where MLX scores 0/48.
        #
        # 2. DILUTION BY THE PROJECT-THEN-ATTEND FRAME SPLIT -- DEAD. The idea
        #    was that a base rotated at the anchor and a residual rotated at its
        #    true position leave the residual's contribution attenuated, so
        #    coarsening it costs little. It requires the split; DKV_REMAT_WHY=1
        #    prints "REMAT ACTIVE" for this configuration, i.e. materialise-then-
        #    SDPA keeps both in ONE frame. The residual was pulling full weight
        #    and int4 still cost only 6/96.
        #
        # 3. "CUDA'S LOW-RANK BASE CARRIES RECALL THAT MLX'S DOES NOT" -- DEAD.
        #    With residuals OFF entirely (DKV_MAX_RESIDUAL_TOKENS=0, verified as a
        #    genuine zero-size buffer) the base retrieves 2/96 exact and loses the
        #    needle's NAME in 93/96, identically in both orientations. The bases
        #    floor together. MLX re-checked its own starved arm and found the same
        #    failure mode (names back, digits wrong), so both sides agree here.
        #
        # 4. MODEL WEIGHT PRECISION -- DEAD ON CUDA. Every MLX number uses
        #    Qwen2.5-1.5B-*4bit* and every CUDA number fp16 weights, so it was
        #    confounded with backend throughout. NF4 here, 16k, 48 needles/cell:
        #        fp16 weights   fp16 36/48   int8 36/48   int4 32/48 (lost 4/won 0)
        #        NF4  weights   fp16 27/48   int8 27/48   int4 27/48 (lost 2/won 2)
        #    NF4 costs ~19 points of baseline and EVERY format pays it equally;
        #    int4's deficit vanishes into noise rather than growing. (Lower
        #    baseline = less sensitive, so this does not prove int4 improves --
        #    but a collapse of MLX's size would have been unmissable.)
        #
        # WHAT IS LEFT. The res=128 cells agreed across backends from the start;
        # the divergence only appears when the residual is STRESSED. The largest
        # unbroken difference is now the NEEDLE SET, not the runtime: MLX's codes
        # (OMEGA-8993-OMEGA, KAPPA-8766-IOTA) fragment 6 of 6 into partial words
        # on the Qwen tokenizer -- ' O','ME','GA' is the exact string
        # multifact_eval_cuda.py:19 warns makes recall a coin flip at a measured
        # 0.19-logit top-2 margin -- while this harness rejects such names at
        # startup and re-verifies per model. Until MLX reruns int4 with
        # whole-word needles, nobody knows how much of its 0/48 is residual
        # format and how much is needle construction.
        #
        # DO NOT attribute the divergence to a backend in either direction until
        # that run exists. Quote each side's own numbers.
        #
        # The mechanism under test: residuals are the EXACT-COPY tokens --
        # DKV_RESIDUAL_EXACT_KEYS / DKV_RESIDUAL_EXCLUDE_SVD drops their lossy SVD
        # twin precisely because the residual is meant to be faithful -- so
        # quantizing them coarsely leaves those tokens with no accurate
        # representation anywhere in the store. On MLX int4 was not recoverable on
        # this tensor: pre-RoPE storage and KIVI per-channel keys each lift it only
        # to ~56-60%, still disjoint from fp16.
        #
        # This now agrees with serving/decode_config.py's BEST_DECODE_DEFAULTS
        # ["DKV_RESIDUAL_QUANT"], which is the same value reached the other way
        # (env setdefault) for cli.py and the gateway. Two live defaults for one
        # dial is how this drifted the first time; they are meant to be equal.
        #
        # DO NOT re-validate a change here with the single-needle sweep. That
        # metric is saturated and ranked int4 (8/9) ABOVE fp16 (7/9) on 3 ctx x 3
        # depths. Use multi-needle verbatim codes with randomised codes and depths.
        self.residual_quant = self._get_str(
            "residual_quant", "DKV_RESIDUAL_QUANT", "int8", config_dict
        ).strip().lower()
        self.residual_quant_group_size = self._get_int(
            "residual_quant_group_size", "DKV_RESIDUAL_QUANT_GROUP_SIZE", 64, config_dict
        )
        # Bit width follows the FORMAT NAME. It used to default to a flat 4 no
        # matter what the format was called, so "int8" packed at 4 bits and was a
        # silent alias for int4 — identical shapes, identical bytes, identical
        # error. DKV_RESIDUAL_QUANT_BITS still overrides.
        self.residual_quant_bits = self._get_int(
            "residual_quant_bits", "DKV_RESIDUAL_QUANT_BITS",
            8 if "8" in self.residual_quant else 4, config_dict
        )
        self.rarity_capture = self._get_bool(
            "rarity_capture", "DKV_RARITY_CAPTURE", True, config_dict
        )
        self.rarity_weight = self._get_float(
            "rarity_weight", "DKV_RARITY_WEIGHT", 1.5, config_dict
        )
        self.rarity_min_idf = self._get_float(
            "rarity_min_idf", "DKV_RARITY_MIN_IDF", 2.0, config_dict
        )
        self.boost_digits = self._get_float(
            "boost_digits", "DKV_BOOST_DIGITS", 20.0, config_dict
        )
        self.boost_owner = self._get_float(
            "boost_owner", "DKV_BOOST_OWNER", 14.6, config_dict
        )
        self.boost_rare = self._get_float(
            "boost_rare", "DKV_BOOST_RARE", 7.3, config_dict
        )

        # ── Shared low-rank bases ────────────────────────────────────────────
        # Blocks whose delta subspaces agree read ONE basis row instead of each
        # storing its own V.  V is 39% of a pool slot and is the item adjacent
        # blocks of a document most nearly agree on, so at frac 0.50 this is
        # 91.4 -> 69.8 MB of pool (-23.6%) with retained delta energy 0.969 and
        # ZERO forced joins -- every merge voluntary and above threshold.
        # `_bytes_per_block` amortises V by the same fraction, so the budget
        # holds proportionally MORE blocks rather than the same context in less
        # memory.
        #
        # OFF IN EVERY PRESET, and the obvious guess is the one that is wrong.
        #
        # `low` is the memory-priority preset, so it looks like the natural home
        # -- it is not, and measurably so. `low` sets kv_quant="q4_0", and 4-bit
        # KV quantisation destroys the subspace agreement this depends on. Same
        # document, same frac 0.50, same 756 written blocks:
        #
        #     preset  kv_quant   groups     joined  forced  mean_kept
        #     mid     f16        293/462       463       0      0.969
        #     low     q4_0       462/462 FULL    0     294      0.685
        #
        # On `low` NO TWO BLOCKS ever clear the join threshold. The store fills
        # with founders, everything after is FORCE-joined, and the feature stops
        # being opportunistic dedup and becomes lossy V-compression at 31.5%
        # delta-energy loss. The VRAM number is identical either way (69.8 MB)
        # because the saving comes from allocating fewer basis ROWS, not from
        # successful grouping -- which is exactly why this failure is invisible
        # if you only watch pool MB.
        #
        # `high` is the max-fidelity rung; spending reconstruction accuracy
        # there contradicts what the preset means.
        #
        # `mid` is where the trade is genuinely good -- 2.58x sharing, 463
        # voluntary joins, ZERO forced, kept 0.969 -- but it is also the
        # default, so enabling it there IS a default change and the accuracy
        # evidence does not carry one. First-step logit fidelity vs a dense
        # control (colab/logit_fidelity.py, n=5) shows no measurable harm, but
        # the ordering is not monotone in frac (0.50 -> KL 8.27, 0.25 -> 9.93,
        # 0.125 -> 8.61), which is the signature of noise rather than a fidelity
        # ordering, and that instrument's own baseline already sits far from
        # dense here (KL 10.58, dense's top-1 at rank 1255) so it cannot resolve
        # a small change on top.
        #
        # So: opt-in, for fp16-KV long-context sessions, via this config key or
        # DKV_SHARED_BASIS. Not a preset default anywhere until an accuracy
        # instrument that can actually resolve it says otherwise.
        self.shared_basis = self._get_bool(
            "shared_basis", "DKV_SHARED_BASIS", False, config_dict
        )
        self.shared_basis_frac = self._get_float(
            "shared_basis_frac", "DKV_SHARED_BASIS_FRAC", 0.50, config_dict
        )

        # The residual quantization dials used to be re-resolved a second time
        # here, from the same keys and the same env vars with the already-resolved
        # value as the default — a no-op that only served to carry a comment
        # claiming the default was "none" when it had been "int4" since f5f96e13.
        # Resolved once above; a second copy is how the first one drifted.

        # Content-Aware / Rarity-Aware Residual Selection dials
        self.rarity_capture = self._get_bool(
            "rarity_capture", "DKV_RARITY_CAPTURE",
            self.rarity_capture, config_dict,
            alias_env="DKV_RESIDUAL_RARITY_CAPTURE"
        )
        self.rarity_weight = self._get_float(
            "rarity_weight", "DKV_RARITY_WEIGHT",
            self.rarity_weight, config_dict,
            alias_env="DKV_RESIDUAL_RARITY_WEIGHT"
        )
        self.rarity_min_idf = self._get_float(
            "rarity_min_idf", "DKV_RARITY_MIN_IDF",
            self.rarity_min_idf, config_dict,
            alias_env="DKV_RESIDUAL_RARITY_MIN_IDF"
        )
        self.boost_digits = self._get_float(
            "boost_digits", "DKV_BOOST_DIGITS",
            self.boost_digits, config_dict
        )
        self.boost_owner = self._get_float(
            "boost_owner", "DKV_BOOST_OWNER",
            self.boost_owner, config_dict
        )
        self.boost_rare = self._get_float(
            "boost_rare", "DKV_BOOST_RARE",
            self.boost_rare, config_dict
        )


        # ── CUDA-specific performance flags ──────────────────────────────────
        # These have no effect on MPS/CPU; they are documented here so that
        # DKV_TELEMETRY=1 output gives a complete picture of active defaults.

        # factual_store: retain full prefill K/V on CPU and build FactualExactStore.
        # Default OFF to match MLX path and documentation.  Enable with
        # DKV_FACTUAL_STORE=1 when factual-recall accuracy matters more than
        # the additional RAM/D2H cost.
        self.factual_store = self._get_bool(
            "factual_store", "DKV_FACTUAL_STORE", False, config_dict
        )

        # gpu_compress: run randomized SVD on the GPU instead of CPU workers.
        # Default ON for CUDA (GPU-rSVD is ~30× faster than CPU rSVD for typical
        # rank/block sizes).  Force CPU with DKV_GPU_COMPRESS=0.
        _cuda_default_gpu_compress = not is_macos
        self.gpu_compress = self._get_bool(
            "gpu_compress", "DKV_GPU_COMPRESS", _cuda_default_gpu_compress, config_dict
        )

        # cuda_graph: capture a static CUDA decode graph.
        # Default OFF until the graph ABI is redesigned around device-resident
        # routing/session state.  The current implementation captures mutable
        # Python state and produces stale outputs after any routing change.
        # DKV_DISABLE_CUDA_GRAPH=0 is retained as a compatibility request,
        # but the current mutable model does not have the static-state ABI
        # required for a valid full-forward graph.  Keep the effective flag
        # false so config telemetry cannot claim graphs are active merely
        # because an environment variable was set.
        _disable_graph = os.environ.get("DKV_DISABLE_CUDA_GRAPH", "1")
        self.cuda_graph_requested = (not is_macos and _disable_graph != "1")
        self.cuda_graph = False

        # gc_interval: decode steps between torch.cuda.empty_cache() calls.
        # 500 on CUDA amortises allocator overhead without large fragmentation.
        # 100 on MPS matches the original value (MPS memory model differs).
        _default_gc = 100 if is_macos else 500
        self.gc_interval = self._get_int(
            "gc_interval", "DKV_GC_INTERVAL", _default_gc, config_dict
        )

        # srl_route_every: run route_query_fixed_k every N decode tokens; reuse
        # cached slots in between.  Reduces D2H traffic from SRL entropy/.item()
        # and centroid/.tolist() calls during long generations.
        # Default 1 = every token (preserves original behaviour).
        # Set to 2-4 on CUDA for 2-4× less D2H during SRL-routed decode.
        self.srl_route_every = self._get_int(
            "srl_route_every", "DKV_SRL_ROUTE_EVERY", 1, config_dict
        )

        # 3. Per-layer rank options
        # early_layer_rank_boost: when True, layers in the first 15% of the network
        # use up to 2× base_rank to improve syntactic representation quality.
        # Default: False for backward compatibility.
        # Enable via: config_dict={'early_layer_rank_boost': True} or DKV_EARLY_LAYER_RANK_BOOST=1
        self.early_layer_rank_boost = self._get_bool(
            "early_layer_rank_boost", "DKV_EARLY_LAYER_RANK_BOOST", False, config_dict
        )
        # max_rank_early: cap for early-layer rank. 0 = auto (2× base_rank).
        # Only used when early_layer_rank_boost=True.
        # Enable via: config_dict={'max_rank_early': 32} or DKV_MAX_RANK_EARLY=32
        self.max_rank_early = self._get_int(
            "max_rank_early", "DKV_MAX_RANK_EARLY", 0, config_dict
        )
        # layer_adaptive_rank: when True, early/late layers use lower ranks (e.g. 8 or 12)
        # and middle layers use higher ranks (e.g. 24), rather than a uniform rank.
        # Default: True. Disable via DKV_LAYER_ADAPTIVE_RANK=0 or config dict.
        # This is a major win for decode throughput (TPS) and VRAM reduction on both CUDA and MLX.
        self.layer_adaptive_rank = self._get_bool(
            "layer_adaptive_rank", "DKV_LAYER_ADAPTIVE_RANK", True, config_dict
        )
        # ── Streaming Compression Default Tradeoffs ───────────────────────────
        # DKV_STREAMING_COMPRESS defaults:
        # - CUDA: OFF (0). SVD compression is a highly parallelizable operation.
        #   Doing it layer-by-layer during the forward pass forces sequential
        #   GPU dispatches (e.g. 624 dispatches for 13k context), incurring massive
        #   launch overhead and serialized latency. Batched deferred SVD at the end
        #   is 20x faster.
        # - MLX: ON (1). macOS unified memory and low launch overhead make streaming
        #   compression critical for bounding peak VRAM without performance penalty.
        # ──────────────────────────────────────────────────────────────────────

        # ── Dual-scale storage ────────────────────────────────────────────────
        # One block size cannot serve both metrics. Measured at rank 224 on
        # Qwen3.5-2B: block 1024 gives synthesis 63.3 (past the dense 60.0) but
        # distractor retrieval 20/24; block 2048 gives retrieval 23/24 (dense's
        # own score) but synthesis 46.7. Block 1536 is worse than both on
        # synthesis, so the middle is not a compromise -- see MLX_PORT item 10c.
        #
        # The idea: keep BOTH scales -- fine for the breadth synthesis needs, a
        # coarse shadow at 2x for the associations retrieval needs.
        #
        # IMPLEMENTED AND DEFAULT OFF, because all three combination policies
        # were measured and none of them earns its cost. Qwen3.5-2B, ultra:
        #
        #   union (attend both scales together, which is what the original
        #     design note proposed via a log-sum-exp merge)
        #                      synthesis 63.3 -> 33.3
        #     Every token then appears TWICE in one softmax as two different
        #     lossy reconstructions of itself; its mass splits between them and
        #     the exact dense window is diluted in the same proportion. Isolated
        #     with DKV_DUAL_SCALE_ATTEND=0: the coarse ingest and the widened
        #     pool alone leave the score at 63.3, so it is the duplication and
        #     nothing else. A log-sum-exp merge would NOT have avoided this --
        #     merging two softmaxes over disjoint key sets is arithmetically the
        #     same as one softmax over their union.
        #
        #   extend (coarse used only where fine routing covered nothing)
        #                      synthesis 63.3, linkbench 20/24 -- no change
        #     At 16k the fine scale routes every block, so the coarse scale
        #     contributes nothing at all. At 32k it does contribute, and moves
        #     neither metric, because the association that linkbench misses is
        #     inside the region fine routing already covered.
        #
        #   swap (coarse replaces fine where 2+ fine blocks fall in its span,
        #     the signature of a split association)
        #                      linkbench 20/24 -> 19/24
        #     The threshold is degenerate: a coarse block spans exactly 2 fine
        #     blocks, so "2+ fine inside" is nearly always true and it replaced
        #     15 of 16. It is "use the coarse scale for everything" wearing a
        #     heuristic, and that is just block 2048 with worse routing.
        #
        # What this rules out is combining two scales AT ATTENTION. The block
        # size result it was meant to solve (item 10c) is unmoved: the choice of
        # granularity has to happen where the representation is BUILT, not where
        # it is read. Left in, default off, DKV_DUAL_SCALE=1 to re-measure.
        self.dual_scale = self._get_bool(
            "dual_scale", "DKV_DUAL_SCALE", getattr(self, "dual_scale", False),
            config_dict)

        # ── Rotated vs unrotated pool ─────────────────────────────────────────
        # Whether the pool stores POST-RoPE keys (MLX's design) or pre-RoPE keys
        # rotated at read time. Storing rotated bakes in the position a block
        # held at COMPRESSION time, which is what makes near-identical
        # distractors collapse together at long context.
        #
        # UNROTATED IS NOW STRICTLY BETTER ON ACCURACY. Measured on Qwen3.5-2B,
        # linkbench at 32k over 48 seeds -- a metric that averages 48 samples per
        # point, unlike multifact, whose +-15-point seed band cannot resolve
        # anything (see the `ultra` branch):
        #
        #     rotated (default)   40/48
        #     UNROTATED           47/48
        #     dense               47/48   <- exact parity
        #
        # The needle regression that previously blocked this IS GONE. The note in
        # triton_fused_decode.pool_stores_rotated_k recorded 6/9 with unrotated
        # keys, including failures at 2k where nothing is compressed -- the
        # signature of a broken read path rather than a fidelity trade. It now
        # scores 9/9 with 9/9 determinism on BOTH Qwen3.5-2B and
        # Qwen2.5-1.5B-Instruct, so whatever caused it was fixed by later work
        # and the trade it implied no longer exists.
        #
        # It is not free, which is why only `ultra` takes it: rotating at read
        # costs decode and memory. Qwen3.5-2B at 32k, interleaved and reversed:
        #     decode  17.60 -> 13.37 and 15.55 -> 12.70 tok/s  (-18% to -24%)
        #     TTFT     9.82 -> 10.15 s
        #     device VRAM 5.21 -> 6.31 GB
        #
        # IT IS A STANDALONE KNOB, NOT AN `ultra` FEATURE. Measured: `mid` with
        # rotated_pool=False also scores 47/48 over the same 48 seeds -- identical
        # to ultra. So the whole win comes from the unrotated pool and none of it
        # from ultra's rank or energy, and DKV_ROTATED_POOL=0 buys dense-parity
        # distractor retrieval on ANY preset without ultra's other costs.
        #
        # DECIDED: low/mid/high keep rotated_pool=True and only `ultra` takes it.
        # The reasoning, recorded so it can be revisited rather than re-derived:
        #   * the cost is ~24% of decode and +1.1 GB, and `mid` is the default
        #     preset -- a quarter of decode is too much to spend by default on
        #     behalf of users who are not doing distractor-heavy retrieval;
        #   * the gain is specific to CONFUSABLE content. Ordinary needle recall
        #     is 9/9 either way on both models, so nothing is lost by default;
        #     ── UPDATE 2026-08-17: "specific" is NARROWER THAN THE TRUTH. The
        #     gain also covers EXACT DIGIT RECALL, which is not an exotic case --
        #     invoices, logs, ledgers, IDs, any table. colab/tablebench_cuda.py,
        #     Qwen3.5-2B at 32k, 24 seeds, asking for one 4-digit amount by its
        #     code out of 60 scattered ledger rows:
        #
        #         dense                     24/24
        #         ultra   (unrotated)       24/24   <- exact parity again
        #         mid     (rotated)         14/24
        #
        #     So rotation costs 42% of exact digit recall, and the unrotated pool
        #     recovers ALL of it. That is now the SECOND metric on which
        #     unrotated == dense exactly and rotated sits well below.
        #
        #     The DECISION below is unchanged, because the ~24% decode and
        #     +1.1 GB are unchanged and this project's stated priorities are
        #     throughput and VRAM. What changes is the ADVICE: recommend `ultra`
        #     for any digit-, code- or table-heavy workload, not just for
        #     distractor-heavy retrieval. Also worth knowing that on this
        #     benchmark NOTHING ELSE helped -- residual budget 40 vs 128 and
        #     attend-every-block are both 14/24, so the rotation is not one
        #     contributor among several, it is the whole gap;
        #   * `ultra` exists precisely to trade speed and memory for quality and
        #     already takes it.
        # Revisit if decode stops being host-bound (~39% GPU-idle today): the
        # rotate-at-read cost may partly hide under dispatch once graphs land.
        #
        # Exported to the environment because pool_stores_rotated_k() reads
        # DKV_ROTATED_POOL at call time; setdefault so an explicit override wins.
        self.rotated_pool = self._get_bool(
            "rotated_pool", "DKV_ROTATED_POOL", getattr(self, "rotated_pool", True),
            config_dict)
        os.environ.setdefault("DKV_ROTATED_POOL", "1" if self.rotated_pool else "0")

        # Publish the fidelity target where the compressor reads it. lowrank's
        # _svd_energy_target() consults DKV_SVD_ENERGY at call time; setdefault so
        # an explicit environment override still wins over the preset.
        os.environ.setdefault("DKV_SVD_ENERGY", str(getattr(self, "svd_energy", 0.999)))

        verbose = os.environ.get("DKV_TELEMETRY", "0") == "1"
        if verbose:
            print(f"[DKV Config] Loaded preset: {self.preset.upper()}")
            print(f"  svd_energy / rank         = {getattr(self, 'svd_energy', None)} / {self.rank}")
            print(f"  decode_cache_enabled      = {self.decode_cache_enabled}")
            print(f"  decode_cache_max_tokens   = {self.decode_cache_max_tokens}")
            print(f"  prefill_chunk_size        = {self.prefill_chunk_size}")
            print(f"  srl_threshold             = {self.srl_threshold}")
            print(f"  async_svd                 = {self.async_svd}")
            print(f"  mps_watermark             = {self.mps_watermark}")
            print(f"  torch_compile             = {self.torch_compile}")
            print(f"  approximate_attn          = {self.approximate_attn}")
            print(f"  srl_age_penalty           = {self.srl_age_penalty}")
            print(f"  early_layer_rank_boost    = {self.early_layer_rank_boost}")
            print(f"  layer_adaptive_rank       = {self.layer_adaptive_rank}")
            print(f"  kv_quant                  = {self.kv_quant}")
            print(f"  max_active_dense_tokens   = {self.max_active_dense_tokens}")
            if self.early_layer_rank_boost:
                print(f"  max_rank_early            = {self.max_rank_early} (0=auto 2×base)")
            if not is_macos:
                print(f"  --- CUDA-specific ---")
                print(f"  factual_store             = {self.factual_store}")
                print(f"  gpu_compress              = {self.gpu_compress}")
                _graph_note = "static ABI unavailable"
                if self.cuda_graph_requested:
                    _graph_note += "; request ignored"
                print(f"  cuda_graph                = {self.cuda_graph} ({_graph_note})")
                print(f"  gc_interval               = {self.gc_interval}")
                print(f"  srl_route_every           = {self.srl_route_every}")

    def _get_bool(self, key: str, env_name: str, default: bool, config_dict: dict,
                  alias_env: str = None) -> bool:
        if key in config_dict:
            val = config_dict[key]
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes", "on")
        for name in (env_name, alias_env):
            if not name:
                continue
            env_val = os.environ.get(name)
            if env_val is not None:
                return env_val.lower() in ("true", "1", "yes", "on")
        return default

    def _get_int(self, key: str, env_name: str, default: int, config_dict: dict,
                 alias_env: str = None) -> int:
        """`alias_env` is a second accepted name for the SAME knob.

        The two runtimes grew different names for identical settings (MLX calls
        the residual budget DKV_MAX_RESIDUAL, this side DKV_MAX_RESIDUAL_TOKENS),
        so a config written against MLX silently configured nothing here. The
        primary name still wins when both are set.
        """
        if key in config_dict:
            try:
                return int(config_dict[key])
            except (ValueError, TypeError):
                pass
        for name in (env_name, alias_env):
            if not name:
                continue
            env_val = os.environ.get(name)
            if env_val is not None:
                try:
                    return int(env_val)
                except (ValueError, TypeError):
                    pass
        return default

    def _get_float(self, key: str, env_name: str, default: float, config_dict: dict,
                   alias_env: str = None) -> float:
        if key in config_dict:
            try:
                return float(config_dict[key])
            except (ValueError, TypeError):
                pass
        for name in (env_name, alias_env):
            if not name:
                continue
            env_val = os.environ.get(name)
            if env_val is not None:
                try:
                    return float(env_val)
                except (ValueError, TypeError):
                    pass
        return default


    def _get_str(self, key: str, env_name: str, default: str, config_dict: dict) -> str:
        if key in config_dict:
            return str(config_dict[key])
        env_val = os.environ.get(env_name)
        if env_val is not None:
            return env_val
        return default
