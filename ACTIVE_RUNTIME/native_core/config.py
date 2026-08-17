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

        # Apply preset defaults
        # NOTE on CUDA prefill_chunk_size: ingest_chunk creates full blocks of exactly
        # (1 + micro_block_size) tokens — 1 anchor + micro_block_size active keys.
        # micro_block_size defaults to 256, so block_capacity = 257.
        # prefill_chunk_size MUST be >= 2 * block_capacity (= 514) so that at least one
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
            self.max_residual_tokens = 40
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
            self.rank = 32
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
            self.max_residual_tokens = 128
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
            self.rank = 128
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
            self.max_residual_tokens = 40
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
            self.rank = 64
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
            self.rank = 64
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
            self.max_residual_tokens = 40
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
        # Safety guard: prefill_chunk_size must accommodate at least 2 full streaming
        # blocks (each block = 1 anchor + micro_block_size active tokens = 257 tokens
        # at the default micro_block_size=256).  If the resolved value is too small,
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
            _mbs = (config_dict.get("micro_block_size")
                    or getattr(self, "micro_block_size", None) or 256)
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

    def _get_bool(self, key: str, env_name: str, default: bool, config_dict: dict) -> bool:
        if key in config_dict:
            val = config_dict[key]
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes", "on")
        env_val = os.environ.get(env_name)
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

    def _get_float(self, key: str, env_name: str, default: float, config_dict: dict) -> float:
        if key in config_dict:
            try:
                return float(config_dict[key])
            except (ValueError, TypeError):
                pass
        env_val = os.environ.get(env_name)
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
