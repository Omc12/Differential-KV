#include <metal_stdlib>
using namespace metal;

// Helper struct to merge two online softmax states
struct SoftmaxState {
    float m; // max score
    float d; // sum of exp(score - max)
};

inline SoftmaxState merge_softmax_states(SoftmaxState a, SoftmaxState b) {
    if (a.m > b.m) {
        return { a.m, a.d + b.d * exp(b.m - a.m) };
    } else {
        return { b.m, b.d + a.d * exp(a.m - b.m) };
    }
}

// Rotate a single dimension `d` of a key/basis vector stored at buf[base..base+D)
// with Rotary Position Embedding, honoring partial rotary (Qwen3.5/GLM-style
// partial_rotary_factor<1.0, where only the first `rotary_dim` of D dims are
// ever rotated). cos_tab/sin_tab are [num_rows, D] tables (same D stride as
// the key vectors); `row` selects which row (anchor/dense-position index).
//
// For d >= rotary_dim (or !has_rope): returns the raw value unrotated.
// For d < rotary_dim: pairs d with its partner WITHIN [0, rotary_dim) using
// half_r = rotary_dim/2 -- not D/2, which would pull from the wrong dimension
// entirely once rotary_dim < D (this was the actual bug: every call site here
// used to hardcode D/2 regardless of how many dims were genuinely rotary).
// When rotary_dim == D (every model before Qwen3.5), half_r == D/2 and this
// reduces to exactly the original full-width rotation.
// `rope_stride` is the element distance between consecutive ROWS of
// cos_tab/sin_tab. It is NOT always D: the per-anchor and per-dense tables are
// host-padded out to head_dim (stride == D), while the full-sequence table used
// for exact-position residual/fact rotation is the model's raw
// [max_pos, rotary_dim] table (stride == rotary_dim). Passing the wrong stride
// silently reads the right buffer at the wrong offsets, so every caller states
// it explicitly rather than letting it default.
inline float dkv_rope_rotate_dim(
    device const half* buf,
    int base,
    int d,
    device const float* cos_tab,
    device const float* sin_tab,
    int row,
    int D,
    int rotary_dim,
    bool has_rope,
    int rope_stride
) {
    float raw = (float)buf[base + d];
    if (!has_rope || d >= rotary_dim) {
        return raw;
    }
    int half_r = rotary_dim / 2;
    int partner = (d < half_r) ? (d + half_r) : (d - half_r);
    float raw_partner = (float)buf[base + partner];
    float partner_contrib = (d < half_r) ? -raw_partner : raw_partner;
    float c = cos_tab[row * rope_stride + d];
    float s = sin_tab[row * rope_stride + d];
    return raw * c + partner_contrib * s;
}

struct AttentionParams {
    int32_t n_q_heads;
    int32_t n_kv_heads;
    int32_t rank;
    int32_t S_max;
    int32_t K;
    int32_t D;
    float scale;
    int32_t has_rope;
    int32_t max_residual;
    int32_t L_dense;
    // Partial RoPE (Qwen3.5/GLM-style partial_rotary_factor<1.0): only the
    // first rotary_dim of D dims are ever rotated; the rest pass through
    // unrotated. Defaults to D (full rotary) for every model that predates
    // this field, so old callers that never set it keep exact prior behavior.
    int32_t rotary_dim;
    // The REAL last-dim width of VK_pool/VV_pool/U_pool in host memory --
    // i.e. VK_pool.size(1) -- which is NOT always equal to `rank` above.
    // DKV_LAYER_ADAPTIVE_RANK (default on) compresses different layers at
    // different ranks (e.g. 24/48/16 for a base rank of 32), so the pool is
    // allocated at the max across layers (pool_rank) and each layer's blocks
    // only use their own share of it. `rank` is how many of those columns
    // THIS call's layer actually wrote/should read (the loop bound below);
    // pool_rank is the true per-slot stride every VK_pool/VV_pool/U_pool
    // offset must use. Using `rank` for both (the original bug) reads the
    // right data only for slot_id==0 -- every other slot lands on a
    // different, wrong slot's row once rank != pool_rank, because the
    // addressing math silently assumed a narrower stride than the tensor
    // actually has.
    int32_t pool_rank;
    // Independent from has_rope (which reflects cos_anc/sin_anc, the
    // compressed-slot RoPE tables). A decode step can have zero active
    // compressed slots (has_rope false) while the dense window is populated
    // and does have real RoPE angles -- reusing has_rope for both used to
    // silently skip rotating the entire dense window on those steps.
    int32_t has_dense_rope;
    // Whether fact_pos/fact_val_K/fact_val_V are real per-slot buffers vs the
    // host's 1-element dummy substitutes. Residuals are safely gated by
    // max_residual==0 (their load loop just never runs) and dense by
    // L_dense==0, but the fact-override load below is a hardcoded `tid<3`
    // with nothing to gate on otherwise -- without this flag it read
    // fact_pos[slot_id*3+tid] out of bounds on the dummy buffer whenever no
    // real fact data was bound, and a garbage position value that happened to
    // match a real token index would silently corrupt that token's score.
    int32_t has_fact;
    // Exact-position RoPE for the position-specific EXACT overrides (residual
    // K and fact K) instead of rotating them at their block's anchor position.
    //
    // Both residual_K and fact_K store the exact (or exact-minus-recon) key for
    // ONE specific token at within-block offset res_pos/fact_pos -- i.e.
    // absolute position anchor_idx + offset. Rotating them at the ANCHOR's
    // angle (what this kernel did originally) applies the wrong phase to
    // exactly the content these overrides exist to preserve verbatim: digits
    // and alphanumeric codes, whose tokens are highly position-sensitive. For a
    // force-exact block the SVD part carries almost nothing, so the residual IS
    // the content -- mis-rotating it corrupts the very thing being protected.
    //
    // The CUDA/Triton path already fixed this (DKV_RESIDUAL_EXACT_ROPE, default
    // on, A100-validated 75%->88% random-code recall, credited with recovering
    // a dropped-digit code) and MLX likewise appends exact rows as real tokens
    // at their true positions. This ports that same fix to Metal. 0 restores
    // the old anchor-position approximation.
    int32_t has_exact_res_rope;
    // Row count of the full-sequence cos/sin tables (for clamping).
    int32_t rope_full_rows;
    // DKV_RESIDUAL_EXACT_KEYS (MLX parity). When set, res_val_K/res_val_V hold
    // the anchor-relative EXACT value (exact - anchor) rather than a correction
    // to the low-rank reconstruction, so a residual token's true K/V is just
    // anchors_{K,V} + res_val_{K,V}. The kernel then SUBSTITUTES that token's
    // K and V for its lossy low-rank twin instead of nudging them:
    //
    //   score: REPLACE  q . RoPE(anchor_K + res_val_K, anchor_pos)
    //   value: ADD      w_t * (res_val_V - svd_estimate_V)
    //
    // Both halves are required. MLX (the reference, which recalls verbatim
    // codes correctly) masks a residual token's SVD twin to -inf so it
    // contributes to NEITHER the softmax nor the block's value accumulation,
    // then re-attends the exact row as an ordinary token. Replacing only the
    // score leaves the twin's approximate value still summed into the block --
    // the token double-counts, which is why the first rollout regressed.
    //
    // The value-side subtraction above is exactly what the `fact` override path
    // (section 5) has always done; this generalises it from 3 slots/block to
    // the full residual set. Since the correction is expressed relative to the
    // SVD estimate, the anchor_V terms cancel and the accumulation loops need
    // no changes.
    //
    // ROTATION: rotate at the block ANCHOR, NOT anchor+t. Deltas are built from
    // keys already RoPE-rotated by their WITHIN-BLOCK offset at ingest
    // (_preprocess_block_for_compression), and RoPE composes, so the kernel's
    // anchor rotation lands delta token t at absolute anchor_pos + t + 1 -- its
    // true position -- on its own. Rotating again at anchor+t would double-
    // rotate. has_exact_res_rope is therefore ignored while this is set.
    int32_t residual_exact_keys;
    // Number of dense-window rows that are REAL tokens. dense_K/dense_V are a
    // fixed-size workspace padded to max_dense_len, so `L_dense` above is the
    // workspace's ROW STRIDE, not the token count -- the valid tokens occupy
    // only the first L_dense_valid rows.
    //
    // Conflating the two made the kernel (a) attend the padding rows as if they
    // were real tokens, giving garbage real softmax weight, and (b) read
    // cos_dense/sin_dense past their end, since the host sizes those to the
    // VALID count. Loop bounds must use L_dense_valid; buffer addressing must
    // keep using L_dense (the true memory layout). The Triton kernel already
    // carried both values separately; Metal did not.
    int32_t L_dense_valid;
    // DKV_DENSE_VALID_LEN. When 0 (DEFAULT), the dense loop keeps its historical
    // bound of min(L_dense, 768) -- i.e. it may over-attend workspace rows past
    // the valid token count. When 1, it uses L_dense_valid strictly.
    //
    // Strict is the mathematically right bound, but it is NOT safe to default on
    // yet: assemble_dense_window_kv caches per-(layer, anchor) write offsets
    // ACROSS decode steps, so when a block is trimmed the survivors keep their
    // old offsets and the live data can sit at a HIGH offset with a gap at the
    // front. L_dense is then a token COUNT that does not describe the occupied
    // EXTENT, and bounding the loop by it drops live tokens. Measured: enabling
    // strict turned a needle-in-dense-window answer from a wrong-but-coherent
    // 'ABC123' into 'ABC' + repeated token 0 (garbage collapse) -- strictly
    // worse. The historical over-attend accidentally covers the high offsets.
    //
    // Fix the offset-cache layout first (make the workspace densely packed, or
    // return the true occupied extent / a real per-row validity mask), THEN
    // flip this on. The cos/sin row clamp below is applied unconditionally
    // either way, since reading those tables out of bounds is never correct.
    int32_t dense_strict_valid;
    // DKV_DEBUG_GPUREAD -- when set, threadgroup 0's thread 0 writes the RAW
    // BYTES it reads for a few fixed addresses into the debug buffer (index 29),
    // so the GPU's view can be compared against the host's hash of the SAME
    // addresses. Every probe so far is host-side and reads through a
    // synchronising copy, so all of them can only ever report "identical".
    int32_t debug_gpuread;
};

// FNV-1a over raw element bit patterns, for DKV_DEBUG_GPUREAD only. Exact (no
// float rounding to hide a difference) and sequential in a single thread (no
// reduction-order effects), so it reports a change in the bytes the GPU reads
// and nothing else.
static uint dkv_dbg_fnv_half(device const half* p, int n, uint h) {
    for (int i = 0; i < n; ++i) {
        h ^= (uint)as_type<ushort>(p[i]);
        h *= 16777619u;
    }
    return h;
}
static uint dkv_dbg_fnv_i8(device const char* p, int n, uint h) {
    for (int i = 0; i < n; ++i) {
        h ^= (uint)(uchar)p[i];
        h *= 16777619u;
    }
    return h;
}

// Fused Project-Then-Attend Metal decode attention kernel.
// Parallelized as 1 Threadgroup per Query Head.
kernel void decode_attention_metal_kernel(
    device const half* Q [[buffer(0)]],                  // [H_q, D]
    device const int8_t* U_pool [[buffer(1)]],            // [N_pool, S_max, R]
    device const half* U_scale_pool [[buffer(2)]],       // [N_pool]
    device const half* VK_pool [[buffer(3)]],            // [N_pool, R, n_kv, D]
    device const half* VV_pool [[buffer(4)]],            // [N_pool, R, n_kv, D]
    device const half* anchors_K [[buffer(5)]],          // [N_pool, n_kv, D]
    device const half* anchors_V [[buffer(6)]],          // [N_pool, n_kv, D]
    device const int32_t* seq_lens [[buffer(7)]],        // [N_pool]
    device const int32_t* slot_indices [[buffer(8)]],    // [K]
    device half* out_buf [[buffer(9)]],                  // [H_q, D]
    device float* lse_buf [[buffer(10)]],                // [H_q]
    device const AttentionParams& params [[buffer(11)]],
    device const half* scales [[buffer(12)]],            // [N_pool]
    device const float* cos_anc [[buffer(13)]],          // [K, D] float32
    device const float* sin_anc [[buffer(14)]],          // [K, D] float32
    
    // Residual and Fact Anchor Override buffers (Track D)
    device const int16_t* res_pos_K [[buffer(15)]],       // [N_pool, max_residual]
    device const half* res_val_K [[buffer(16)]],          // [N_pool, max_residual, n_kv_heads, D]
    device const int16_t* res_pos_V [[buffer(17)]],       // [N_pool, max_residual]
    device const half* res_val_V [[buffer(18)]],          // [N_pool, max_residual, n_kv_heads, D]
    device const int16_t* fact_pos [[buffer(19)]],        // [N_pool, 3]
    device const half* fact_val_K [[buffer(20)]],         // [N_pool, 3, n_kv_heads, D]
    device const half* fact_val_V [[buffer(21)]],         // [N_pool, 3, n_kv_heads, D]

    // Dense window buffers
    device const half* dense_K [[buffer(22)]],           // [H_kv, L_dense, D]
    device const half* dense_V [[buffer(23)]],           // [H_kv, L_dense, D]
    device const float* cos_dense [[buffer(24)]],        // [L_dense, D]
    device const float* sin_dense [[buffer(25)]],        // [L_dense, D]

    // Full-sequence RoPE tables + per-slot anchor positions, for exact-position
    // rotation of residual/fact overrides (see AttentionParams.has_exact_res_rope).
    // NOTE these are the model's RAW tables: row stride is rotary_dim, NOT D.
    device const float* cos_full [[buffer(26)]],         // [rope_full_rows, rotary_dim]
    device const float* sin_full [[buffer(27)]],         // [rope_full_rows, rotary_dim]
    device const int32_t* anchor_pos [[buffer(28)]],     // [K] absolute anchor position per routed slot
    device float* dbg_buf [[buffer(29)]],                // [32] DKV_DEBUG_GPUREAD scratch

    uint tg_idx [[threadgroup_position_in_grid]],       // Query head index (0..H_q-1)
    uint tid [[thread_position_in_threadgroup]],        // Thread index within threadgroup
    uint t_per_tg [[threads_per_threadgroup]]           // Size of threadgroup
) {
    const int32_t n_q_heads = params.n_q_heads;
    const int32_t n_kv_heads = params.n_kv_heads;
    const int32_t rank = params.rank;
    // Real per-slot stride of VK_pool/VV_pool/U_pool (see AttentionParams.pool_rank).
    const int32_t pool_rank = params.pool_rank;
    const int32_t S_max = params.S_max;
    const int32_t K = params.K;
    const int32_t D = params.D;
    const float scale = params.scale;
    const int32_t has_rope = params.has_rope;
    const int32_t has_dense_rope = params.has_dense_rope;
    const int32_t has_fact = params.has_fact;
    const int32_t has_exact_res_rope = params.has_exact_res_rope;
    const int32_t rope_full_rows = params.rope_full_rows;
    // Independent of has_exact_res_rope: exact keys rotate at the block anchor
    // because the within-block offset is already baked in (see struct comment).
    const int32_t residual_exact_keys = params.residual_exact_keys;
    const int32_t max_residual = params.max_residual;
    const int32_t L_dense = params.L_dense;              // padded row stride
    const int32_t L_dense_valid = params.L_dense_valid;  // real token count
    const int32_t rotary_dim = params.rotary_dim;
    // Return early if threadgroup is out of bounds
    if (tg_idx >= (uint)n_q_heads) return;

    // GQA parameters
    const int g = n_q_heads / n_kv_heads;
    const int kv_head = tg_idx / g;

    // ── Shared memory allocations ─────────────────────────────────────────────
    // D-sized buffers (q_shared, ak_rot_shared) must be >= the largest head_dim
    // of any model DKV targets, NOT the largest rank. They were hardcoded to
    // 128 (the head_dim of every pre-Qwen3.5 model this kernel was written
    // against) -- Qwen3.5-2B's head_dim is 256, so every read of q_shared[d]/
    // ak_rot_shared[d] for d in [128, 256) silently read uninitialized/stale
    // threadgroup memory from whatever the Metal compiler placed adjacent to
    // a too-small allocation. This was the actual root cause of NaN decode
    // output on Qwen3.5 (reproduced on the very first decode call, before any
    // routing/compression logic runs) -- not a Python-side bug, and not
    // fixable by anything upstream of this kernel. Size for 256 going forward;
    // decode_attention_metal() in metal_runtime.mm throws if D > 256 so a
    // future larger-head_dim model fails loudly instead of repeating this.
    threadgroup float q_shared[256];
    // rank-sized buffers: must be >= pool_rank (the widest per-layer active
    // rank DKV_LAYER_ADAPTIVE_RANK ever assigns, currently 48 for a base rank
    // of 32 -- see AttentionParams.pool_rank), not the flat base rank. A
    // middle-schedule layer's rank(48) indexing q_proj_shared[tid] for
    // tid < rank used to silently overflow this when it was sized 32.
    threadgroup float q_proj_shared[48];

    // Shared buffers for reductions
    threadgroup float red_m[128]; // Max threadgroup size 128
    threadgroup float red_d[128];
    threadgroup float red_w_proj[48];
    threadgroup float red_sum_w;
    threadgroup float red_proj_temp[64 * 48]; // [threads_per_tg, rank] temp buffer
    // Anchor-score / query-projection caches, capped at 32 blocks (reduced
    // from an earlier 64 to keep total threadgroup memory under Metal's
    // 32768-byte hard limit once rank-sized buffers above grew to 48) --
    // blocks beyond the cap already fall back to the recompute-in-place path
    // below (k >= cap), so this only costs extra recompute for K > 32, never
    // a wrong answer.
    threadgroup float scores_anc_cached[32];
    threadgroup float q_proj_cached[32 * 48];
    // Shared buffer to hold rotated anchor key for current block [D]
    threadgroup float ak_rot_shared[256];

    // Shared buffers for residual and fact overrides (Track D)
    // Capacity of the residual-position scratch. MUST be >= the pool's
    // max_residual_tokens. MLX's default is 128 (DKV_MAX_RESIDUAL), and the
    // presets here go up to 128 as well.
    //
    // This was 64 while the READ loops below iterated to `max_residual`, so any
    // pool configured above 64 read indices 64..max_residual-1 PAST the array --
    // picking up whatever threadgroup memory sits after it (the V positions, the
    // fact positions, the dense weights) and treating those bytes as residual
    // token POSITIONS. Those bogus positions then matched real tokens and
    // triggered score replacements and value substitutions for them. The writes
    // were guarded at 64, so nothing crashed; the output just quietly became
    // garbage, and non-reproducibly so, since it depended on adjacent memory.
    // That is exactly what happened on raising max_residual 64 -> 128 to match
    // MLX. Reads are now clamped to this capacity as well, belt and braces.
    constexpr int DKV_MAX_RESIDUAL_SHARED = 128;
    threadgroup int16_t res_pos_K_shared[DKV_MAX_RESIDUAL_SHARED];
    threadgroup int16_t res_pos_V_shared[DKV_MAX_RESIDUAL_SHARED];
    threadgroup int16_t fact_pos_shared[3];
    threadgroup float weights_shared[256];

    // Shared buffer for normalized dense weights. Capped at 768 (matches the
    // default DKV_RECENCY_WINDOW=512 + block_size=256) rather than 2048:
    // that pushed total threadgroup memory to 36876 bytes, over Metal's
    // 32768-byte per-threadgroup hard limit (pipeline creation failed
    // outright — not a wrong-answer bug, a can't-run-at-all bug). Loop bounds
    // below clamp to this capacity; the global-buffer stride math still uses
    // the true L_dense since that reflects the host-side tensor's actual layout.
    const int32_t DKV_MAX_DENSE_SHARED = 768;
    threadgroup float dense_w_shared[768];

    // Per-thread value-accumulator capacity (see thread_val in PASS 2). A thread
    // owns dims tid, tid + t_per_tg, ... so it needs ceil(D / t_per_tg) slots: 4
    // at Qwen3.5's D=256 with the 64-thread group the host dispatches. 8 leaves
    // room for D up to 512 without touching the private-memory budget again.
    // The host must keep t_per_tg >= D / DKV_MAX_VAL_PER_THREAD; loops clamp to
    // this capacity so an out-of-contract dispatch drops dims rather than
    // scribbling over the stack.
    constexpr int DKV_MAX_VAL_PER_THREAD = 8;

    // ── Zero ALL threadgroup memory before first use ──────────────────────────
    //
    // Metal does not initialise threadgroup memory; its contents on entry are
    // whatever the previous dispatch left in that block. Any path that READS a
    // shared element it has not written this dispatch therefore consumes stale
    // data from an unrelated invocation -- which reproduces as: identical
    // inputs, identical host state, different output, varying run to run.
    //
    // That is exactly the signature measured in handoff §10d/§10e. Every other
    // mechanism has been eliminated BY MEASUREMENT: the host state going into
    // the call is byte-identical across runs (whole decode_workspace, block
    // list, pool, hidden states), the kernel is bitwise deterministic on fixed
    // inputs in isolation, the branch taken is identical, and forcing
    // COMMIT_AND_WAIT around the dispatch does NOT collapse the distribution
    // (4 runs, 4 distinct outputs), which rules out GPU ordering.
    //
    // Index-set audits of the individual arrays looked correct by eye, but an
    // eyeball argument is not a measurement and the isolated suite (22/22) does
    // not exercise every production path. Zeroing is cheap -- one strided pass
    // over ~40 KB by 64 threads, once per dispatch -- and makes the kernel's
    // output a function of its inputs alone, which it must be.
    for (int i = (int)tid; i < 256; i += (int)t_per_tg) {
        q_shared[i] = 0.0f;
        ak_rot_shared[i] = 0.0f;
        weights_shared[i] = 0.0f;
    }
    for (int i = (int)tid; i < 128; i += (int)t_per_tg) {
        red_m[i] = 0.0f;
        red_d[i] = 0.0f;
        res_pos_K_shared[i] = (int16_t)-1;
        res_pos_V_shared[i] = (int16_t)-1;
    }
    for (int i = (int)tid; i < 48; i += (int)t_per_tg) {
        q_proj_shared[i] = 0.0f;
        red_w_proj[i] = 0.0f;
    }
    for (int i = (int)tid; i < 32; i += (int)t_per_tg) {
        scores_anc_cached[i] = 0.0f;
    }
    for (int i = (int)tid; i < 3; i += (int)t_per_tg) {
        fact_pos_shared[i] = (int16_t)-1;
    }
    for (int i = (int)tid; i < 64 * 48; i += (int)t_per_tg) {
        red_proj_temp[i] = 0.0f;
    }
    for (int i = (int)tid; i < 32 * 48; i += (int)t_per_tg) {
        q_proj_cached[i] = 0.0f;
    }
    for (int i = (int)tid; i < 768; i += (int)t_per_tg) {
        dense_w_shared[i] = 0.0f;
    }
    if (tid == 0) {
        red_sum_w = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. Cache the query vector in shared memory
    for (int d = tid; d < D; d += t_per_tg) {
        q_shared[d] = (float)Q[tg_idx * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── PASS 1: Compute online softmax maximum and denominator ────────────────
    SoftmaxState sm_state = { -1e30f, 0.0f };

    for (int k = 0; k < K; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];

        // Load residual and fact positions for current block into shared memory
        for (int r = (int)tid; r < max_residual; r += (int)t_per_tg) {
            if (r < DKV_MAX_RESIDUAL_SHARED) {
                res_pos_K_shared[r] = res_pos_K[slot_id * max_residual + r];
                res_pos_V_shared[r] = res_pos_V[slot_id * max_residual + r];
            }
        }
        if (tid < 3) {
            // has_fact gate: without it, this read fact_pos[slot_id*3+tid]
            // out of bounds on the host's 1-element dummy buffer whenever no
            // real fact data existed (see AttentionParams.has_fact comment).
            // -1 is the "no override" sentinel every fpos==t check below
            // expects; real token positions are always >= 0.
            fact_pos_shared[tid] = has_fact ? fact_pos[slot_id * 3 + tid] : (int16_t)-1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 1. Compute anchor dot product score (using all threads to collaborate)
        for (int d = tid; d < D; d += t_per_tg) {
            int ak_base = slot_id * n_kv_heads * D + kv_head * D;
            ak_rot_shared[d] = dkv_rope_rotate_dim(
                anchors_K, ak_base, d, cos_anc, sin_anc, k, D, rotary_dim, has_rope, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float thread_anc_sum = 0.0f;
        for (int d = tid; d < D; d += t_per_tg) {
            thread_anc_sum += q_shared[d] * ak_rot_shared[d];
        }
        
        // Reduction over threadgroup to get final anchor dot product
        red_m[tid] = thread_anc_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                red_m[tid] += red_m[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score_anc = red_m[0];
        if (k < 32 && tid == 0) {
            scores_anc_cached[k] = score_anc;
        }
        
        // 2. Compute query projection q_proj[r] = dot(q, VK_pool[slot_id, r])
        if (tid < (uint)rank) {
            float proj_val = 0.0f;
            int base_vk_offset = slot_id * pool_rank * n_kv_heads * D + tid * n_kv_heads * D + kv_head * D;
            for (int d = 0; d < D; ++d) {
                float vk_rot = dkv_rope_rotate_dim(
                    VK_pool, base_vk_offset, d, cos_anc, sin_anc, k, D, rotary_dim, has_rope, D);
                proj_val += q_shared[d] * vk_rot;
            }
            q_proj_shared[tid] = proj_val;

            if (k < 32) {
                q_proj_cached[k * rank + tid] = proj_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Process anchor token
        // Let thread 0 handle the anchor score update
        if (tid == 0) {
            float s_anc_scaled = score_anc * scale;
            sm_state = merge_softmax_states(sm_state, { s_anc_scaled, 1.0f });
        }

        // 4. Process reconstructed tokens
        float scale_u = (float)U_scale_pool[slot_id];
        float block_scale = (float)scales[slot_id];
        for (int t = tid; t < slen; t += t_per_tg) {
            float delta_sum = 0.0f;
            int u_offset = slot_id * S_max * pool_rank + t * pool_rank;
            for (int r = 0; r < rank; ++r) {
                delta_sum += q_proj_shared[r] * (float)U_pool[u_offset + r];
            }
            float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
            
            // Fact anchor override check (K)
            float final_t_score = t_score;
            for (int fi = 0; fi < 3; ++fi) {
                int fpos = (int)fact_pos_shared[fi];
                if (fpos == t) {
                    float exact_k_sum = 0.0f;
                    int base_fact_k = slot_id * 3 * n_kv_heads * D + fi * n_kv_heads * D + kv_head * D;
                    // fact_val_K is the EXACT key of the token at absolute
                    // position anchor+fpos -- rotate it there, not at the
                    // anchor (see AttentionParams.has_exact_res_rope).
                    int f_row = k;
                    int f_stride = D;
                    device const float* f_cos = cos_anc;
                    device const float* f_sin = sin_anc;
                    if (has_exact_res_rope) {
                        f_row = clamp(anchor_pos[k] + fpos, 0, rope_full_rows - 1);
                        f_stride = rotary_dim;
                        f_cos = cos_full;
                        f_sin = sin_full;
                    }
                    for (int d = 0; d < D; ++d) {
                        float fk_rot = dkv_rope_rotate_dim(
                            fact_val_K, base_fact_k, d, f_cos, f_sin, f_row, D, rotary_dim, has_rope, f_stride);
                        exact_k_sum += q_shared[d] * fk_rot;
                    }
                    final_t_score = exact_k_sum * scale;
                    break;
                }
            }

            // Residual correction check (K)
            for (int r = 0; r < min(max_residual, DKV_MAX_RESIDUAL_SHARED); ++r) {
                int rpos = (int)res_pos_K_shared[r];
                if (rpos == t) {
                    float exact_rk_sum = 0.0f;
                    int base_res_k = slot_id * max_residual * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    // Correction form: res_val_K is (exact - recon) for the
                    // token at absolute position anchor+rpos -- rotate it
                    // there, not at the anchor. This is THE fix that recovers
                    // position-sensitive digit/code tokens (see
                    // AttentionParams.has_exact_res_rope).
                    //
                    // Exact-keys form: rotate at the block ANCHOR instead. The
                    // stored delta already carries the token's within-block
                    // rotation, applied at ingest, and RoPE composes -- so the
                    // anchor rotation alone lands it at its true position.
                    // Rotating at anchor+rpos would apply the offset twice.
                    int r_row = k;
                    int r_stride = D;
                    device const float* r_cos = cos_anc;
                    device const float* r_sin = sin_anc;
                    if (has_exact_res_rope && !residual_exact_keys) {
                        r_row = clamp(anchor_pos[k] + rpos, 0, rope_full_rows - 1);
                        r_stride = rotary_dim;
                        r_cos = cos_full;
                        r_sin = sin_full;
                    }
                    int ak_base_res = slot_id * n_kv_heads * D + kv_head * D;
                    for (int d = 0; d < D; ++d) {
                        float rk_rot = dkv_rope_rotate_dim(
                            res_val_K, base_res_k, d, r_cos, r_sin, r_row, D, rotary_dim, has_rope, r_stride);
                        if (residual_exact_keys) {
                            // exact_K = anchor_K + res_val_K, and RoPE is
                            // linear, so rotate each term and add.
                            rk_rot += dkv_rope_rotate_dim(
                                anchors_K, ak_base_res, d, r_cos, r_sin, r_row, D, rotary_dim, has_rope, r_stride);
                        }
                        exact_rk_sum += q_shared[d] * rk_rot;
                    }
                    if (residual_exact_keys) {
                        // SUBSTITUTE: the exact key supersedes the low-rank
                        // estimate entirely (MLX deletes the twin instead).
                        final_t_score = exact_rk_sum * scale;
                    } else {
                        final_t_score += exact_rk_sum * scale;
                    }
                    break;
                }
            }
            
            sm_state = merge_softmax_states(sm_state, { final_t_score, 1.0f });
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Process dense window tokens in PASS 1 ──
    // Loop bound = REAL tokens, never the padded stride. Addressing uses L_dense.
    //
    // NOT capped by dense_w_shared's width. It used to be
    // (`min(L_dense_bound, DKV_MAX_DENSE_SHARED)`), which meant a dense window
    // longer than 768 had every token past row 767 SILENTLY DROPPED FROM
    // ATTENTION -- not down-weighted, not approximated: never scored at all.
    // max_dense_len here is recency_window + block_size (1419 at the default
    // mid preset), so real prompts routinely exceed 768 and simply lose the
    // NEWEST part of their own recency window. A needle in the last few hundred
    // tokens was invisible to the model even though the workspace held it
    // exactly, at its correct RoPE position. The host-side warning for this
    // (metal_runtime.mm) only fired when dense_strict_valid was set, and that
    // defaults OFF -- so it dropped in silence.
    //
    // MLX has no such limit: it runs scaled_dot_product_attention over the whole
    // dense buffer with an `arange < dense_len` mask. Scores are now recomputed
    // per tile in PASS 2 instead of being cached for the whole window, which
    // costs one extra q.k dot per dense token per step and removes the cap.
    const int32_t L_dense_bound = params.dense_strict_valid ? L_dense_valid : L_dense;
    // DKV_DEBUG_GPUREAD: snapshot the compressed-block-only max before the dense
    // window contributes, so a runaway global_m can be attributed to one side or
    // the other.
    const float dbg_m_blocks = sm_state.m;
    for (int t = tid; t < L_dense_bound; t += t_per_tg) {
        float score = 0.0f;
        int base_k = kv_head * L_dense * D + t * D;
        for (int d = 0; d < D; ++d) {
            // Clamp the rope row: cos_dense/sin_dense are sized to the VALID
            // token count, so a loop that runs past it (see dense_strict_valid)
            // would otherwise read them out of bounds. Pure memory safety --
            // in-range rows are unaffected.
            const int t_rope = (t < L_dense_valid) ? t : (L_dense_valid > 0 ? L_dense_valid - 1 : 0);
            float k_rot = dkv_rope_rotate_dim(
                dense_K, base_k, d, cos_dense, sin_dense, t_rope, D, rotary_dim, has_dense_rope, D);
            score += q_shared[d] * k_rot;
        }
        sm_state = merge_softmax_states(sm_state, { score * scale, 1.0f });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce local softmax states of all threads in the threadgroup
    red_m[tid] = sm_state.m;
    red_d[tid] = sm_state.d;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SoftmaxState s1 = { red_m[tid], red_d[tid] };
            SoftmaxState s2 = { red_m[tid + stride], red_d[tid + stride] };
            SoftmaxState merged = merge_softmax_states(s1, s2);
            red_m[tid] = merged.m;
            red_d[tid] = merged.d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_m = red_m[0];
    float global_d = red_d[0];

    // DKV_DEBUG_GPUREAD: separate max-reduce of the blocks-only snapshot, and of
    // the dense window alone, so the runaway score can be attributed. The branch
    // is on a kernel param, so it is uniform across the threadgroup and the
    // barriers inside stay legal.
    float dbg_blocks_max = 0.0f;
    float dbg_dense_max = 0.0f;
    if (params.debug_gpuread) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red_m[tid] = dbg_m_blocks;
        red_d[tid] = sm_state.m;   // this thread's max INCLUDING dense
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                red_m[tid] = max(red_m[tid], red_m[tid + stride]);
                red_d[tid] = max(red_d[tid], red_d[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        dbg_blocks_max = red_m[0];
        dbg_dense_max = red_d[0];
        // Restore the state the rest of the kernel reads from these arrays.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { red_m[0] = global_m; red_d[0] = global_d; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write Log-Sum-Exp to global output buffer
    if (tid == 0) {
        lse_buf[tg_idx] = global_m + log(max(global_d, 1e-9f));
    }

    // Dense weights are no longer precomputed for the whole window here -- they
    // are rebuilt one tile at a time in PASS 2 (see the dense accumulation
    // below), so a window longer than dense_w_shared's width stays correct.

    // ── PASS 2: Accumulate values ─────────────────────────────────────────────
    // Thread-PRIVATE value accumulator, indexed by the thread's own slot, NOT by
    // the head dim.
    //
    // Every accumulation loop below strides `for (d = tid; d < D; d += t_per_tg)`,
    // so a thread only ever touches D/t_per_tg dims -- 4 of them at Qwen3.5's
    // D=256 with a 64-thread group. Sizing this array [D] instead therefore asked
    // for 256 floats in EVERY thread of the group: 64 threads x 256 x 4 B = 64 KB
    // of thread-private memory per threadgroup, 64x more than is used and far past
    // what the hardware backs with registers. Only DKV_MAX_VAL_PER_THREAD slots
    // are needed; keeping it that size costs 1 KB.
    float thread_val[DKV_MAX_VAL_PER_THREAD] = { 0.0f };

    for (int k = 0; k < K; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];

        // Load residual and fact positions for current block into shared memory
        for (int r = (int)tid; r < max_residual; r += (int)t_per_tg) {
            if (r < DKV_MAX_RESIDUAL_SHARED) {
                res_pos_K_shared[r] = res_pos_K[slot_id * max_residual + r];
                res_pos_V_shared[r] = res_pos_V[slot_id * max_residual + r];
            }
        }
        if (tid < 3) {
            // has_fact gate: without it, this read fact_pos[slot_id*3+tid]
            // out of bounds on the host's 1-element dummy buffer whenever no
            // real fact data existed (see AttentionParams.has_fact comment).
            // -1 is the "no override" sentinel every fpos==t check below
            // expects; real token positions are always >= 0.
            fact_pos_shared[tid] = has_fact ? fact_pos[slot_id * 3 + tid] : (int16_t)-1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 1. Recompute anchor score & query projection (or load from cache)
        float score_anc;
        if (k < 32) {
            score_anc = scores_anc_cached[k];
            if (tid < (uint)rank) {
                q_proj_shared[tid] = q_proj_cached[k * rank + tid];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Recompute using RoPE-rotated anchor keys
            for (int d = tid; d < D; d += t_per_tg) {
                int ak_base = slot_id * n_kv_heads * D + kv_head * D;
                ak_rot_shared[d] = dkv_rope_rotate_dim(
                    anchors_K, ak_base, d, cos_anc, sin_anc, k, D, rotary_dim, has_rope, D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float thread_anc_sum = 0.0f;
            for (int d = tid; d < D; d += t_per_tg) {
                thread_anc_sum += q_shared[d] * ak_rot_shared[d];
            }
            red_m[tid] = thread_anc_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    red_m[tid] += red_m[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            score_anc = red_m[0];

            if (tid < (uint)rank) {
                float proj_val = 0.0f;
                int base_vk_offset = slot_id * pool_rank * n_kv_heads * D + tid * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float vk_rot = dkv_rope_rotate_dim(
                        VK_pool, base_vk_offset, d, cos_anc, sin_anc, k, D, rotary_dim, has_rope, D);
                    proj_val += q_shared[d] * vk_rot;
                }
                q_proj_shared[tid] = proj_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 2. Compute local sum of weights and projected weights for deltas
        float w_anc = exp(score_anc * scale - global_m) / max(global_d, 1e-9f);
        float local_sum_w = 0.0f;
        float local_w_proj[48] = { 0.0f }; // Must be >= pool_rank (see q_proj_shared comment above)
        
        float scale_u = (float)U_scale_pool[slot_id];
        float block_scale = (float)scales[slot_id];
        for (int t = (int)tid; t < slen; t += (int)t_per_tg) {
            float delta_sum = 0.0f;
            int u_offset = slot_id * S_max * pool_rank + t * pool_rank;
            for (int r = 0; r < rank; ++r) {
                delta_sum += q_proj_shared[r] * (float)U_pool[u_offset + r];
            }
            float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
            
            // Fact anchor override check (K)
            float final_t_score = t_score;
            for (int fi = 0; fi < 3; ++fi) {
                int fpos = (int)fact_pos_shared[fi];
                if (fpos == t) {
                    float exact_k_sum = 0.0f;
                    int base_fact_k = slot_id * 3 * n_kv_heads * D + fi * n_kv_heads * D + kv_head * D;
                    // fact_val_K is the EXACT key of the token at absolute
                    // position anchor+fpos -- rotate it there, not at the
                    // anchor (see AttentionParams.has_exact_res_rope).
                    int f_row = k;
                    int f_stride = D;
                    device const float* f_cos = cos_anc;
                    device const float* f_sin = sin_anc;
                    if (has_exact_res_rope) {
                        f_row = clamp(anchor_pos[k] + fpos, 0, rope_full_rows - 1);
                        f_stride = rotary_dim;
                        f_cos = cos_full;
                        f_sin = sin_full;
                    }
                    for (int d = 0; d < D; ++d) {
                        float fk_rot = dkv_rope_rotate_dim(
                            fact_val_K, base_fact_k, d, f_cos, f_sin, f_row, D, rotary_dim, has_rope, f_stride);
                        exact_k_sum += q_shared[d] * fk_rot;
                    }
                    final_t_score = exact_k_sum * scale;
                    break;
                }
            }

            // Residual correction check (K)
            for (int r = 0; r < min(max_residual, DKV_MAX_RESIDUAL_SHARED); ++r) {
                int rpos = (int)res_pos_K_shared[r];
                if (rpos == t) {
                    float exact_rk_sum = 0.0f;
                    int base_res_k = slot_id * max_residual * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    // Correction form: res_val_K is (exact - recon) for the
                    // token at absolute position anchor+rpos -- rotate it
                    // there, not at the anchor. This is THE fix that recovers
                    // position-sensitive digit/code tokens (see
                    // AttentionParams.has_exact_res_rope).
                    //
                    // Exact-keys form: rotate at the block ANCHOR instead. The
                    // stored delta already carries the token's within-block
                    // rotation, applied at ingest, and RoPE composes -- so the
                    // anchor rotation alone lands it at its true position.
                    // Rotating at anchor+rpos would apply the offset twice.
                    int r_row = k;
                    int r_stride = D;
                    device const float* r_cos = cos_anc;
                    device const float* r_sin = sin_anc;
                    if (has_exact_res_rope && !residual_exact_keys) {
                        r_row = clamp(anchor_pos[k] + rpos, 0, rope_full_rows - 1);
                        r_stride = rotary_dim;
                        r_cos = cos_full;
                        r_sin = sin_full;
                    }
                    int ak_base_res = slot_id * n_kv_heads * D + kv_head * D;
                    for (int d = 0; d < D; ++d) {
                        float rk_rot = dkv_rope_rotate_dim(
                            res_val_K, base_res_k, d, r_cos, r_sin, r_row, D, rotary_dim, has_rope, r_stride);
                        if (residual_exact_keys) {
                            // exact_K = anchor_K + res_val_K, and RoPE is
                            // linear, so rotate each term and add.
                            rk_rot += dkv_rope_rotate_dim(
                                anchors_K, ak_base_res, d, r_cos, r_sin, r_row, D, rotary_dim, has_rope, r_stride);
                        }
                        exact_rk_sum += q_shared[d] * rk_rot;
                    }
                    if (residual_exact_keys) {
                        // SUBSTITUTE: the exact key supersedes the low-rank
                        // estimate entirely (MLX deletes the twin instead).
                        final_t_score = exact_rk_sum * scale;
                    } else {
                        final_t_score += exact_rk_sum * scale;
                    }
                    break;
                }
            }
            
            float w_t = exp(final_t_score - global_m) / max(global_d, 1e-9f);
            
            local_sum_w += w_t;
            for (int r = 0; r < rank; ++r) {
                local_w_proj[r] += w_t * (float)U_pool[u_offset + r] * scale_u;
            }
            
            if (t < 256) {
                weights_shared[t] = w_t;
            }
        }

        // Reduce sum of weights and projected weights over threadgroup
        red_m[tid] = local_sum_w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                red_m[tid] += red_m[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            red_sum_w = red_m[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce w_proj[r] for all r in parallel using the shared temp buffer
        for (int r = 0; r < rank; ++r) {
            red_proj_temp[tid * rank + r] = local_w_proj[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)rank) {
            float sum_proj = 0.0f;
            for (int t = 0; t < (int)t_per_tg; ++t) {
                sum_proj += red_proj_temp[t * rank + tid];
            }
            red_w_proj[tid] = sum_proj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Add block value contributions to thread_val
        float w_total_anc = w_anc + red_sum_w;
        for (int d = (int)tid, vi = 0; d < D && vi < DKV_MAX_VAL_PER_THREAD; d += (int)t_per_tg, ++vi) {
            // Anchor component (base for all tokens in block)
            thread_val[vi] += w_total_anc * (float)anchors_V[slot_id * n_kv_heads * D + kv_head * D + d];

            // SVD basis component
            float svd_v_contribution = 0.0f;
            int base_vv_offset = slot_id * pool_rank * n_kv_heads * D + kv_head * D + d;
            for (int r = 0; r < rank; ++r) {
                svd_v_contribution += red_w_proj[r] * (float)VV_pool[base_vv_offset + r * n_kv_heads * D];
            }
            thread_val[vi] += svd_v_contribution * block_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Add residual value corrections
        for (int r = 0; r < min(max_residual, DKV_MAX_RESIDUAL_SHARED); ++r) {
            int rpos = (int)res_pos_V_shared[r];
            if (rpos >= 0 && rpos < slen && rpos < 256) {
                // A position that is ALSO a fact anchor is substituted by
                // section 5 below; doing it here too would subtract the token's
                // low-rank estimate twice.
                bool is_fact = false;
                if (residual_exact_keys) {
                    for (int fi = 0; fi < 3; ++fi) {
                        if ((int)fact_pos_shared[fi] == rpos) { is_fact = true; break; }
                    }
                }
                if (is_fact) { continue; }

                float w_res = weights_shared[rpos];
                int base_res_v = slot_id * max_residual * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                if (residual_exact_keys) {
                    // SUBSTITUTE the token's value, matching the score-side
                    // replacement above -- this is MLX's "-inf the twin, attend
                    // the exact row" expressed as a correction.
                    //
                    //   exact_V = anchors_V + res_val_V
                    //   svd_V   = anchors_V + (U[rpos] @ VV) * scale_u * block_scale
                    //
                    // so the anchor terms cancel and only the low-rank estimate
                    // has to be backed out. Steps 2/3 already added svd_V for
                    // this token; without this subtraction the exact value is
                    // ADDED on top of it and the token double-counts, which is
                    // what regressed the score-only version of this mode.
                    int u_offset = slot_id * S_max * pool_rank + rpos * pool_rank;
                    for (int d = (int)tid, vi = 0; d < D && vi < DKV_MAX_VAL_PER_THREAD; d += (int)t_per_tg, ++vi) {
                        float svd_v = 0.0f;
                        int base_vv_offset = slot_id * pool_rank * n_kv_heads * D + kv_head * D + d;
                        for (int rr = 0; rr < rank; ++rr) {
                            svd_v += (float)U_pool[u_offset + rr] * scale_u
                                   * (float)VV_pool[base_vv_offset + rr * n_kv_heads * D];
                        }
                        svd_v *= block_scale;
                        thread_val[vi] += w_res * ((float)res_val_V[base_res_v + d] - svd_v);
                    }
                } else {
                    for (int d = (int)tid, vi = 0; d < D && vi < DKV_MAX_VAL_PER_THREAD; d += (int)t_per_tg, ++vi) {
                        thread_val[vi] += w_res * (float)res_val_V[base_res_v + d];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Add fact anchor override value corrections
        for (int fi = 0; fi < 3; ++fi) {
            int fpos = (int)fact_pos_shared[fi];
            if (fpos >= 0 && fpos < slen && fpos < 256) {
                float w_fact = weights_shared[fpos];
                int base_fact_v = slot_id * 3 * n_kv_heads * D + fi * n_kv_heads * D + kv_head * D;
                
                int u_offset = slot_id * S_max * pool_rank + fpos * pool_rank;
                
                for (int d = (int)tid, vi = 0; d < D && vi < DKV_MAX_VAL_PER_THREAD; d += (int)t_per_tg, ++vi) {
                    float svd_v = 0.0f;
                    int base_vv_offset = slot_id * pool_rank * n_kv_heads * D + kv_head * D + d;
                    for (int r = 0; r < rank; ++r) {
                        svd_v += (float)U_pool[u_offset + r] * scale_u * (float)VV_pool[base_vv_offset + r * n_kv_heads * D];
                    }
                    svd_v = svd_v * block_scale + (float)anchors_V[slot_id * n_kv_heads * D + kv_head * D + d];

                    float exact_fact_v = (float)fact_val_V[base_fact_v + d];
                    thread_val[vi] += w_fact * (exact_fact_v - svd_v);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Add dense window value contributions in PASS 2 (TILED) ────────────────
    // Walk the dense window in tiles of DKV_MAX_DENSE_SHARED. Each tile rebuilds
    // its own weights into dense_w_shared, then accumulates values, so the shared
    // buffer bounds only the TILE size -- never how much context is attended.
    // The softmax is already globally normalised (global_m / global_d from PASS
    // 1 over the full window), so tiles are independent and simply sum.
    int base_v_head = kv_head * L_dense * D;
    for (int tile = 0; tile < L_dense_bound; tile += DKV_MAX_DENSE_SHARED) {
        const int tile_end = min(tile + DKV_MAX_DENSE_SHARED, (int)L_dense_bound);
        const int tile_len = tile_end - tile;

        // Rebuild this tile's normalised weights.
        for (int i = tid; i < tile_len; i += t_per_tg) {
            const int t = tile + i;
            float score = 0.0f;
            int base_k = kv_head * L_dense * D + t * D;
            for (int d = 0; d < D; ++d) {
                const int t_rope = (t < L_dense_valid) ? t : (L_dense_valid > 0 ? L_dense_valid - 1 : 0);
                float k_rot = dkv_rope_rotate_dim(
                    dense_K, base_k, d, cos_dense, sin_dense, t_rope, D, rotary_dim, has_dense_rope, D);
                score += q_shared[d] * k_rot;
            }
            dense_w_shared[i] = exp(score * scale - global_m) / max(global_d, 1e-9f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int d = (int)tid, vi = 0; d < D && vi < DKV_MAX_VAL_PER_THREAD; d += (int)t_per_tg, ++vi) {
            float dense_accum = 0.0f;
            for (int i = 0; i < tile_len; ++i) {
                dense_accum += dense_w_shared[i] * (float)dense_V[base_v_head + (tile + i) * D + d];
            }
            thread_val[vi] += dense_accum;
        }
        // Barrier before the next tile overwrites dense_w_shared.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── DKV_DEBUG_GPUREAD: report what the GPU actually READ ──────────────────
    // Thread 0 of threadgroup 0 only. Reads are issued fresh from device memory
    // here (not from anything cached in registers earlier), so a mismatch against
    // the host's hash of the same addresses means the memory CHANGED between bind
    // and execute; a match with a diverging accumulator means the arithmetic
    // diverged on identical data.
    if (params.debug_gpuread && tg_idx == 0 && tid == 0) {
        // WHOLE-ARRAY checksums, not element samples.
        //
        // The previous version of this block dumped Q[0..7], dense_K[0..7] and a
        // handful of other scalars and was read as "every kernel input is
        // identical across runs". It was not: 8 of Q's 256 elements and 8 of
        // dense_K's L_dense*D (~363k at this context) is a vanishingly small
        // sample, and dense_V / anchors_V / res_val_V -- which between them
        // supply almost all of the output's magnitude -- were never read at all.
        // FNV-1a over the raw half bit patterns is exact (no float rounding) and
        // sequential in one thread (no reduction-order effects), so any change
        // in the bytes the GPU actually reads shows up here.
        const uint FNV0 = 2166136261u;

        // Q for this head, in full.
        dbg_buf[0] = (float)(dkv_dbg_fnv_half(Q + (int)tg_idx * D, D, FNV0) & 0xFFFFFFu);

        // The dense window for THIS kv_head, in full -- both halves. dense_V is
        // the term that supplies most of the output's magnitude and it had never
        // been read back at all.
        if (L_dense > 0) {
            const int dense_off = (int)kv_head * L_dense * D;
            dbg_buf[1] = (float)(dkv_dbg_fnv_half(dense_K + dense_off, L_dense * D, FNV0) & 0xFFFFFFu);
            dbg_buf[2] = (float)(dkv_dbg_fnv_half(dense_V + dense_off, L_dense * D, FNV0) & 0xFFFFFFu);
        } else {
            dbg_buf[1] = -1.0f;
            dbg_buf[2] = -1.0f;
        }

        // Every routed slot's pool state, restricted to exactly the slices this
        // threadgroup reads. Folded into one running hash per array so K slots
        // cost K indices, not K*6.
        uint h_ak = FNV0, h_av = FNV0, h_u = FNV0;
        uint h_vk = FNV0, h_vv = FNV0, h_rk = FNV0, h_rv = FNV0;
        for (int k = 0; k < K; ++k) {
            const int s = (int)slot_indices[k];
            h_ak = dkv_dbg_fnv_half(anchors_K + s * n_kv_heads * D + kv_head * D, D, h_ak);
            h_av = dkv_dbg_fnv_half(anchors_V + s * n_kv_heads * D + kv_head * D, D, h_av);
            h_u  = dkv_dbg_fnv_i8((device const char*)U_pool + s * S_max * pool_rank,
                                  S_max * pool_rank, h_u);
            for (int r = 0; r < pool_rank; ++r) {
                const int vo = s * pool_rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                h_vk = dkv_dbg_fnv_half(VK_pool + vo, D, h_vk);
                h_vv = dkv_dbg_fnv_half(VV_pool + vo, D, h_vv);
            }
            for (int r = 0; r < max_residual; ++r) {
                const int ro = s * max_residual * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                h_rk = dkv_dbg_fnv_half(res_val_K + ro, D, h_rk);
                h_rv = dkv_dbg_fnv_half(res_val_V + ro, D, h_rv);
            }
        }
        dbg_buf[3] = (float)(h_ak & 0xFFFFFFu);
        dbg_buf[4] = (float)(h_av & 0xFFFFFFu);
        dbg_buf[5] = (float)(h_u  & 0xFFFFFFu);
        dbg_buf[6] = (float)(h_vk & 0xFFFFFFu);
        dbg_buf[7] = (float)(h_vv & 0xFFFFFFu);
        dbg_buf[8] = (float)(h_rk & 0xFFFFFFu);
        dbg_buf[9] = (float)(h_rv & 0xFFFFFFu);

        // Integer/scalar params that select which of the bytes above get used.
        for (int i = 0; i < 4; ++i) {
            dbg_buf[10 + i] = (i < K) ? (float)slot_indices[i] : -1.0f;
        }
        dbg_buf[14] = (float)K;
        dbg_buf[15] = (float)L_dense;
        dbg_buf[16] = (float)L_dense_valid;
        dbg_buf[17] = (float)L_dense_bound;
        dbg_buf[18] = (K > 0) ? (float)seq_lens[(int)slot_indices[0]] : -1.0f;
        dbg_buf[19] = scale;
        dbg_buf[20] = (K > 0) ? (float)U_scale_pool[(int)slot_indices[0]] : -1.0f;
        dbg_buf[21] = (float)rank;
        dbg_buf[22] = (float)max_residual;
        dbg_buf[23] = has_rope ? cos_anc[0] : -1.0f;
        // Softmax state that everything downstream is normalised by.
        dbg_buf[24] = global_m;
        dbg_buf[25] = global_d;
        // Which side of PASS 1 produced global_m.
        dbg_buf[26] = dbg_blocks_max;
        dbg_buf[27] = dbg_dense_max;

        // The small per-slot arrays and RoPE tables -- everything that scales or
        // positions a score without being one of the big value arrays above. A
        // single garbage entry in any of these produces exactly the observed
        // signature (one enormous score, global_d collapsing to 1).
        float sc_min = INFINITY, sc_max = -INFINITY;
        float us_min = INFINITY, us_max = -INFINITY;
        float sl_min = INFINITY, sl_max = -INFINITY;
        float ap_min = INFINITY, ap_max = -INFINITY;
        for (int k = 0; k < K; ++k) {
            const int s = (int)slot_indices[k];
            float v = (float)scales[s];         sc_min = min(sc_min, v); sc_max = max(sc_max, v);
            float u = (float)U_scale_pool[s];   us_min = min(us_min, u); us_max = max(us_max, u);
            float l = (float)seq_lens[s];       sl_min = min(sl_min, l); sl_max = max(sl_max, l);
            float a = has_exact_res_rope ? (float)anchor_pos[k] : 0.0f;
            ap_min = min(ap_min, a); ap_max = max(ap_max, a);
        }
        dbg_buf[28] = sc_min; dbg_buf[29] = sc_max;
        dbg_buf[30] = us_min; dbg_buf[31] = us_max;
        dbg_buf[40] = sl_min; dbg_buf[41] = sl_max;
        dbg_buf[42] = ap_min; dbg_buf[43] = ap_max;

        // RoPE tables: a garbage angle rotates a good key into a huge one.
        uint h_cd = FNV0, h_ca = FNV0;
        for (int i = 0; i < L_dense_valid * D; ++i) {
            h_cd ^= as_type<uint>(cos_dense[i]); h_cd *= 16777619u;
            h_cd ^= as_type<uint>(sin_dense[i]); h_cd *= 16777619u;
        }
        if (has_rope) {
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < D; ++i) {
                    h_ca ^= as_type<uint>(cos_anc[k * D + i]); h_ca *= 16777619u;
                    h_ca ^= as_type<uint>(sin_anc[k * D + i]); h_ca *= 16777619u;
                }
            }
        }
        dbg_buf[44] = (float)(h_cd & 0xFFFFFFu);
        dbg_buf[45] = (float)(h_ca & 0xFFFFFFu);
        // Residual/fact POSITIONS -- an out-of-range position picks up an
        // unrelated token's exact key and substitutes it at full weight.
        uint h_rp = FNV0;
        for (int k = 0; k < K; ++k) {
            const int s = (int)slot_indices[k];
            for (int r = 0; r < max_residual; ++r) {
                h_rp ^= (uint)(ushort)res_pos_K[s * max_residual + r]; h_rp *= 16777619u;
                h_rp ^= (uint)(ushort)res_pos_V[s * max_residual + r]; h_rp *= 16777619u;
            }
        }
        dbg_buf[46] = (float)(h_rp & 0xFFFFFFu);
    }

    // Write final output values to out_buf
    for (int d = (int)tid, vi = 0; d < D && vi < DKV_MAX_VAL_PER_THREAD; d += (int)t_per_tg, ++vi) {
        out_buf[tg_idx * D + d] = (half)thread_val[vi];
    }

    // Post-store snapshot: out_buf[0..7] spans EIGHT DIFFERENT LANES (thread d
    // owns dim d for d < t_per_tg), so this is the first place the debug buffer
    // sees more than one thread's arithmetic.
    threadgroup_barrier(mem_flags::mem_device);
    if (params.debug_gpuread && tg_idx == 0 && tid == 0) {
        // 32.. -- indices 0..25 now hold the whole-array input checksums.
        for (int i = 0; i < 8 && i < D; ++i) {
            dbg_buf[32 + i] = (float)out_buf[i];
        }
    }
}
