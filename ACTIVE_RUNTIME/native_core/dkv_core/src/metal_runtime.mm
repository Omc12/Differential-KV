#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_runtime.hpp"
#include "dkv_metallib.hpp"
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <tuple>
#include <memory>
#include <vector>
#include <initializer_list>
#include <map>
#include <utility>

namespace dkv {

// Last DKV_DEBUG_GPUREAD scratch buffer, exposed via dkv_core.last_debug_buffer()
// rather than widening decode_attention_metal's return arity (every caller
// unpacks a 2-tuple). Read it from PYTHON, after the stream has settled -- see
// the note at the dispatch site for why reading it in-function crashes.
static torch::Tensor g_last_dbg;

torch::Tensor last_debug_buffer() { return g_last_dbg; }

// Pipeline State Manager for Metal Decode Kernel
class MetalDecodePipeline {
public:
    static MetalDecodePipeline& getInstance() {
        static MetalDecodePipeline instance;
        return instance;
    }

    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    bool initialized = false;

    MetalDecodePipeline() {
        @autoreleasepool {
            // Get PyTorch's active MPS device
            device = at::mps::MPSDevice::getInstance()->device();
            if (!device) {
                std::cerr << "[DKV Metal] Failed to retrieve MPS MTLDevice!" << std::endl;
                return;
            }

            // Create dispatch data from the embedded C++ metallib byte array
            dispatch_data_t data = dispatch_data_create(
                dkv_metallib, 
                dkv_metallib_len, 
                nil, 
                DISPATCH_DATA_DESTRUCTOR_DEFAULT
            );

            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
            if (error || !library) {
                std::cerr << "[DKV Metal] Failed to load embedded Metal library: " 
                          << (error ? [[error localizedDescription] UTF8String] : "Unknown error") 
                          << std::endl;
                return;
            }

            id<MTLFunction> function = [library newFunctionWithName:@"decode_attention_metal_kernel"];
            if (!function) {
                std::cerr << "[DKV Metal] Kernel function 'decode_attention_metal_kernel' not found!" << std::endl;
                return;
            }

            pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
            if (error || !pipelineState) {
                std::cerr << "[DKV Metal] Failed to create compute pipeline state: " 
                          << (error ? [[error localizedDescription] UTF8String] : "Unknown error") 
                          << std::endl;
                return;
            }

            initialized = true;
        }
    }
};

bool is_metal_available() {
    return MetalDecodePipeline::getInstance().initialized;
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
    // Must match dkv_decode.metal's AttentionParams field-for-field (same
    // memory layout is uploaded as raw bytes). See that file's comment for
    // what this does.
    int32_t rotary_dim;
    // Real per-slot stride of VK_pool/VV_pool/U_pool -- see the .metal file's
    // comment; derived from VK_pool.size(1) below, NOT equal to `rank` above
    // whenever DKV_LAYER_ADAPTIVE_RANK (default on) gives this layer a
    // different rank than the pool's max-across-layers allocation width.
    int32_t pool_rank;
    // Independent from `has_rope` (which reflects cos_anc/sin_anc, the
    // COMPRESSED-slot RoPE tables). A decode step can have zero active
    // compressed slots (has_rope false) while the dense window is populated
    // and does have real RoPE angles -- reusing has_rope for both used to
    // silently skip rotating the entire dense window on those steps.
    int32_t has_dense_rope;
    // Whether fact_pos/fact_val_K/fact_val_V carry real per-slot data (vs the
    // 1-element dummy buffers substituted when the host has none). Unlike
    // residuals (naturally gated by max_residual==0, so their load loop never
    // executes) and dense (gated by L_dense==0), the kernel's fact-override
    // load site is a hardcoded `tid<3` with no count to gate on -- without
    // this flag it unconditionally read fact_pos[slot_id*3+tid] for tid up to
    // 2 even when only a 1-element dummy buffer was bound, an out-of-bounds
    // read whose garbage value could spuriously match a real token position
    // and silently corrupt that token's score with more out-of-bounds reads
    // from the dummy fact_val_K/V buffers.
    int32_t has_fact;
    // Rotate residual-K / fact-K at their TRUE absolute token position
    // (anchor + within-block offset) instead of at the block anchor. See the
    // .metal file's comment -- this is the Metal port of the CUDA/Triton
    // DKV_RESIDUAL_EXACT_ROPE fix. Set only when the caller supplies the
    // full-sequence rope tables + anchor positions below.
    int32_t has_exact_res_rope;
    int32_t rope_full_rows;
    // DKV_RESIDUAL_EXACT_KEYS -- see the .metal file's comment. Read from the
    // environment here (same var the compressor reads) so the stored residual
    // form and the kernel's interpretation of it cannot disagree.
    int32_t residual_exact_keys;
    // Real dense-token count (dense_K/dense_V are a padded workspace, so
    // L_dense above is only their row STRIDE). See the .metal comment.
    int32_t L_dense_valid;
    int32_t dense_strict_valid;   // DKV_DENSE_VALID_LEN, default 0 -- see .metal comment
    int32_t debug_gpuread;        // DKV_DEBUG_GPUREAD -- see .metal comment
};

// DKV_DENSE_VALID_LEN -- OPT-IN (default OFF), PENDING RE-MEASUREMENT.
//
// The earlier note here ("measured, do not flip -- the strict bound made the
// needle answer WORSE, '...ZEBRA' -> '...Z'") is NOT reliable evidence: that A/B
// predates DKV_DETERMINISTIC, and two runs of the SAME build on the same prompt
// were later shown to disagree at temperature 0. A single-run token comparison
// could not have measured this either way.
//
// Two things have since changed on the host side, both MLX parity:
//   * assemble_dense_window_kv now verifies its cached offsets still describe a
//     layout packed from row 0 and repacks when they do not, and it returns the
//     ACTUAL written extent rather than the block-list sum -- so no live token
//     can sit past the bound, which is the failure mode the old note was
//     (correctly) afraid of.
//   * it now zeroes the workspace tail, as MLX does at every point its dense
//     window shrinks.
//
// MLX masks unconditionally -- `dense_add = where(arange(max_dense_len) <
// dense_len, 0, -inf)` (mlx_dkv_wrapper.py:4154) -- and has no equivalent flag.
// The strict bound is the mathematically correct one; leaving it off means the
// kernel attends the padded tail as if it were live context. Re-measure under
// DKV_DETERMINISTIC=1 and, if it holds up, make it the default to match MLX.
static int dkv_dense_strict_valid() {
    static int cached = -1;
    if (cached < 0) {
        // DEFAULT ON as of 2026-07-29. Metal was the ONLY backend attending the
        // padded tail of the dense workspace as if it were live context:
        //   MLX    — `where(arange(max_dense_len) < dense_len, 0, -inf)`,
        //            unconditional (mlx_dkv_wrapper.py:4154)
        //   Triton — `valid_t = mask_t & (offs_t < L_dense_valid)`,
        //            unconditional (triton_fused_decode.py:2455)
        //   Metal  — opt-in, and the opt-in defaulted OFF
        // The strict bound is the mathematically correct one; the old note here
        // kept it off on the strength of a single-run token comparison that
        // predates the determinism fix (§10q) and could not have measured
        // anything. Set DKV_DENSE_VALID_LEN=0 to restore the old behaviour.
        const char* e = std::getenv("DKV_DENSE_VALID_LEN");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

// Cached once: residuals stored as (exact_K - anchor_K) rather than as a
// correction to the low-rank reconstruction. Both the compressor
// (compression/lowrank.py) and this kernel read DKV_RESIDUAL_EXACT_KEYS.
// DEFAULT ON, matching MLX (DKV_RESIDUAL_EXCLUDE_SVD defaults "1",
// mlx_dkv_wrapper.py:1763). Must agree with
// compression/lowrank._exact_keys_enabled -- both read this same variable, so
// they cannot disagree, but their DEFAULTS have to match or a build would store
// one form and decode the other. Measured: every full_attention layer's
// attention cos vs dense improves (see the table in _exact_keys_enabled).
// Set DKV_RESIDUAL_EXACT_KEYS=0 to revert both sides to the correction form.
static int dbg_gpuread_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("DKV_DEBUG_GPUREAD");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static int dkv_residual_exact_keys() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("DKV_RESIDUAL_EXACT_KEYS");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

std::tuple<torch::Tensor, torch::Tensor> decode_attention_metal(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& scales,
    const torch::Tensor& cos_anc,
    const torch::Tensor& sin_anc,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank,
    // Residual and Fact Anchor Override buffers (Track D)
    const torch::Tensor& res_pos_K,
    const torch::Tensor& res_val_K,
    const torch::Tensor& res_pos_V,
    const torch::Tensor& res_val_V,
    const torch::Tensor& fact_pos,
    const torch::Tensor& fact_val_K,
    const torch::Tensor& fact_val_V,
    // Dense window buffers
    const torch::Tensor& dense_K,
    const torch::Tensor& dense_V,
    const torch::Tensor& cos_dense,
    const torch::Tensor& sin_dense,
    // Partial RoPE (Qwen3.5/GLM-style partial_rotary_factor<1.0): only the
    // first rotary_dim dims of D are ever rotated. -1 means "full rotary"
    // (rotary_dim == D); default lives in the header declaration, not here.
    int rotary_dim,
    // Full-sequence RoPE tables [max_pos, rotary_dim] (the model's RAW tables,
    // NOT padded to head_dim) + per-routed-slot absolute anchor positions [K].
    // When all three are non-empty, residual-K / fact-K are rotated at their
    // true absolute token position instead of the block anchor's. Undefined /
    // empty tensors simply disable that (falls back to anchor rotation), so
    // callers that don't have these stay bit-identical to before.
    const c10::optional<torch::Tensor>& cos_full,
    const c10::optional<torch::Tensor>& sin_full,
    const c10::optional<torch::Tensor>& anchor_pos
) {
    auto& mps_pipeline = MetalDecodePipeline::getInstance();
    if (!mps_pipeline.initialized) {
        throw std::runtime_error("DKV Metal compute pipeline not initialized!");
    }
    // dkv_decode.metal's q_proj_shared/red_w_proj/local_w_proj/red_proj_temp/
    // q_proj_cached buffers are sized for rank<=48 -- the loop-bound rank
    // (DKV_LAYER_ADAPTIVE_RANK's widest per-layer schedule value for a base
    // rank of 32), not pool_rank (see below).
    if (rank > 48) {
        throw std::runtime_error(
            "decode_attention_metal: rank (" + std::to_string(rank) +
            ") exceeds the Metal kernel's compiled-in max of 48. Bump "
            "q_proj_shared/red_w_proj/local_w_proj/red_proj_temp/q_proj_cached "
            "in dkv_decode.metal (and the threadgroup memory budget check) "
            "before raising the base rank past 32 (whose adaptive schedule "
            "tops out at 48).");
    }

    const auto device = Q.device();
    const int D = Q.size(1);
    // dkv_decode.metal's q_shared/ak_rot_shared threadgroup buffers are sized
    // for D<=256 (Qwen3.5-2B's head_dim, the largest this kernel has ever
    // been built against). A larger head_dim would silently overflow those
    // fixed-size threadgroup arrays and produce NaN output that looks like a
    // completely unrelated bug (this is exactly how the original Qwen3.5 decode
    // corruption was found) -- fail loudly instead. thread_val is no longer in
    // this list: it is sized per THREAD now, and the threadgroup size below
    // scales with D to cover it.
    if (D > 256) {
        throw std::runtime_error(
            "decode_attention_metal: head_dim (" + std::to_string(D) +
            ") exceeds the Metal kernel's compiled-in max of 256. "
            "Bump q_shared/ak_rot_shared in dkv_decode.metal "
            "(and the threadgroup memory budget check) before using a "
            "model with this head_dim on MPS.");
    }
    const int K_slots = slot_indices.size(0);
    const int L_dense = (dense_K.defined() && dense_K.numel() > 0) ? dense_K.size(-2) : 0;

    // If no slots are active and dense window is empty, return zero outputs immediately
    if (K_slots == 0 && L_dense == 0) {
        auto out = torch::zeros({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
        auto lse = torch::full({n_q_heads}, -std::numeric_limits<float>::infinity(), torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return {out, lse};
    }

    // Pre-allocate output tensors.
    //
    // DKV_DEBUG_PERSIST_OUT=1 — DIAGNOSTIC. Reuse ONE out/lse pair per shape
    // instead of allocating fresh each call. `torch::empty` takes a block from
    // the MPS caching allocator, which may be RECYCLED memory whose previous
    // owner still has a GPU write in flight — freed on the host, not yet retired
    // on the device. That stale write would land after ours and corrupt the
    // result. It is not aliasing between the buffers we bind (measured: none,
    // handoff §10g) and barriers around OUR dispatch cannot help, because the
    // racing write belongs to an earlier op on a since-freed buffer.
    //
    // A tensor that is allocated once and never freed cannot be recycled memory,
    // so if the output stabilises under this flag, that is the mechanism. The
    // kernel fully overwrites both tensors every call, so reuse is safe here.
    // Not enabled by default: it is not thread-safe across concurrent sessions.
    static int persist_out = -1;
    if (persist_out < 0) {
        const char* e = std::getenv("DKV_DEBUG_PERSIST_OUT");
        persist_out = (e && e[0] == '1') ? 1 : 0;
    }
    // Allocated HERE, not at the bind site: creating a tensor after the compute
    // encoder exists makes PyTorch open a second encoder on the same command
    // buffer, which trips
    //   "A command encoder is already encoding to this command buffer".
    // Always allocated (1 element when the diagnostic is off) so buffer 29 always
    // has a valid binding.
    auto dbg = torch::zeros({dbg_gpuread_enabled() ? 64 : 1},
                            torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor out, lse;
    if (persist_out) {
        static std::map<std::pair<int64_t, int64_t>, std::pair<torch::Tensor, torch::Tensor>> cache;
        auto key = std::make_pair((int64_t)n_q_heads, (int64_t)D);
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto o = torch::empty({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
            auto l = torch::empty({n_q_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            it = cache.emplace(key, std::make_pair(o, l)).first;
        }
        out = it->second.first;
        lse = it->second.second;
    } else {
        out = torch::empty({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
        lse = torch::empty({n_q_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    @autoreleasepool {
        // Ensure all input tensors are contiguous in memory before passing to Metal
        // We must perform all contiguous checks and PyTorch tensor operations FIRST
        // before synchronizing the stream to ensure PyTorch does not open new command encoders afterwards.
        auto Q_c = Q.is_contiguous() ? Q : Q.contiguous();
        auto U_c = U_pool.is_contiguous() ? U_pool : U_pool.contiguous();
        auto U_scale_c = U_scale_pool.is_contiguous() ? U_scale_pool : U_scale_pool.contiguous();
        auto VK_c = VK_pool.is_contiguous() ? VK_pool : VK_pool.contiguous();
        auto VV_c = VV_pool.is_contiguous() ? VV_pool : VV_pool.contiguous();
        auto AK_c = anchors_K.is_contiguous() ? anchors_K : anchors_K.contiguous();
        auto AV_c = anchors_V.is_contiguous() ? anchors_V : anchors_V.contiguous();
        auto slens_c = seq_lens.is_contiguous() ? seq_lens : seq_lens.contiguous();
        auto scales_c = scales.is_contiguous() ? scales : scales.contiguous();
        auto slots_c = slot_indices.is_contiguous() ? slot_indices : slot_indices.contiguous();
        // Element-width contract with dkv_decode.metal.
        //
        // The shader indexes every buffer through a typed pointer, so a caller
        // that hands over the right values in the WRONG dtype does not get a
        // rounded answer -- it gets a reinterpretation plus an out-of-bounds
        // walk whenever the actual element is narrower than the declared one.
        // That is a silent, run-to-run-varying corruption (see the cos_dense
        // note just below), and nothing in the type system was catching it, so
        // check it here where the binding is established.
        auto expect_dtype = [](const torch::Tensor& t, torch::ScalarType want,
                               const char* name) {
            if (!t.defined() || t.numel() == 0) return;   // dummy/empty is fine
            if (t.scalar_type() != want) {
                throw std::runtime_error(
                    std::string("decode_attention_metal: ") + name + " must be " +
                    c10::toString(want) + " to match dkv_decode.metal, got " +
                    c10::toString(t.scalar_type()) +
                    ". Convert at the call site -- the shader reads this buffer "
                    "through a typed pointer and a narrower element silently "
                    "reads past the end of the allocation.");
            }
        };
        expect_dtype(Q,            torch::kFloat16, "Q");
        expect_dtype(U_pool,       torch::kChar,    "U_pool");
        expect_dtype(U_scale_pool, torch::kFloat16, "U_scale_pool");
        expect_dtype(VK_pool,      torch::kFloat16, "VK_pool");
        expect_dtype(VV_pool,      torch::kFloat16, "VV_pool");
        expect_dtype(anchors_K,    torch::kFloat16, "anchors_K");
        expect_dtype(anchors_V,    torch::kFloat16, "anchors_V");
        expect_dtype(seq_lens,     torch::kInt32,   "seq_lens");
        expect_dtype(scales,       torch::kFloat16, "scales");
        expect_dtype(slot_indices, torch::kInt32,   "slot_indices");
        expect_dtype(res_pos_K,    torch::kInt16,   "res_pos_K");
        expect_dtype(res_pos_V,    torch::kInt16,   "res_pos_V");
        expect_dtype(res_val_K,    torch::kFloat16, "res_val_K");
        expect_dtype(res_val_V,    torch::kFloat16, "res_val_V");
        expect_dtype(fact_pos,     torch::kInt16,   "fact_pos");
        expect_dtype(fact_val_K,   torch::kFloat16, "fact_val_K");
        expect_dtype(fact_val_V,   torch::kFloat16, "fact_val_V");
        expect_dtype(dense_K,      torch::kFloat16, "dense_K");
        expect_dtype(dense_V,      torch::kFloat16, "dense_V");
        if (cos_full.has_value())   expect_dtype(*cos_full,   torch::kFloat32, "cos_full");
        if (sin_full.has_value())   expect_dtype(*sin_full,   torch::kFloat32, "sin_full");
        if (anchor_pos.has_value()) expect_dtype(*anchor_pos, torch::kInt32,   "anchor_pos");

        // Every RoPE table below is declared `device const float*` in
        // dkv_decode.metal. A caller that hands over the model's native fp16
        // cos/sin instead does NOT merely lose precision: the shader reads each
        // pair of halves as one float32 and indexes twice as far as the buffer
        // is long, so it rotates keys by nonsense angles and then runs off the
        // end of the allocation. That is exactly what the dense window did --
        // it produced single scores in the 10^5 range, collapsed the softmax to
        // one junk token (global_d == 1), and, because the overrun read recycled
        // memory, made decode output differ run to run on byte-identical inputs
        // at temperature 0. Convert rather than trust the caller, and say so
        // once so the real fix can be made upstream.
        auto rope_f32 = [](const torch::Tensor& t, const char* name) {
            if (!t.defined() || t.numel() == 0 || t.scalar_type() == torch::kFloat32) {
                return t;
            }
            static std::map<std::string, bool> warned;
            if (!warned[name]) {
                warned[name] = true;
                fprintf(stderr,
                        "[dkv] %s arrived as %s; the Metal kernel reads float32 "
                        "RoPE tables. Converting, but fix the caller.\n",
                        name, c10::toString(t.scalar_type()));
            }
            return t.to(torch::kFloat32);
        };
        const torch::Tensor cos_anc_f   = rope_f32(cos_anc,   "cos_anc");
        const torch::Tensor sin_anc_f   = rope_f32(sin_anc,   "sin_anc");
        const torch::Tensor cos_dense_f = rope_f32(cos_dense, "cos_dense");
        const torch::Tensor sin_dense_f = rope_f32(sin_dense, "sin_dense");

        // cos_anc / sin_anc: [K, D] float32. May be empty if RoPE info unavailable.
        bool has_rope = (cos_anc.defined() && cos_anc.numel() > 0);
        auto cos_c = has_rope ? (cos_anc_f.is_contiguous() ? cos_anc_f : cos_anc_f.contiguous())
                              : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        auto sin_c = has_rope ? (sin_anc_f.is_contiguous() ? sin_anc_f : sin_anc_f.contiguous())
                              : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        int has_rope_flag = has_rope ? 1 : 0;

        // Prepare contiguous tensors for dense window
        bool has_dense = (dense_K.defined() && dense_K.numel() > 0 && dense_V.defined() && dense_V.numel() > 0);
        auto dense_K_c = has_dense ? (dense_K.is_contiguous() ? dense_K : dense_K.contiguous())
                                   : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        auto dense_V_c = has_dense ? (dense_V.is_contiguous() ? dense_V : dense_V.contiguous())
                                   : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        bool has_dense_rope = has_dense && cos_dense.defined() && cos_dense.numel() > 0;
        auto cos_dense_c = has_dense_rope ? (cos_dense_f.is_contiguous() ? cos_dense_f : cos_dense_f.contiguous())
                                          : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        auto sin_dense_c = has_dense_rope ? (sin_dense_f.is_contiguous() ? sin_dense_f : sin_dense_f.contiguous())
                                          : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));

        // Track D: Prepare contiguous tensors for Residual and Fact Overrides
        bool has_res = (res_pos_K.defined() && res_pos_K.numel() > 0 && res_val_K.defined() && res_val_K.numel() > 0);
        auto res_pos_K_c = has_res ? (res_pos_K.is_contiguous() ? res_pos_K : res_pos_K.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt16).device(Q.device()));
        auto res_val_K_c = has_res ? (res_val_K.is_contiguous() ? res_val_K : res_val_K.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        auto res_pos_V_c = has_res ? (res_pos_V.is_contiguous() ? res_pos_V : res_pos_V.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt16).device(Q.device()));
        auto res_val_V_c = has_res ? (res_val_V.is_contiguous() ? res_val_V : res_val_V.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        int max_res_val = has_res ? static_cast<int>(res_pos_K_c.size(1)) : 0;

        // Exact-position residual/fact RoPE requires all three of the full
        // tables + anchor positions; any missing piece falls back to the
        // anchor-position approximation rather than reading a dummy buffer.
        bool has_exact_res_rope = (
            cos_full.has_value()   && cos_full->defined()   && cos_full->numel() > 0 &&
            sin_full.has_value()   && sin_full->defined()   && sin_full->numel() > 0 &&
            anchor_pos.has_value() && anchor_pos->defined() && anchor_pos->numel() > 0);
        auto cos_full_c = has_exact_res_rope ? (cos_full->is_contiguous() ? *cos_full : cos_full->contiguous())
                                             : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        auto sin_full_c = has_exact_res_rope ? (sin_full->is_contiguous() ? *sin_full : sin_full->contiguous())
                                             : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        auto anchor_pos_c = has_exact_res_rope ? (anchor_pos->is_contiguous() ? *anchor_pos : anchor_pos->contiguous())
                                               : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(Q.device()));
        int rope_full_rows = has_exact_res_rope ? static_cast<int>(cos_full_c.size(0)) : 0;

        bool has_fact = (fact_pos.defined() && fact_pos.numel() > 0 && fact_val_K.defined() && fact_val_K.numel() > 0);
        auto fact_pos_c = has_fact ? (fact_pos.is_contiguous() ? fact_pos : fact_pos.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt16).device(Q.device()));
        auto fact_val_K_c = has_fact ? (fact_val_K.is_contiguous() ? fact_val_K : fact_val_K.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        auto fact_val_V_c = has_fact ? (fact_val_V.is_contiguous() ? fact_val_V : fact_val_V.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));

        // Get the active PyTorch MPS stream to queue the execution
        at::mps::MPSStream* mps_stream = at::mps::getCurrentMPSStream();
        // Commit any active command encoder and flush the stream to get a clean command buffer.
        // Doing this AFTER the contiguous operations commits any command encoder PyTorch opened.
        mps_stream->synchronize(at::mps::SyncType::COMMIT);

        // Retrieve internal Metal storage buffers from ATen tensors
        id<MTLBuffer> buf_q = at::native::mps::getMTLBufferStorage(Q_c);
        id<MTLBuffer> buf_u = at::native::mps::getMTLBufferStorage(U_c);
        id<MTLBuffer> buf_u_scale = at::native::mps::getMTLBufferStorage(U_scale_c);
        id<MTLBuffer> buf_vk = at::native::mps::getMTLBufferStorage(VK_c);
        id<MTLBuffer> buf_vv = at::native::mps::getMTLBufferStorage(VV_c);
        id<MTLBuffer> buf_ak = at::native::mps::getMTLBufferStorage(AK_c);
        id<MTLBuffer> buf_av = at::native::mps::getMTLBufferStorage(AV_c);
        id<MTLBuffer> buf_slens = at::native::mps::getMTLBufferStorage(slens_c);
        id<MTLBuffer> buf_scales = at::native::mps::getMTLBufferStorage(scales_c);
        id<MTLBuffer> buf_slots = at::native::mps::getMTLBufferStorage(slots_c);
        id<MTLBuffer> buf_out = at::native::mps::getMTLBufferStorage(out);
        id<MTLBuffer> buf_lse = at::native::mps::getMTLBufferStorage(lse);
        id<MTLBuffer> buf_cos = at::native::mps::getMTLBufferStorage(cos_c);
        id<MTLBuffer> buf_sin = at::native::mps::getMTLBufferStorage(sin_c);

        id<MTLBuffer> buf_dense_k = at::native::mps::getMTLBufferStorage(dense_K_c);
        id<MTLBuffer> buf_dense_v = at::native::mps::getMTLBufferStorage(dense_V_c);
        id<MTLBuffer> buf_cos_dense = at::native::mps::getMTLBufferStorage(cos_dense_c);
        id<MTLBuffer> buf_sin_dense = at::native::mps::getMTLBufferStorage(sin_dense_c);

        id<MTLBuffer> buf_res_pos_K = at::native::mps::getMTLBufferStorage(res_pos_K_c);
        id<MTLBuffer> buf_res_val_K = at::native::mps::getMTLBufferStorage(res_val_K_c);
        id<MTLBuffer> buf_res_pos_V = at::native::mps::getMTLBufferStorage(res_pos_V_c);
        id<MTLBuffer> buf_res_val_V = at::native::mps::getMTLBufferStorage(res_val_V_c);
        id<MTLBuffer> buf_fact_pos   = at::native::mps::getMTLBufferStorage(fact_pos_c);
        id<MTLBuffer> buf_fact_val_K = at::native::mps::getMTLBufferStorage(fact_val_K_c);
        id<MTLBuffer> buf_fact_val_V = at::native::mps::getMTLBufferStorage(fact_val_V_c);

        id<MTLBuffer> buf_cos_full   = at::native::mps::getMTLBufferStorage(cos_full_c);
        id<MTLBuffer> buf_sin_full   = at::native::mps::getMTLBufferStorage(sin_full_c);
        id<MTLBuffer> buf_anchor_pos = at::native::mps::getMTLBufferStorage(anchor_pos_c);

        // Compute byte offsets from storage offsets
        size_t off_q = Q_c.storage_offset() * Q_c.element_size();
        size_t off_u = U_c.storage_offset() * U_c.element_size();
        size_t off_u_scale = U_scale_c.storage_offset() * U_scale_c.element_size();
        size_t off_vk = VK_c.storage_offset() * VK_c.element_size();
        size_t off_vv = VV_c.storage_offset() * VV_c.element_size();
        size_t off_ak = AK_c.storage_offset() * AK_c.element_size();
        size_t off_av = AV_c.storage_offset() * AV_c.element_size();
        size_t off_slens = slens_c.storage_offset() * slens_c.element_size();
        size_t off_scales = scales_c.storage_offset() * scales_c.element_size();
        size_t off_slots = slots_c.storage_offset() * slots_c.element_size();
        size_t off_out = out.storage_offset() * out.element_size();
        size_t off_lse = lse.storage_offset() * lse.element_size();
        size_t off_cos = cos_c.storage_offset() * cos_c.element_size();
        size_t off_sin = sin_c.storage_offset() * sin_c.element_size();

        size_t off_dense_k = dense_K_c.storage_offset() * dense_K_c.element_size();
        size_t off_dense_v = dense_V_c.storage_offset() * dense_V_c.element_size();
        size_t off_cos_dense = cos_dense_c.storage_offset() * cos_dense_c.element_size();
        size_t off_sin_dense = sin_dense_c.storage_offset() * sin_dense_c.element_size();

        size_t off_res_pos_K = res_pos_K_c.storage_offset() * res_pos_K_c.element_size();
        size_t off_res_val_K = res_val_K_c.storage_offset() * res_val_K_c.element_size();
        size_t off_res_pos_V = res_pos_V_c.storage_offset() * res_pos_V_c.element_size();
        size_t off_res_val_V = res_val_V_c.storage_offset() * res_val_V_c.element_size();
        size_t off_fact_pos   = fact_pos_c.storage_offset() * fact_pos_c.element_size();
        size_t off_fact_val_K = fact_val_K_c.storage_offset() * fact_val_K_c.element_size();
        size_t off_fact_val_V = fact_val_V_c.storage_offset() * fact_val_V_c.element_size();

        size_t off_cos_full   = cos_full_c.storage_offset() * cos_full_c.element_size();
        size_t off_sin_full   = sin_full_c.storage_offset() * sin_full_c.element_size();
        size_t off_anchor_pos = anchor_pos_c.storage_offset() * anchor_pos_c.element_size();

        id<MTLBuffer> buf_dbg = at::native::mps::getMTLBufferStorage(dbg);
        size_t off_dbg = dbg.storage_offset() * dbg.element_size();

        id<MTLCommandBuffer> commandBuffer = mps_stream->commandBuffer();
        if (!commandBuffer) {
            throw std::runtime_error("Failed to retrieve active PyTorch MPS command buffer!");
        }

        // Create a compute command encoder from the active command buffer
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            throw std::runtime_error("Failed to create MTLComputeCommandEncoder!");
        }

        [encoder setComputePipelineState:mps_pipeline.pipelineState];

        // Bind raw Metal buffers
        [encoder setBuffer:buf_q offset:off_q atIndex:0];
        [encoder setBuffer:buf_u offset:off_u atIndex:1];
        [encoder setBuffer:buf_u_scale offset:off_u_scale atIndex:2];
        [encoder setBuffer:buf_vk offset:off_vk atIndex:3];
        [encoder setBuffer:buf_vv offset:off_vv atIndex:4];
        [encoder setBuffer:buf_ak offset:off_ak atIndex:5];
        [encoder setBuffer:buf_av offset:off_av atIndex:6];
        [encoder setBuffer:buf_slens offset:off_slens atIndex:7];
        [encoder setBuffer:buf_slots offset:off_slots atIndex:8];
        [encoder setBuffer:buf_out offset:off_out atIndex:9];
        [encoder setBuffer:buf_lse offset:off_lse atIndex:10];

        // Bind uniform parameters directly as bytes using AttentionParams struct
        int S_max = U_c.size(1);
        // Real per-slot rank-dim width of VK_pool/VV_pool/U_pool -- see struct
        // comment above. Derived from the tensor itself (same pattern as
        // S_max) rather than trusted from the caller, since `rank` (the
        // logical/active rank) and this can legitimately differ per layer.
        int pool_rank = VK_c.size(1);
        AttentionParams params;
        params.n_q_heads = n_q_heads;
        params.n_kv_heads = n_kv_heads;
        params.rank = rank;
        params.S_max = S_max;
        params.K = K_slots;
        params.D = D;
        params.scale = scale;
        params.has_rope = has_rope_flag;
        params.max_residual = max_res_val;
        params.L_dense = L_dense;
        params.rotary_dim = (rotary_dim < 0) ? D : rotary_dim;
        params.pool_rank = pool_rank;
        params.has_dense_rope = has_dense_rope ? 1 : 0;
        params.has_fact = has_fact ? 1 : 0;
        params.has_exact_res_rope = has_exact_res_rope ? 1 : 0;
        params.rope_full_rows = rope_full_rows;
        params.residual_exact_keys = dkv_residual_exact_keys();
        // dense_K/dense_V are a FIXED-SIZE workspace padded to max_dense_len;
        // only the leading rows are real tokens. The host sizes cos_dense to
        // exactly that valid count, so it is the authoritative source. Without
        // this the kernel treated the padded stride as the token count --
        // attending padding rows as real tokens AND reading cos/sin past their
        // end. Fall back to L_dense only when there is no dense rope table to
        // derive it from (then the two genuinely coincide).
        int L_dense_valid = L_dense;
        if (has_dense_rope) {
            const int cd_rows = static_cast<int>(cos_dense_c.size(-2));
            if (cd_rows > 0 && cd_rows <= L_dense) {
                L_dense_valid = cd_rows;
            }
        }
        params.L_dense_valid = L_dense_valid;
        params.dense_strict_valid = dkv_dense_strict_valid();
        const int dbg_gpuread = dbg_gpuread_enabled();
        params.debug_gpuread = dbg_gpuread;
        // (A warning used to live here: a dense window longer than
        // dense_w_shared's 768 entries had every token past row 767 silently
        // dropped from attention. The kernel now walks the dense window in tiles
        // of that width, so the shared buffer bounds only the TILE, never how
        // much context is attended -- and the warning would be false. It was also
        // gated on dense_strict_valid, which defaults OFF, so the case it warned
        // about was exactly the case where it stayed quiet.)

        [encoder setBytes:&params length:sizeof(AttentionParams) atIndex:11];
        
        [encoder setBuffer:buf_scales offset:off_scales atIndex:12];
        // RoPE buffers: cos_anc [K, D] float32 and sin_anc [K, D] float32
        [encoder setBuffer:buf_cos offset:off_cos atIndex:13];
        [encoder setBuffer:buf_sin offset:off_sin atIndex:14];

        // Bind Track D Residual and Fact Override Buffers
        [encoder setBuffer:buf_res_pos_K offset:off_res_pos_K atIndex:15];
        [encoder setBuffer:buf_res_val_K offset:off_res_val_K atIndex:16];
        [encoder setBuffer:buf_res_pos_V offset:off_res_pos_V atIndex:17];
        [encoder setBuffer:buf_res_val_V offset:off_res_val_V atIndex:18];
        [encoder setBuffer:buf_fact_pos   offset:off_fact_pos   atIndex:19];
        [encoder setBuffer:buf_fact_val_K offset:off_fact_val_K atIndex:20];
        [encoder setBuffer:buf_fact_val_V offset:off_fact_val_V atIndex:21];

        // Bind dense window buffers
        [encoder setBuffer:buf_dense_k offset:off_dense_k atIndex:22];
        [encoder setBuffer:buf_dense_v offset:off_dense_v atIndex:23];
        [encoder setBuffer:buf_cos_dense offset:off_cos_dense atIndex:24];
        [encoder setBuffer:buf_sin_dense offset:off_sin_dense atIndex:25];

        // Full-sequence rope tables + anchor positions (exact-position residual/fact rope)
        [encoder setBuffer:buf_cos_full   offset:off_cos_full   atIndex:26];
        [encoder setBuffer:buf_sin_full   offset:off_sin_full   atIndex:27];
        [encoder setBuffer:buf_anchor_pos offset:off_anchor_pos atIndex:28];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(dbg)
                    offset:(dbg.storage_offset() * dbg.element_size()) atIndex:29];

        // Threadgroup grid: [n_q_heads, 1, 1] (1 threadgroup per head)
        MTLSize grid = MTLSizeMake(n_q_heads, 1, 1);
        
        // Threadgroup size: 64 threads (covers typical warp sizes for maximum occupancy)
        //
        // DKV_DEBUG_TG1=1 forces ONE thread per threadgroup — DIAGNOSTIC ONLY,
        // and very slow. With a single thread there is no intra-threadgroup
        // concurrency at all, so a missing `threadgroup_barrier` between a shared
        // write and a shared read cannot manifest. Every loop here is written as
        // `for (i = tid; i < N; i += t_per_tg)` and every reduction strides by
        // `t_per_tg / 2`, so one thread is still functionally correct: it simply
        // does all the work itself and the reduction loops become no-ops.
        //
        // This is the last untested mechanism for handoff §10f (byte-identical
        // arguments producing different returns, with GPU ordering, threadgroup
        // initialisation and buffer aliasing all already eliminated). If the
        // output becomes stable at 1 thread, a barrier is missing somewhere; if it
        // still varies, intra-threadgroup racing is not the cause either.
        //
        // CONTRACT with the kernel's thread_val accumulator: a thread owns dims
        // tid, tid + tg_threads, ... and holds them in a private array of
        // DKV_MAX_VAL_PER_THREAD (=8) floats, so tg_threads must be at least
        // ceil(D / 8) or the kernel drops the tail dims. 64 satisfies that up to
        // D=512. The array is deliberately NOT sized [D]: at D=256 that reserved
        // 256 floats in every one of the 64 threads (64 KB of thread-private
        // memory per threadgroup) to hold 4 useful values each.
        const int kMaxValPerThread = 8;   // must match DKV_MAX_VAL_PER_THREAD
        int tg_threads = 64;
        {
            static int dbg_tg1 = -1;
            if (dbg_tg1 < 0) {
                const char* e = std::getenv("DKV_DEBUG_TG1");
                dbg_tg1 = (e && e[0] == '1') ? 1 : 0;
            }
            if (dbg_tg1) tg_threads = 1;
        }
        {
            const int min_threads = (D + kMaxValPerThread - 1) / kMaxValPerThread;
            if (tg_threads < min_threads) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    fprintf(stderr,
                            "[dkv] threadgroup size %d is below the %d required for "
                            "head_dim=%d; raising it (DKV_DEBUG_TG1 cannot be honoured "
                            "at this head_dim).\n",
                            tg_threads, min_threads, D);
                }
                tg_threads = min_threads;
            }
        }
        MTLSize threadgroup = MTLSizeMake(tg_threads, 1, 1);

        // DKV_DEBUG_ALIAS=1 — DIAGNOSTIC ONLY.
        //
        // Does any buffer bound to this kernel overlap any other WITHIN THE SAME
        // MTLBuffer? `out` and `lse` come from torch::empty via the MPS caching
        // allocator; if one were handed a block still live as an input, the
        // kernel would write over data it is concurrently reading — which is
        // per-threadgroup order dependent and reproduces exactly as: identical
        // arguments, different return, deterministic in isolation (handoff §10f).
        //
        // This has to be done here, not from Python: `data_ptr()` on MPS does not
        // linearly encode the sub-allocation, so range arithmetic over it is
        // meaningless (two freshly-allocated distinct tensors "overlap"). The
        // real test is buffer IDENTITY plus [offset, offset+len).
        {
            static int dbg_alias = -1;
            if (dbg_alias < 0) {
                const char* e = std::getenv("DKV_DEBUG_ALIAS");
                dbg_alias = (e && e[0] == '1') ? 1 : 0;
            }
            if (dbg_alias) {
                struct Bnd { const char* name; id<MTLBuffer> buf; size_t off; size_t len; };
                std::vector<Bnd> bs = {
                    {"Q", buf_q, off_q, (size_t)(Q_c.numel() * Q_c.element_size())},
                    {"out", buf_out, off_out, (size_t)(out.numel() * out.element_size())},
                    {"lse", buf_lse, off_lse, (size_t)(lse.numel() * lse.element_size())},
                    {"dense_K", buf_dense_k, off_dense_k, (size_t)(dense_K_c.numel() * dense_K_c.element_size())},
                    {"dense_V", buf_dense_v, off_dense_v, (size_t)(dense_V_c.numel() * dense_V_c.element_size())},
                    {"U", buf_u, off_u, (size_t)(U_c.numel() * U_c.element_size())},
                    {"VK", buf_vk, off_vk, (size_t)(VK_c.numel() * VK_c.element_size())},
                    {"VV", buf_vv, off_vv, (size_t)(VV_c.numel() * VV_c.element_size())},
                    {"AK", buf_ak, off_ak, (size_t)(AK_c.numel() * AK_c.element_size())},
                    {"AV", buf_av, off_av, (size_t)(AV_c.numel() * AV_c.element_size())},
                    {"cos_anc", buf_cos, off_cos, (size_t)(cos_c.numel() * cos_c.element_size())},
                    {"sin_anc", buf_sin, off_sin, (size_t)(sin_c.numel() * sin_c.element_size())},
                };
                int found = 0;
                for (size_t i = 0; i < bs.size(); ++i) {
                    for (size_t j = i + 1; j < bs.size(); ++j) {
                        if (bs[i].buf != bs[j].buf) continue;      // different MTLBuffer: cannot alias
                        size_t a0 = bs[i].off, a1 = a0 + bs[i].len;
                        size_t b0 = bs[j].off, b1 = b0 + bs[j].len;
                        if (a0 < b1 && b0 < a1) {
                            ++found;
                            std::cerr << "[DKV ALIAS] " << bs[i].name << "[" << a0 << "," << a1
                                      << ") OVERLAPS " << bs[j].name << "[" << b0 << "," << b1
                                      << ") in MTLBuffer " << (void*)bs[i].buf << std::endl;
                        }
                    }
                }
                std::cerr << "[DKV ALIAS] overlaps=" << found
                          << " out.buf=" << (void*)buf_out << " off=" << off_out
                          << " denseK.buf=" << (void*)buf_dense_k << " off=" << off_dense_k
                          << std::endl;
            }
        }

        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        if (dbg_gpuread) {
            g_last_dbg = dbg;
        }

        // NOTE: the debug buffer is deliberately NOT read here. Doing so
        // required synchronize(COMMIT_AND_WAIT) followed by dbg.to(kCPU), which
        // asks the stream to encode a blit onto the buffer that was just
        // committed -- reliably SIGSEGV. It is returned to the caller instead
        // (third output below) and read from Python once the stream has settled.
        // That also drops the COMMIT_AND_WAIT, so the probe observes the DEFAULT
        // execution mode rather than a serialised one.

        // ── Keep every input tensor alive until the GPU has actually run ──
        //
        // We deliberately do NOT commit/wait here, so at this point the kernel is
        // only ENQUEUED. The `_c` locals above go out of scope at the closing
        // brace. Any of them produced by `.contiguous()` owns a fresh allocation,
        // and freeing it returns the block to PyTorch's MPS caching allocator,
        // which will hand it to a later op — which can then overwrite it BEFORE
        // this kernel executes. The kernel reads recycled memory.
        //
        // This is not hypothetical: `Q` at the decode call site is
        // `query_states[b_idx, :, 0, :]`, a strided slice, so it ALWAYS takes the
        // copy path. The failure is intermittent (it depends on whether the
        // allocator reuses that block before the queue drains), input-independent,
        // and invisible to the isolated tests — those call `.cpu()` right after,
        // and nothing allocates in between, so the block is never recycled.
        //
        // Retain them on the command buffer's completion handler: the tensors
        // stay referenced until the GPU signals done, and nothing else changes —
        // still no commit, still pipelined.
        auto keepalive = std::make_shared<std::vector<torch::Tensor>>(
            std::initializer_list<torch::Tensor>{
                Q_c, U_c, U_scale_c, VK_c, VV_c, AK_c, AV_c, slens_c, scales_c,
                slots_c, cos_c, sin_c, dense_K_c, dense_V_c,
                res_pos_K_c, res_val_K_c, res_pos_V_c, res_val_V_c,
                fact_pos_c, fact_val_K_c, fact_val_V_c,
                cos_dense_c, sin_dense_c, cos_full_c, sin_full_c, anchor_pos_c,
                out, lse, dbg});
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
            keepalive->clear();
        }];

        // DKV_DEBUG_COMMIT_WAIT=1 — DIAGNOSTIC ONLY, never ship it on.
        //
        // Commits this command buffer and blocks until the GPU has finished,
        // which removes ALL overlap between this kernel and anything around it.
        // It exists to partition one remaining question: layer 7's decode output
        // varies run to run even though the ENTIRE observable host state going
        // into it is byte-identical (handoff §10d), and every probe hashes
        // through .cpu() -- which synchronises -- so no host-side probe can rule
        // out a GPU read-before-write.
        //
        //   distribution collapses to one value -> it IS an ordering bug, and the
        //       fix is a correct barrier around whichever buffer is racing;
        //   distribution still varies          -> ordering is NOT the cause, and
        //       the remaining suspect is uninitialised threadgroup memory read on
        //       a path the isolated suite does not exercise.
        //
        // Costs all pipelining, so it is a measurement instrument, not a fix.
        {
            static int commit_wait = -1;
            if (commit_wait < 0) {
                const char* e = std::getenv("DKV_DEBUG_COMMIT_WAIT");
                commit_wait = (e && e[0] == '1') ? 1 : 0;
            }
            if (commit_wait) {
                mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
            }
        }
    }

    return {out, lse};
}

} // namespace dkv
