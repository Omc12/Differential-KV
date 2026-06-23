#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "runtime/diffkv_attention.hpp"
#include <iostream>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mach-o/dyld.h>
#include <unordered_map>
#include <mutex>
#include <dispatch/dispatch.h>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static id<MTLBuffer> g_dummy_rope_buf = nil;
static double g_accumulated_wait_ms = 0.0;

// ── Pipelined per-layer semaphores ─────────────────────────────────────────────
// Instead of waitUntilCompleted on every layer (blocking CPU for ~2ms × 28 = 56ms/token),
// we signal each layer's semaphore from its completion handler and wait on it at the
// START of the NEXT layer's callback. The FFN compute (~1-2ms) runs between callbacks,
// so by the time we wait the sparse kernel (~0.05ms) is long done → near-zero stall.
// Layer 0 pre-signals its own semaphore so it never blocks.
static constexpr int MAX_LAYERS = 64;
static dispatch_semaphore_t g_layer_sems[MAX_LAYERS];  // one per layer
static int                  g_n_layer_sems = 0;
static std::once_flag       g_sems_init_flag;

static void ensure_layer_sems(int n_layers) {
    std::call_once(g_sems_init_flag, [n_layers]() {
        int n = std::min(n_layers, MAX_LAYERS);
        for (int i = 0; i < n; ++i)
            g_layer_sems[i] = dispatch_semaphore_create(1); // pre-signaled so layer 0 never waits
        g_n_layer_sems = n;
    });
}

// ── Shared pool Metal buffer cache ────────────────────────────────────────────
// Pool data (U, VK, VV, anchors, seq_lens, scales, anchor_positions) is IDENTICAL
// across all 28 transformer layers for a given session — they all read from the same
// NativeBlockPool. Previously each of the 28 CustomAttnUserData objects had its own
// copies, so every pool version bump (background SVD finishing a block) triggered
// 28 independent memcpys of ~17 MB each = ~500 MB total → stutter every new block.
//
// Fix: one shared set of MTLBuffers per pool pointer. All 28 layers read from the
// same buffers. Pool data is copied exactly ONCE per version bump.
struct GlobalPoolMtlBufs {
    id<MTLBuffer> u_pool    = nil;
    id<MTLBuffer> u_scale   = nil;
    id<MTLBuffer> vk_pool   = nil;
    id<MTLBuffer> vv_pool   = nil;
    id<MTLBuffer> anchors_k = nil;
    id<MTLBuffer> anchors_v = nil;
    id<MTLBuffer> seq_lens  = nil;
    id<MTLBuffer> scales    = nil;
    id<MTLBuffer> anc_pos   = nil;
    int           pool_ver  = -1;
};
static std::unordered_map<void*, GlobalPoolMtlBufs> g_pool_buf_cache;
static std::mutex g_pool_buf_mutex;

static void init_metal_runtime() {
    if (g_device != nil) return;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        std::cerr << "[Metal Attention] Error: MTLCreateSystemDefaultDevice returned nil!" << std::endl;
        return;
    }
    g_queue = [g_device newCommandQueue];

    // Dummy non-nil MTLBuffer used as placeholder for optional buffers (e.g. empty
    // slot_indices, or dense buffers when T_dense == 0).  Must be non-nil to avoid
    // GPU command-encoder exceptions.
    g_dummy_rope_buf = [g_device newBufferWithLength:65536 options:MTLResourceStorageModeShared];

    NSError* error = nil;
    NSString* source = nil;
    NSString* path = nil;

    // 1. Try relative to the executable's path
    char exec_path[1024];
    uint32_t sz = sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &sz) == 0) {
        NSString* binPath = [NSString stringWithUTF8String:exec_path];
        NSString* binDir  = [binPath stringByDeletingLastPathComponent];
        path   = [binDir stringByAppendingPathComponent:@"../native_core/diffkv_core/metal/diffkv_decode.metal"];
        source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (!source) {
            path   = [binDir stringByAppendingPathComponent:@"native_core/diffkv_core/metal/diffkv_decode.metal"];
            source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        }
    }

    // 2. Try standard relative paths from CWD
    if (!source) {
        path   = @"../native_core/diffkv_core/metal/diffkv_decode.metal";
        source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    }
    if (!source) {
        path   = @"native_core/diffkv_core/metal/diffkv_decode.metal";
        source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    }
    if (!source) {
        path   = @"diffkv_native/native_core/diffkv_core/metal/diffkv_decode.metal";
        source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    }

    // 3. Absolute path fallback
    if (!source) {
        path   = @"/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/native_core/diffkv_core/metal/diffkv_decode.metal";
        source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    }

    if (!source) {
        std::cerr << "[Metal Attention] Error: Could not load diffkv_decode.metal from any known location!" << std::endl;
        return;
    }
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "[Metal Attention] Error compiling Metal library: "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"decode_attention_metal_kernel"];
    g_pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!g_pipeline) {
        std::cerr << "[Metal Attention] Error creating Compute Pipeline State: "
                  << [[error localizedDescription] UTF8String] << std::endl;
    } else {
        std::cerr << "[Metal Attention] Successfully initialized Metal runtime and compiled shader library." << std::endl;
    }
}

// Wrap a GGML tensor's backing memory as a Metal shared buffer (zero-copy when page-aligned).
static id<MTLBuffer> wrap_tensor(struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nil;
    size_t bytes = ggml_nbytes(tensor);
    size_t aligned_len = (bytes + 4095) & ~4095;
    if (((uintptr_t)tensor->data % 4096) == 0) {
        id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:tensor->data
                                                         length:aligned_len
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (buf) return buf;
    }
    return [g_device newBufferWithBytes:tensor->data length:bytes options:MTLResourceStorageModeShared];
}

// Wrap a raw CPU pointer as a Metal shared buffer.
// Returns g_dummy_rope_buf if ptr is null or size is 0 (avoids nil-buffer GPU exceptions).
static id<MTLBuffer> wrap_cpu_ptr(const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return g_dummy_rope_buf;
    if (((uintptr_t)ptr % 4096) == 0) {
        size_t aligned_len = (bytes + 4095) & ~4095;
        id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:(void*)ptr
                                                         length:aligned_len
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (buf) return buf;
    }
    return [g_device newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

// Wrap output tensor: zero-copy if page-aligned, otherwise allocate aligned temp and copy back.
static id<MTLBuffer> wrap_output_tensor(struct ggml_tensor* tensor, void** out_temp_ptr) {
    *out_temp_ptr = nullptr;
    if (!tensor || !tensor->data) return nil;
    size_t bytes = ggml_nbytes(tensor);
    if (((uintptr_t)tensor->data % 4096) == 0) {
        size_t aligned_len = (bytes + 4095) & ~4095;
        return [g_device newBufferWithBytesNoCopy:tensor->data
                                           length:aligned_len
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    } else {
        void* temp_mem = nullptr;
        if (posix_memalign(&temp_mem, 4096, (bytes + 4095) & ~4095) != 0) {
            std::cerr << "[Metal Attention] Error: posix_memalign failed!" << std::endl;
            return nil;
        }
        *out_temp_ptr = temp_mem;
        return [g_device newBufferWithBytesNoCopy:temp_mem
                                           length:(bytes + 4095) & ~4095
                                          options:MTLResourceStorageModeShared
                                      deallocator:^(void* pointer, NSUInteger length) {
                                          free(pointer);
                                      }];
    }
}

namespace diffkv {

CustomAttnUserData::CustomAttnUserData()
    : kv_engine(nullptr), session_id(""), layer_idx(-1), slot_indices(nullptr), n_q_heads(0), n_kv_heads(0),
      rank(0), S_max(0), K(0), D(0), scale(0.0f), has_rope(false), rope_freq_base(0.0f), approximate_attn(false),
      active_k_dense(nullptr), active_v_dense(nullptr), active_positions_dense(nullptr),
      active_block_tokens(0), active_slot(0), ignore_c(false), current_pos(0),
      srl_state(nullptr), W_proj(nullptr), desc_dim(0), max_active_dense_tokens(16384),
      mtl_dense_k(nullptr), mtl_dense_v(nullptr), mtl_dense_pos(nullptr),
      mtl_slot_indices(nullptr), mtl_output_buf(nullptr), mtl_lse_buf(nullptr),
      mtl_q_buf(nullptr), mtl_u_pool(nullptr), mtl_u_scale(nullptr), mtl_vk_pool(nullptr), mtl_vv_pool(nullptr),
      mtl_anchors_k(nullptr), mtl_anchors_v(nullptr), mtl_seq_lens(nullptr),
      mtl_scales(nullptr), mtl_anc_pos(nullptr), last_seen_pool_version(-1) {}

CustomAttnUserData::~CustomAttnUserData() {
    if (mtl_dense_k) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_dense_k;
        buf = nil;
    }
    if (mtl_dense_v) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_dense_v;
        buf = nil;
    }
    if (mtl_dense_pos) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_dense_pos;
        buf = nil;
    }
    if (mtl_slot_indices) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_slot_indices;
        buf = nil;
    }
    if (mtl_output_buf) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_output_buf;
        buf = nil;
    }
    if (mtl_lse_buf) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_lse_buf;
        buf = nil;
    }
    if (mtl_q_buf) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_q_buf;
        buf = nil;
    }
    if (mtl_u_pool) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_u_pool;
        buf = nil;
    }
    if (mtl_u_scale) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_u_scale;
        buf = nil;
    }
    if (mtl_vk_pool) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_vk_pool;
        buf = nil;
    }
    if (mtl_vv_pool) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_vv_pool;
        buf = nil;
    }
    if (mtl_anchors_k) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_anchors_k;
        buf = nil;
    }
    if (mtl_anchors_v) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_anchors_v;
        buf = nil;
    }
    if (mtl_seq_lens) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_seq_lens;
        buf = nil;
    }
    if (mtl_scales) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_scales;
        buf = nil;
    }
    if (mtl_anc_pos) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_anc_pos;
        buf = nil;
    }
}

CustomAttnUserData::CustomAttnUserData(CustomAttnUserData&& other) noexcept {
    kv_engine = other.kv_engine;
    session_id = std::move(other.session_id);
    layer_idx = other.layer_idx;
    slot_indices = other.slot_indices;
    n_q_heads = other.n_q_heads;
    n_kv_heads = other.n_kv_heads;
    rank = other.rank;
    S_max = other.S_max;
    K = other.K;
    D = other.D;
    scale = other.scale;
    has_rope = other.has_rope;
    rope_freq_base = other.rope_freq_base;
    approximate_attn = other.approximate_attn;
    active_k_dense = other.active_k_dense;
    active_v_dense = other.active_v_dense;
    active_positions_dense = other.active_positions_dense;
    active_block_tokens = other.active_block_tokens;
    active_slot = other.active_slot;
    ignore_c = other.ignore_c;
    current_pos = other.current_pos;
    srl_state = other.srl_state;
    W_proj = other.W_proj;
    desc_dim = other.desc_dim;
    max_active_dense_tokens = other.max_active_dense_tokens;

    mtl_dense_k = other.mtl_dense_k;
    mtl_dense_v = other.mtl_dense_v;
    mtl_dense_pos = other.mtl_dense_pos;
    mtl_slot_indices = other.mtl_slot_indices;
    mtl_output_buf = other.mtl_output_buf;
    mtl_lse_buf = other.mtl_lse_buf;

    mtl_q_buf = other.mtl_q_buf;
    mtl_u_pool = other.mtl_u_pool;
    mtl_u_scale = other.mtl_u_scale;
    mtl_vk_pool = other.mtl_vk_pool;
    mtl_vv_pool = other.mtl_vv_pool;
    mtl_anchors_k = other.mtl_anchors_k;
    mtl_anchors_v = other.mtl_anchors_v;
    mtl_seq_lens = other.mtl_seq_lens;
    mtl_scales = other.mtl_scales;
    mtl_anc_pos = other.mtl_anc_pos;

    other.mtl_dense_k = nullptr;
    other.mtl_dense_v = nullptr;
    other.mtl_dense_pos = nullptr;
    other.mtl_slot_indices = nullptr;
    other.mtl_output_buf = nullptr;
    other.mtl_lse_buf = nullptr;

    other.mtl_q_buf = nullptr;
    other.mtl_u_pool = nullptr;
    other.mtl_u_scale = nullptr;
    other.mtl_vk_pool = nullptr;
    other.mtl_vv_pool = nullptr;
    other.mtl_anchors_k = nullptr;
    other.mtl_anchors_v = nullptr;
    other.mtl_seq_lens = nullptr;
    other.mtl_scales = nullptr;
    other.mtl_anc_pos = nullptr;
}

CustomAttnUserData& CustomAttnUserData::operator=(CustomAttnUserData&& other) noexcept {
    if (this != &other) {
        if (mtl_dense_k) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_dense_k;
            buf = nil;
        }
        if (mtl_dense_v) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_dense_v;
            buf = nil;
        }
        if (mtl_dense_pos) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_dense_pos;
            buf = nil;
        }
        if (mtl_slot_indices) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_slot_indices;
            buf = nil;
        }
        if (mtl_output_buf) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_output_buf;
            buf = nil;
        }
        if (mtl_lse_buf) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_lse_buf;
            buf = nil;
        }
        if (mtl_q_buf) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_q_buf;
            buf = nil;
        }
        if (mtl_u_pool) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_u_pool;
            buf = nil;
        }
        if (mtl_u_scale) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_u_scale;
            buf = nil;
        }
        if (mtl_vk_pool) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_vk_pool;
            buf = nil;
        }
        if (mtl_vv_pool) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_vv_pool;
            buf = nil;
        }
        if (mtl_anchors_k) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_anchors_k;
            buf = nil;
        }
        if (mtl_anchors_v) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_anchors_v;
            buf = nil;
        }
        if (mtl_seq_lens) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_seq_lens;
            buf = nil;
        }
        if (mtl_scales) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_scales;
            buf = nil;
        }
        if (mtl_anc_pos) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mtl_anc_pos;
            buf = nil;
        }

        kv_engine = other.kv_engine;
        session_id = std::move(other.session_id);
        layer_idx = other.layer_idx;
        slot_indices = other.slot_indices;
        n_q_heads = other.n_q_heads;
        n_kv_heads = other.n_kv_heads;
        rank = other.rank;
        S_max = other.S_max;
        K = other.K;
        D = other.D;
        scale = other.scale;
        has_rope = other.has_rope;
        rope_freq_base = other.rope_freq_base;
        approximate_attn = other.approximate_attn;
        active_k_dense = other.active_k_dense;
        active_v_dense = other.active_v_dense;
        active_positions_dense = other.active_positions_dense;
        active_block_tokens = other.active_block_tokens;
        active_slot = other.active_slot;
        ignore_c = other.ignore_c;
        current_pos = other.current_pos;
        srl_state = other.srl_state;
        W_proj = other.W_proj;
        desc_dim = other.desc_dim;
        max_active_dense_tokens = other.max_active_dense_tokens;

        mtl_dense_k = other.mtl_dense_k;
        mtl_dense_v = other.mtl_dense_v;
        mtl_dense_pos = other.mtl_dense_pos;
        mtl_slot_indices = other.mtl_slot_indices;
        mtl_output_buf = other.mtl_output_buf;
        mtl_lse_buf = other.mtl_lse_buf;

        mtl_q_buf = other.mtl_q_buf;
        mtl_u_pool = other.mtl_u_pool;
        mtl_u_scale = other.mtl_u_scale;
        mtl_vk_pool = other.mtl_vk_pool;
        mtl_vv_pool = other.mtl_vv_pool;
        mtl_anchors_k = other.mtl_anchors_k;
        mtl_anchors_v = other.mtl_anchors_v;
        mtl_seq_lens = other.mtl_seq_lens;
        mtl_scales = other.mtl_scales;
        mtl_anc_pos = other.mtl_anc_pos;

        other.mtl_dense_k = nullptr;
        other.mtl_dense_v = nullptr;
        other.mtl_dense_pos = nullptr;
        other.mtl_slot_indices = nullptr;
        other.mtl_output_buf = nullptr;
        other.mtl_lse_buf = nullptr;

        other.mtl_q_buf = nullptr;
        other.mtl_u_pool = nullptr;
        other.mtl_u_scale = nullptr;
        other.mtl_vk_pool = nullptr;
        other.mtl_vv_pool = nullptr;
        other.mtl_anchors_k = nullptr;
        other.mtl_anchors_v = nullptr;
        other.mtl_seq_lens = nullptr;
        other.mtl_scales = nullptr;
        other.mtl_anc_pos = nullptr;
    }
    return *this;
}

// Unified GPU decode attention.
// Dispatches a single Metal kernel per token that computes sparse compressed
// block attention (Project-Then-Attend) AND dense window attention together,
// writing the fully combined result into dst.
//
// dense_K / dense_V : float32 CPU buffers [T_dense × n_kv × D]
// dense_pos         : int32  CPU buffer   [T_dense]  (actual sequence positions)
// T_dense           : number of dense window tokens  (0 → dense path skipped)
void execute_metal_attention(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    struct ggml_tensor * slot_indices,
    CustomAttnUserData * data,
    float* lse_out,
    const float*   dense_K,
    const float*   dense_V,
    const int32_t* dense_pos,
    int            T_dense
) {
    extern std::atomic<long> g_metal_attn_count;
    g_metal_attn_count.fetch_add(1, std::memory_order_relaxed);  // unconditional: is the Metal path live?
    init_metal_runtime();
    if (!g_pipeline) {
        std::cerr << "[Metal Attention] Error: Compute pipeline not initialized!" << std::endl;
        // Zero-out output so the model doesn't crash on garbage data
        if (dst && dst->data) memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }

    const int n_q_heads     = data->n_q_heads;
    const int n_kv_heads    = data->n_kv_heads;
    const int rank          = data->rank;
    const int S_max         = data->S_max;
    const int D             = data->D;
    const float scale       = data->scale;
    const bool has_rope     = data->has_rope;
    const float rope_freq   = data->rope_freq_base;
    NativeBlockPool* engine = data->kv_engine;

    // ── Deduplicate and validate slot indices ──────────────────────────────────
    const int raw_K   = (slot_indices != nullptr) ? (int)slot_indices->ne[0] : 0;
    const int n_slots = engine->get_seq_lens()->ne[0];

    std::vector<int32_t> unique_slots;
    unique_slots.reserve(raw_K);
    if (raw_K > 0 && slot_indices && slot_indices->data) {
        const int32_t* slots_ptr = (const int32_t*)slot_indices->data;
        for (int k = 0; k < raw_K; ++k) {
            int32_t sid = slots_ptr[k];
            if (sid >= 0 && sid < n_slots) {
                if (std::find(unique_slots.begin(), unique_slots.end(), sid) == unique_slots.end()) {
                    unique_slots.push_back(sid);
                }
            }
        }
    }
    const int active_K = (int)unique_slots.size();

    // Nothing to do?  Write zeros and return.
    if (active_K == 0 && T_dense == 0) {
        if (dst && dst->data) memset(dst->data, 0, ggml_nbytes(dst));
        if (lse_out) {
            for (int h = 0; h < n_q_heads; ++h) lse_out[h] = -1e30f;
        }
        return;
    }

    @autoreleasepool {
        // ── Build sparse slot-indices Metal buffer ─────────────────────────────
        id<MTLBuffer> slot_indices_buf = nil;
        if (active_K > 0) {
            if (data->mtl_slot_indices) {
                slot_indices_buf = (__bridge id<MTLBuffer>)data->mtl_slot_indices;
            } else {
                slot_indices_buf = [g_device newBufferWithLength:128 * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared];
                data->mtl_slot_indices = (__bridge_retained void*)slot_indices_buf;
            }
            std::memcpy(slot_indices_buf.contents, unique_slots.data(), active_K * sizeof(int32_t));
        } else {
            slot_indices_buf = g_dummy_rope_buf;
        }

        // ── Dense window Metal buffers (zero-copy when page-aligned) ───────────
        const int   F_kv        = n_kv_heads * D;
        const int   T_clamped   = T_dense;
        
        size_t max_k_bytes   = data->max_active_dense_tokens * F_kv * sizeof(float);
        size_t max_v_bytes   = data->max_active_dense_tokens * F_kv * sizeof(float);
        size_t max_pos_bytes = data->max_active_dense_tokens * sizeof(int32_t);

        id<MTLBuffer> dense_k_buf   = nil;
        id<MTLBuffer> dense_v_buf   = nil;
        id<MTLBuffer> dense_pos_buf = nil;

        if (data->mtl_dense_k) {
            dense_k_buf = (__bridge id<MTLBuffer>)data->mtl_dense_k;
            if (dense_k_buf.contents != dense_K && T_clamped > 0) {
                std::memcpy(dense_k_buf.contents, dense_K, T_clamped * F_kv * sizeof(float));
            }
        } else if (dense_K && T_clamped > 0) {
            dense_k_buf = wrap_cpu_ptr(dense_K, max_k_bytes);
            data->mtl_dense_k = (__bridge_retained void*)dense_k_buf;
            if (dense_k_buf.contents != dense_K) {
                std::memcpy(dense_k_buf.contents, dense_K, T_clamped * F_kv * sizeof(float));
            }
        } else {
            dense_k_buf = g_dummy_rope_buf;
        }

        if (data->mtl_dense_v) {
            dense_v_buf = (__bridge id<MTLBuffer>)data->mtl_dense_v;
            if (dense_v_buf.contents != dense_V && T_clamped > 0) {
                std::memcpy(dense_v_buf.contents, dense_V, T_clamped * F_kv * sizeof(float));
            }
        } else if (dense_V && T_clamped > 0) {
            dense_v_buf = wrap_cpu_ptr(dense_V, max_v_bytes);
            data->mtl_dense_v = (__bridge_retained void*)dense_v_buf;
            if (dense_v_buf.contents != dense_V) {
                std::memcpy(dense_v_buf.contents, dense_V, T_clamped * F_kv * sizeof(float));
            }
        } else {
            dense_v_buf = g_dummy_rope_buf;
        }

        if (data->mtl_dense_pos) {
            dense_pos_buf = (__bridge id<MTLBuffer>)data->mtl_dense_pos;
            if (dense_pos_buf.contents != dense_pos && T_clamped > 0) {
                std::memcpy(dense_pos_buf.contents, dense_pos, T_clamped * sizeof(int32_t));
            }
        } else if (dense_pos && T_clamped > 0) {
            dense_pos_buf = wrap_cpu_ptr(dense_pos, max_pos_bytes);
            data->mtl_dense_pos = (__bridge_retained void*)dense_pos_buf;
            if (dense_pos_buf.contents != dense_pos) {
                std::memcpy(dense_pos_buf.contents, dense_pos, T_clamped * sizeof(int32_t));
            }
        } else {
            dense_pos_buf = g_dummy_rope_buf;
        }

        // ── Output and LSE buffers ────────────────────────────────────────────
        void* temp_out_ptr = nullptr;
        id<MTLBuffer> out_buf  = nil;
        if (((uintptr_t)dst->data % 4096) == 0) {
            out_buf = wrap_tensor(dst);
        } else {
            if (data->mtl_output_buf) {
                out_buf = (__bridge id<MTLBuffer>)data->mtl_output_buf;
            } else {
                size_t aligned_len = (ggml_nbytes(dst) + 4095) & ~4095;
                out_buf = [g_device newBufferWithLength:aligned_len options:MTLResourceStorageModeShared];
                data->mtl_output_buf = (__bridge_retained void*)out_buf;
            }
            temp_out_ptr = out_buf.contents;
        }

        id<MTLBuffer> lse_buf = nil;
        if (data->mtl_lse_buf) {
            lse_buf = (__bridge id<MTLBuffer>)data->mtl_lse_buf;
        } else {
            lse_buf = [g_device newBufferWithLength:n_q_heads * sizeof(float)
                                           options:MTLResourceStorageModeShared];
            data->mtl_lse_buf = (__bridge_retained void*)lse_buf;
        }

        // ── Query buffer: zero-copy on Apple Silicon unified memory ─────────────
        // Q->data is in ggml Metal shared memory — wrap it directly without memcpy.
        // On the rare case wrap_tensor falls back to a copy (non-page-aligned), we still
        // need to copy, but this is exceptional. We re-wrap each call so the pointer is fresh.
        id<MTLBuffer> q_buf = wrap_tensor((struct ggml_tensor*)Q);
        if (!q_buf) {
            // Absolute fallback: alloc + copy
            size_t q_bytes = ggml_nbytes((struct ggml_tensor*)Q);
            q_buf = [g_device newBufferWithBytes:Q->data length:q_bytes
                                        options:MTLResourceStorageModeShared];
        }

        // ── Shared pool Metal buffers (1 copy per pool version, shared across all 28 layers) ─
        int cur_pool_ver = engine->get_pool_version();
        id<MTLBuffer> u_pool_buf, u_scale_buf, vk_pool_buf, vv_pool_buf;
        id<MTLBuffer> anchors_k_buf, anchors_v_buf, seq_lens_buf, scales_buf, anc_pos_buf;
        {
            std::lock_guard<std::mutex> lk(g_pool_buf_mutex);
            GlobalPoolMtlBufs& pb = g_pool_buf_cache[(void*)engine];
            bool dirty = (pb.pool_ver != cur_pool_ver);

            auto ensure = [&](__strong id<MTLBuffer>& buf, const void* src, size_t bytes) {
                if (!buf) {
                    buf = [g_device newBufferWithLength:(bytes + 4095) & ~4095
                                               options:MTLResourceStorageModeShared];
                    dirty = true; // force copy on first allocation
                }
                if (dirty) std::memcpy(buf.contents, src, bytes);
            };

            ensure(pb.u_pool,    engine->get_host_U(),                ggml_nbytes(engine->get_U()));
            ensure(pb.u_scale,   engine->get_host_U_scale(),          ggml_nbytes(engine->get_U_scale()));
            ensure(pb.vk_pool,   engine->get_host_VK(),               ggml_nbytes(engine->get_VK()));
            ensure(pb.vv_pool,   engine->get_host_VV(),               ggml_nbytes(engine->get_VV()));
            ensure(pb.anchors_k, engine->get_host_anchors_K(),        ggml_nbytes(engine->get_anchors_K()));
            ensure(pb.anchors_v, engine->get_host_anchors_V(),        ggml_nbytes(engine->get_anchors_V()));
            ensure(pb.seq_lens,  engine->get_host_seq_lens(),         ggml_nbytes(engine->get_seq_lens()));
            ensure(pb.scales,    engine->get_host_scales(),           ggml_nbytes(engine->get_scales()));
            ensure(pb.anc_pos,   engine->get_host_anchor_positions(), ggml_nbytes(engine->get_anchor_positions()));

            pb.pool_ver = cur_pool_ver;

            u_pool_buf    = pb.u_pool;
            u_scale_buf   = pb.u_scale;
            vk_pool_buf   = pb.vk_pool;
            vv_pool_buf   = pb.vv_pool;
            anchors_k_buf = pb.anchors_k;
            anchors_v_buf = pb.anchors_v;
            seq_lens_buf  = pb.seq_lens;
            scales_buf    = pb.scales;
            anc_pos_buf   = pb.anc_pos;
        }

        // ── Encode and dispatch ───────────────────────────────────────────────
        id<MTLCommandBuffer>        commandBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder      = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_pipeline];

        // Buffer bindings 0-21 (sparse path)
        [encoder setBuffer:q_buf          offset:0 atIndex:0];
        [encoder setBuffer:u_pool_buf     offset:0 atIndex:1];
        [encoder setBuffer:u_scale_buf    offset:0 atIndex:2];
        [encoder setBuffer:vk_pool_buf    offset:0 atIndex:3];
        [encoder setBuffer:vv_pool_buf    offset:0 atIndex:4];
        [encoder setBuffer:anchors_k_buf  offset:0 atIndex:5];
        [encoder setBuffer:anchors_v_buf  offset:0 atIndex:6];
        [encoder setBuffer:seq_lens_buf   offset:0 atIndex:7];
        [encoder setBuffer:slot_indices_buf offset:0 atIndex:8];
        [encoder setBuffer:out_buf        offset:0 atIndex:9];
        [encoder setBuffer:lse_buf        offset:0 atIndex:10];

        int32_t n_q_heads_i32  = n_q_heads;
        int32_t n_kv_heads_i32 = n_kv_heads;
        int32_t rank_i32       = rank;
        int32_t S_max_i32      = S_max;
        int32_t K_i32          = active_K;
        int32_t D_i32          = D;
        float   scale_f32      = scale;
        int32_t has_rope_i32   = has_rope ? 1 : 0;
        float   rope_f32       = rope_freq;
        int32_t T_dense_i32    = T_clamped;

        [encoder setBytes:&n_q_heads_i32  length:sizeof(n_q_heads_i32)  atIndex:11];
        [encoder setBytes:&n_kv_heads_i32 length:sizeof(n_kv_heads_i32) atIndex:12];
        [encoder setBytes:&rank_i32       length:sizeof(rank_i32)       atIndex:13];
        [encoder setBytes:&S_max_i32      length:sizeof(S_max_i32)      atIndex:14];
        [encoder setBytes:&K_i32          length:sizeof(K_i32)          atIndex:15];
        [encoder setBytes:&D_i32          length:sizeof(D_i32)          atIndex:16];
        [encoder setBytes:&scale_f32      length:sizeof(scale_f32)      atIndex:17];
        [encoder setBuffer:scales_buf     offset:0                      atIndex:18];
        [encoder setBytes:&has_rope_i32   length:sizeof(has_rope_i32)   atIndex:19];
        [encoder setBytes:&rope_f32       length:sizeof(rope_f32)       atIndex:20];
        [encoder setBuffer:anc_pos_buf    offset:0                      atIndex:21];

        // Buffer bindings 22-25 (dense window)
        [encoder setBuffer:dense_k_buf    offset:0                      atIndex:22];
        [encoder setBuffer:dense_v_buf    offset:0                      atIndex:23];
        [encoder setBuffer:dense_pos_buf  offset:0                      atIndex:24];
        [encoder setBytes:&T_dense_i32    length:sizeof(T_dense_i32)    atIndex:25];

        // Buffer binding 26 (approximate_attn config)
        int32_t approx_attn_i32 = data->approximate_attn ? 1 : 0;
        [encoder setBytes:&approx_attn_i32 length:sizeof(approx_attn_i32) atIndex:26];

        MTLSize threadsPerTG    = MTLSizeMake(64, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake(n_q_heads, 1, 1);
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerTG];
        [encoder endEncoding];
        auto t_k0 = std::chrono::high_resolution_clock::now();
        [commandBuffer commit];

        // ── Spin-wait: correct + fast for our tiny (~0.05ms) sparse kernel ────
        // waitUntilCompleted incurs ~2-3ms sleep/wake overhead per call on macOS.
        // 28 layers × 2-3ms = ~56-84ms wasted per token — the dominant TPS cost.
        // Spin-polling costs only the true kernel time (~0.05ms × 28 = ~1.4ms).
        // We MUST wait here: ggml reads dst->data immediately after this returns.
        while ([commandBuffer status] < MTLCommandBufferStatusCompleted) {
            // tiny spin — kernel finishes in microseconds
        }
        auto t_spin_end = std::chrono::high_resolution_clock::now();
        g_accumulated_wait_ms += std::chrono::duration<double, std::milli>(t_spin_end - t_k0).count();

        // Copy temp output back if dst wasn't page-aligned (rare path)
        if (temp_out_ptr != nullptr) {
            memcpy(dst->data, temp_out_ptr, ggml_nbytes(dst));
        }
        if (lse_out) {
            memcpy(lse_out, lse_buf.contents, n_q_heads * sizeof(float));
        }
    }
}


void cleanup_metal_attention() {

    g_dummy_rope_buf = nil;
    g_pipeline = nil;
    g_queue = nil;
    g_device = nil;
}

double get_and_reset_accumulated_wait_ms() {
    double val = g_accumulated_wait_ms;
    g_accumulated_wait_ms = 0.0;
    return val;
}

} // namespace diffkv
