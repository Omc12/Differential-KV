#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "runtime/diffkv_attention.hpp"
#include <iostream>
#include <cmath>
#include <mach-o/dyld.h>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static id<MTLBuffer> g_dummy_rope_buf = nil;

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
    : kv_engine(nullptr), slot_indices(nullptr), n_q_heads(0), n_kv_heads(0),
      rank(0), S_max(0), K(0), D(0), scale(0.0f), has_rope(false), rope_freq_base(0.0f),
      active_k_dense(nullptr), active_v_dense(nullptr), active_positions_dense(nullptr),
      active_block_tokens(0), active_slot(0), ignore_c(false), current_pos(0),
      mtl_dense_k(nullptr), mtl_dense_v(nullptr), mtl_dense_pos(nullptr),
      mtl_slot_indices(nullptr), mtl_output_buf(nullptr), mtl_lse_buf(nullptr),
      mtl_q_buf(nullptr), mtl_u_pool(nullptr), mtl_u_scale(nullptr),
      mtl_vk_pool(nullptr), mtl_vv_pool(nullptr), mtl_anchors_k(nullptr),
      mtl_anchors_v(nullptr), mtl_seq_lens(nullptr), mtl_scales(nullptr),
      mtl_anc_pos(nullptr) {}

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
    active_k_dense = other.active_k_dense;
    active_v_dense = other.active_v_dense;
    active_positions_dense = other.active_positions_dense;
    active_block_tokens = other.active_block_tokens;
    active_slot = other.active_slot;
    ignore_c = other.ignore_c;
    current_pos = other.current_pos;

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
        active_k_dense = other.active_k_dense;
        active_v_dense = other.active_v_dense;
        active_positions_dense = other.active_positions_dense;
        active_block_tokens = other.active_block_tokens;
        active_slot = other.active_slot;
        ignore_c = other.ignore_c;
        current_pos = other.current_pos;

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
    const int raw_K   = data->K;
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
        const int   T_clamped   = std::min(T_dense, 512);  // kernel shared-mem limit
        
        size_t max_k_bytes   = 16384 * F_kv * sizeof(float);
        size_t max_v_bytes   = 16384 * F_kv * sizeof(float);
        size_t max_pos_bytes = 16384 * sizeof(int32_t);

        id<MTLBuffer> dense_k_buf   = nil;
        id<MTLBuffer> dense_v_buf   = nil;
        id<MTLBuffer> dense_pos_buf = nil;

        if (data->mtl_dense_k) {
            dense_k_buf = (__bridge id<MTLBuffer>)data->mtl_dense_k;
        } else if (dense_K && T_clamped > 0) {
            dense_k_buf = wrap_cpu_ptr(dense_K, max_k_bytes);
            data->mtl_dense_k = (__bridge_retained void*)dense_k_buf;
        } else {
            dense_k_buf = g_dummy_rope_buf;
        }

        if (data->mtl_dense_v) {
            dense_v_buf = (__bridge id<MTLBuffer>)data->mtl_dense_v;
        } else if (dense_V && T_clamped > 0) {
            dense_v_buf = wrap_cpu_ptr(dense_V, max_v_bytes);
            data->mtl_dense_v = (__bridge_retained void*)dense_v_buf;
        } else {
            dense_v_buf = g_dummy_rope_buf;
        }

        if (data->mtl_dense_pos) {
            dense_pos_buf = (__bridge id<MTLBuffer>)data->mtl_dense_pos;
        } else if (dense_pos && T_clamped > 0) {
            dense_pos_buf = wrap_cpu_ptr(dense_pos, max_pos_bytes);
            data->mtl_dense_pos = (__bridge_retained void*)dense_pos_buf;
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

        // ── Sparse pool and query buffers (cached) ───────────────────────────
        id<MTLBuffer> q_buf = nil;
        if (data->mtl_q_buf) {
            q_buf = (__bridge id<MTLBuffer>)data->mtl_q_buf;
        } else {
            q_buf = wrap_tensor((struct ggml_tensor*)Q);
            data->mtl_q_buf = (__bridge_retained void*)q_buf;
        }

        id<MTLBuffer> u_pool_buf = nil;
        if (data->mtl_u_pool) {
            u_pool_buf = (__bridge id<MTLBuffer>)data->mtl_u_pool;
        } else {
            u_pool_buf = wrap_tensor(engine->get_U());
            data->mtl_u_pool = (__bridge_retained void*)u_pool_buf;
        }

        id<MTLBuffer> u_scale_buf = nil;
        if (data->mtl_u_scale) {
            u_scale_buf = (__bridge id<MTLBuffer>)data->mtl_u_scale;
        } else {
            u_scale_buf = wrap_tensor(engine->get_U_scale());
            data->mtl_u_scale = (__bridge_retained void*)u_scale_buf;
        }

        id<MTLBuffer> vk_pool_buf = nil;
        if (data->mtl_vk_pool) {
            vk_pool_buf = (__bridge id<MTLBuffer>)data->mtl_vk_pool;
        } else {
            vk_pool_buf = wrap_tensor(engine->get_VK());
            data->mtl_vk_pool = (__bridge_retained void*)vk_pool_buf;
        }

        id<MTLBuffer> vv_pool_buf = nil;
        if (data->mtl_vv_pool) {
            vv_pool_buf = (__bridge id<MTLBuffer>)data->mtl_vv_pool;
        } else {
            vv_pool_buf = wrap_tensor(engine->get_VV());
            data->mtl_vv_pool = (__bridge_retained void*)vv_pool_buf;
        }

        id<MTLBuffer> anchors_k_buf = nil;
        if (data->mtl_anchors_k) {
            anchors_k_buf = (__bridge id<MTLBuffer>)data->mtl_anchors_k;
        } else {
            anchors_k_buf = wrap_tensor(engine->get_anchors_K());
            data->mtl_anchors_k = (__bridge_retained void*)anchors_k_buf;
        }

        id<MTLBuffer> anchors_v_buf = nil;
        if (data->mtl_anchors_v) {
            anchors_v_buf = (__bridge id<MTLBuffer>)data->mtl_anchors_v;
        } else {
            anchors_v_buf = wrap_tensor(engine->get_anchors_V());
            data->mtl_anchors_v = (__bridge_retained void*)anchors_v_buf;
        }

        id<MTLBuffer> seq_lens_buf = nil;
        if (data->mtl_seq_lens) {
            seq_lens_buf = (__bridge id<MTLBuffer>)data->mtl_seq_lens;
        } else {
            seq_lens_buf = wrap_tensor(engine->get_seq_lens());
            data->mtl_seq_lens = (__bridge_retained void*)seq_lens_buf;
        }

        id<MTLBuffer> scales_buf = nil;
        if (data->mtl_scales) {
            scales_buf = (__bridge id<MTLBuffer>)data->mtl_scales;
        } else {
            scales_buf = wrap_tensor(engine->get_scales());
            data->mtl_scales = (__bridge_retained void*)scales_buf;
        }

        id<MTLBuffer> anc_pos_buf = nil;
        if (data->mtl_anc_pos) {
            anc_pos_buf = (__bridge id<MTLBuffer>)data->mtl_anc_pos;
        } else {
            anc_pos_buf = wrap_tensor(engine->get_anchor_positions());
            data->mtl_anc_pos = (__bridge_retained void*)anc_pos_buf;
        }

        // ── Encode and dispatch ───────────────────────────────────────────────
        id<MTLCommandBuffer>     commandBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder   = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_pipeline];

        // Buffer bindings 0-21 (sparse path, unchanged layout)
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

        // Buffer bindings 22-25 (dense window, new)
        [encoder setBuffer:dense_k_buf    offset:0                      atIndex:22];
        [encoder setBuffer:dense_v_buf    offset:0                      atIndex:23];
        [encoder setBuffer:dense_pos_buf  offset:0                      atIndex:24];
        [encoder setBytes:&T_dense_i32    length:sizeof(T_dense_i32)    atIndex:25];

        // 1 threadgroup per query head, 64 threads per threadgroup
        MTLSize threadsPerTG  = MTLSizeMake(64, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake(n_q_heads, 1, 1);
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerTG];

        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy temp output back if needed
        if (temp_out_ptr != nullptr) {
            memcpy(dst->data, temp_out_ptr, ggml_nbytes(dst));
        }

        // Copy combined LSE out for callers that inspect it (e.g. debug)
        if (lse_out) {
            memcpy(lse_out, lse_buf.contents, n_q_heads * sizeof(float));
        }
    }
}

} // namespace diffkv
