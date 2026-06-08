// diffkv_core/include/decode_attention.hpp
// C++ declaration for the fused Project-Then-Attend decode attention kernel.
//
// This replaces the Python/MPS decode path in runtime/diffkv_attention.py:
//   - fused_decode_mps()                      [MPS path, Phase 34]
//   - native_triton_sparse_attn_decode()       [CUDA path, delegated to Triton]
//
// The implementation (decode_attention.cpp) uses the ATen C++ API and supports
// both MPS (Apple Silicon unified memory) and CUDA devices.
//
// Thread safety:
//   MPS: Callers must not issue concurrent MPS ops from multiple threads.
//        The C++ function itself is stateless. Thread serialization is the
//        responsibility of the Python callsite (same constraint as Triton kernels).
//   CUDA: Each call uses the calling thread's current CUDA stream.
//
// Tensor layout contract (must match NativeBlockPool in native_block_pool.py):
//   U_pool       : [N_pool, S_max, R]      int8    pool.U
//   U_scale_pool : [N_pool]                float16 pool.U_scale  (per-block scale)
//   VK_pool      : [N_pool, R, kv_heads, D] float16 pool.V_K
//   VV_pool      : [N_pool, R, kv_heads, D] float16 pool.V_V
//   anchors_K    : [N_pool, kv_heads, D]   float16 pool.anchors_K
//   anchors_V    : [N_pool, kv_heads, D]   float16 pool.anchors_V
//   seq_lens     : [N_pool]                int32   pool.seq_lens (actual tokens in block)
//   slot_indices : [K_active]              int32   selected pool slots (from SRL routing)

#pragma once
#include <torch/extension.h>

namespace diffkv {

// ── decode_attention_aten ─────────────────────────────────────────────────────
// Fused Project-Then-Attend decode attention in C++ (ATen API).
//
// Algorithm (per block b in slot_indices):
//   1. Dequantize U_pool[b]:    U_b = U_pool[b].float() * U_scale_pool[b]   [S_b, R]
//   2. Dequantize VK_pool[b]:   VK_b = VK_pool[b].float()                   [R, kv_heads, D]
//   3. Anchor score:            s_anc = (q_kv_mean @ anc_K_b.T) * scale      [kv_heads]
//   4. Q projection:            q_proj = q_kv_mean @ VK_b.reshape(R, kv_heads*D).T  [kv_heads, R]
//                               (or GQA-correct: per-group projection)
//   5. Delta scores:            s_delta = (q_proj.view(kv_heads,1,R) @ U_b.T).squeeze() [kv_heads, S_b]
//   6. Online softmax over (s_anc, s_delta[0..S_b-1]) → accumulate V
//
// Online softmax is maintained across all blocks so the final output is
// numerically equivalent to materializing all KV pairs and running SDPA.
//
// Args:
//   Q            : [H_q, D]                  query (float16 on MPS/CUDA)
//   U_pool       : [N_pool, S_max, R]         int8
//   U_scale_pool : [N_pool]                   float16
//   VK_pool      : [N_pool, R, kv_heads, D]   float16
//   VV_pool      : [N_pool, R, kv_heads, D]   float16
//   anchors_K    : [N_pool, kv_heads, D]      float16
//   anchors_V    : [N_pool, kv_heads, D]      float16
//   seq_lens     : [N_pool]                   int32
//   slot_indices : [K_active]                 int32
//   scale        : float (1.0 / sqrt(head_dim))
//   n_q_heads    : int (num_attention_heads)
//   n_kv_heads   : int (num_key_value_heads)
//   rank         : int (SVD rank R)
//
// Returns:
//   [H_q, D] float16 tensor — attention output ready for HuggingFace o_proj
//
// Caller is responsible for:
//   - Contiguous layout of all input tensors (call .contiguous() if needed)
//   - Same device for all tensors
//   - No in-place mutation of pool tensors during this call (MPS unified memory)
torch::Tensor decode_attention_aten(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
);

// ── decode_attention_aten_lse ─────────────────────────────────────────────────
// Variant that also returns the log-sum-exp values per head.
// Used when combining sparse-history output with dense-window SDPA output.
//
// Returns:
//   Tuple of:
//     [H_q, D]  float16 — attention output
//     [H_q]     float32 — log-sum-exp per query head (for LSE combine)
std::tuple<torch::Tensor, torch::Tensor> decode_attention_aten_lse(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
);

} // namespace diffkv
