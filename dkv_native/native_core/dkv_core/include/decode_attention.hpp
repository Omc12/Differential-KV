#pragma once
#include <cstdint>

namespace dkv {

void decode_attention(
    const float*    Q,
    const int8_t*   U_pool,
    const float*    U_scale_pool,
    const uint16_t* VK_pool,
    const uint16_t* VV_pool,
    const uint16_t* anchors_K,
    const uint16_t* anchors_V,
    const int32_t*  seq_lens,
    const uint16_t* scales,
    const float*    cos_anc,
    const float*    sin_anc,
    const int32_t*  slot_indices,
    int K_active,
    int N_pool, int S_max, int R,
    int H_q, int kv_heads, int D,
    float scale,
    float* out
);

} // namespace dkv
