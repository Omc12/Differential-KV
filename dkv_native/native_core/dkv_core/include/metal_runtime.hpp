#pragma once
#include <cstdint>

namespace dkv {

bool is_metal_available();

void decode_attention_metal(
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
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank,
    int K_active,
    int N_pool,
    int S_max,
    int D,
    float* out,
    float* lse
);

} // namespace dkv
