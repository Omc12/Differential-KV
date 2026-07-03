#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstring>
#include "runtime/native_block_pool.hpp"
#include "ggml-cpu.h"

// Declare execute_cpu_attention so we can invoke it
namespace diffkv {
void execute_cpu_attention(
    const float* Q,
    const int32_t* slots,
    float* cpu_output,
    float* lse_sparse,
    diffkv::NativeBlockPool* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale,
    bool has_rope, float rope_freq_base, bool approximate_attn
);
}

struct Array {
    std::string name;
    std::string dtype;
    std::vector<int32_t> shape;
    std::vector<char> data;
};

std::map<std::string, Array> load_bin(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open file " + filename);
    }
    
    uint32_t num_items = 0;
    f.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    
    std::map<std::string, Array> res;
    for (uint32_t i = 0; i < num_items; ++i) {
        Array arr;
        
        uint32_t name_len = 0;
        f.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        arr.name.resize(name_len);
        f.read(&arr.name[0], name_len);
        
        uint32_t dtype_len = 0;
        f.read(reinterpret_cast<char*>(&dtype_len), sizeof(dtype_len));
        arr.dtype.resize(dtype_len);
        f.read(&arr.dtype[0], dtype_len);
        
        uint32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        arr.shape.resize(ndim);
        f.read(reinterpret_cast<char*>(arr.shape.data()), ndim * sizeof(int32_t));
        
        uint64_t data_len = 0;
        f.read(reinterpret_cast<char*>(&data_len), sizeof(data_len));
        arr.data.resize(data_len);
        f.read(arr.data.data(), data_len);
        
        res[arr.name] = arr;
    }
    return res;
}

int main() {
    try {
        std::cout << "Loading golden conformance vectors..." << std::endl;
        auto arrays = load_bin("tools/conformance_vectors.bin");
        
        // Params
        int H_q = 8;
        int H_kv = 2;
        int D = 64;
        int rank = 16;
        int S_max = 64;
        int n_slots = 4;
        int K_active = 3;
        float scale = 0.125f;
        bool has_rope = false;
        float rope_freq_base = 1000000.0f;
        bool approximate_attn = true;
        int desc_dim = 64;
        
        // Get buffers from load
        const float* Q = reinterpret_cast<const float*>(arrays["Q"].data.data());
        const int32_t* slots = reinterpret_cast<const int32_t*>(arrays["slots"].data.data());
        
        const int8_t* host_U = reinterpret_cast<const int8_t*>(arrays["host_U"].data.data());
        const uint16_t* host_U_scale = reinterpret_cast<const uint16_t*>(arrays["host_U_scale"].data.data());
        const uint16_t* host_U_row_scale = reinterpret_cast<const uint16_t*>(arrays["host_U_row_scale"].data.data());
        
        const uint16_t* host_VK = reinterpret_cast<const uint16_t*>(arrays["host_VK"].data.data());
        const uint16_t* host_VV = reinterpret_cast<const uint16_t*>(arrays["host_VV"].data.data());
        
        const uint16_t* host_anchors_K = reinterpret_cast<const uint16_t*>(arrays["host_anchors_K"].data.data());
        const uint16_t* host_anchors_V = reinterpret_cast<const uint16_t*>(arrays["host_anchors_V"].data.data());
        
        const uint16_t* host_scales = reinterpret_cast<const uint16_t*>(arrays["host_scales"].data.data());
        const uint16_t* host_valid_mask = reinterpret_cast<const uint16_t*>(arrays["host_valid_mask"].data.data());
        
        const int32_t* host_seq_lens = reinterpret_cast<const int32_t*>(arrays["host_seq_lens"].data.data());
        const int32_t* host_anchor_positions = reinterpret_cast<const int32_t*>(arrays["host_anchor_positions"].data.data());
        
        const int32_t* host_res_K_pos = reinterpret_cast<const int32_t*>(arrays["host_res_K_pos"].data.data());
        const int32_t* host_res_V_pos = reinterpret_cast<const int32_t*>(arrays["host_res_V_pos"].data.data());
        const uint16_t* host_res_K_val = reinterpret_cast<const uint16_t*>(arrays["host_res_K_val"].data.data());
        const uint16_t* host_res_V_val = reinterpret_cast<const uint16_t*>(arrays["host_res_V_val"].data.data());
        
        const float* expected_out = reinterpret_cast<const float*>(arrays["expected_out"].data.data());
        const float* expected_lse = reinterpret_cast<const float*>(arrays["expected_lse"].data.data());
        
        // Initialize pool
        diffkv::NativeBlockPool pool;
        ggml_backend_t backend_cpu = ggml_backend_cpu_init();
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_cpu);
        
        // Initialize with kv_type = GGML_TYPE_F16 (which matches float16 arrays)
        if (!pool.initialize(n_slots, rank, D, H_kv, desc_dim, buft, S_max, GGML_TYPE_F16)) {
            std::cerr << "Failed to initialize NativeBlockPool" << std::endl;
            return 1;
        }
        
        // Set states of slots 0, 1, 2 to CompressedResident so execute_cpu_attention processes them
        for (int s = 0; s < 3; ++s) {
            pool.get_state_table().transition(s, diffkv::BlockState::Freed, diffkv::BlockState::DenseResident);
            pool.get_state_table().transition(s, diffkv::BlockState::DenseResident, diffkv::BlockState::Compressing);
            pool.get_state_table().transition(s, diffkv::BlockState::Compressing, diffkv::BlockState::CompressedResident);
        }
        
        // Populate the pool tensors from loaded arrays
        std::memcpy(pool.get_host_U_scale(), host_U_scale, n_slots * sizeof(uint16_t));
        std::memcpy(pool.get_host_scales(), host_scales, n_slots * sizeof(uint16_t));
        std::memcpy(pool.get_host_seq_lens(), host_seq_lens, n_slots * sizeof(int32_t));
        std::memcpy(pool.get_host_anchor_positions(), host_anchor_positions, n_slots * sizeof(int32_t));
        
        for (int s = 0; s < n_slots; ++s) {
            std::memcpy(pool.get_host_U(s), host_U + s * S_max * rank, S_max * rank * sizeof(int8_t));
            std::memcpy(pool.get_host_U_row_scale(s), host_U_row_scale + s * S_max, S_max * sizeof(uint16_t));
            std::memcpy(pool.get_host_VK(s), host_VK + s * rank * H_kv * D, rank * H_kv * D * sizeof(uint16_t));
            std::memcpy(pool.get_host_VV(s), host_VV + s * rank * H_kv * D, rank * H_kv * D * sizeof(uint16_t));
            std::memcpy(pool.get_host_anchors_K(s), host_anchors_K + s * H_kv * D, H_kv * D * sizeof(uint16_t));
            std::memcpy(pool.get_host_anchors_V(s), host_anchors_V + s * H_kv * D, H_kv * D * sizeof(uint16_t));
            
            std::memcpy(pool.get_host_res_K_pos(s), host_res_K_pos + s * 64, 64 * sizeof(int32_t));
            std::memcpy(pool.get_host_res_V_pos(s), host_res_V_pos + s * 64, 64 * sizeof(int32_t));
            std::memcpy(pool.get_host_res_K_val(s), host_res_K_val + s * 64 * H_kv * D, 64 * H_kv * D * sizeof(uint16_t));
            std::memcpy(pool.get_host_res_V_val(s), host_res_V_val + s * 64 * H_kv * D, 64 * H_kv * D * sizeof(uint16_t));
        }
        
        // Output buffers
        std::vector<float> cpu_output(H_q * D, 0.0f);
        std::vector<float> lse_sparse(H_q, 0.0f);
        
        std::cout << "Executing C++ CPU sparse attention..." << std::endl;
        diffkv::execute_cpu_attention(Q, slots, cpu_output.data(), lse_sparse.data(), &pool,
                              H_q, H_kv, rank, S_max, K_active, D, scale,
                              has_rope, rope_freq_base, approximate_attn);
                              
        // Verify results
        std::cout << "Verifying attention outputs..." << std::endl;
        double max_diff_out = 0.0;
        for (int i = 0; i < H_q * D; ++i) {
            double diff = std::abs((double)cpu_output[i] - (double)expected_out[i]);
            if (diff > max_diff_out) max_diff_out = diff;
        }
        
        double max_diff_lse = 0.0;
        for (int i = 0; i < H_q; ++i) {
            double diff = std::abs((double)lse_sparse[i] - (double)expected_lse[i]);
            if (diff > max_diff_lse) max_diff_lse = diff;
        }
        
        std::cout << "Max output discrepancy: " << max_diff_out << std::endl;
        std::cout << "Max LSE discrepancy:    " << max_diff_lse << std::endl;
        
        // Parity margin check: we expect discrepancies to be extremely tiny (e.g., < 1e-4)
        if (max_diff_out < 1e-4 && max_diff_lse < 1e-4) {
            std::cout << "========================================" << std::endl;
            std::cout << "CONFORMANCE TEST: PASS ✓" << std::endl;
            std::cout << "========================================" << std::endl;
            ggml_backend_free(backend_cpu);
            return 0;
        } else {
            std::cout << "========================================" << std::endl;
            std::cout << "CONFORMANCE TEST: FAIL ✗ (Discrepancy exceeds margin)" << std::endl;
            std::cout << "========================================" << std::endl;
            ggml_backend_free(backend_cpu);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during conformance test execution: " << e.what() << std::endl;
        return 1;
    }
}
