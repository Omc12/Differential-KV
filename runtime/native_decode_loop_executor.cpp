#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

struct DecodeSlot {
    int slot_id;
    int session_id_hash;
    int current_step;
    int max_tokens;
    bool active;
};

// Global residency array for native batch execution
#define MAX_NATIVE_SLOTS 64
static DecodeSlot native_residency_table[MAX_NATIVE_SLOTS] = {0};
static int active_native_slot_count = 0;

extern "C" {
    // Native residency initialization
    __declspec(dllexport) void init_native_residency_table() {
        for (int i = 0; i < MAX_NATIVE_SLOTS; ++i) {
            native_residency_table[i].slot_id = i;
            native_residency_table[i].session_id_hash = 0;
            native_residency_table[i].current_step = 0;
            native_residency_table[i].max_tokens = 0;
            native_residency_table[i].active = false;
        }
        active_native_slot_count = 0;
    }

    // Native slot allocation
    __declspec(dllexport) int allocate_native_slot(int session_hash, int max_tokens) {
        for (int i = 0; i < MAX_NATIVE_SLOTS; ++i) {
            if (!native_residency_table[i].active) {
                native_residency_table[i].session_id_hash = session_hash;
                native_residency_table[i].current_step = 0;
                native_residency_table[i].max_tokens = max_tokens;
                native_residency_table[i].active = true;
                active_native_slot_count++;
                return i;
            }
        }
        return -1; // No slot available
    }

    // Native slot release
    __declspec(dllexport) void release_native_slot(int slot_id) {
        if (slot_id >= 0 && slot_id < MAX_NATIVE_SLOTS) {
            if (native_residency_table[slot_id].active) {
                native_residency_table[slot_id].active = false;
                native_residency_table[slot_id].session_id_hash = 0;
                native_residency_table[slot_id].current_step = 0;
                native_residency_table[slot_id].max_tokens = 0;
                active_native_slot_count--;
            }
        }
    }

    // Native loop advancement step
    __declspec(dllexport) void execute_native_decode_step(
        int step,
        int active_slots,
        float* latency_ms_out,
        int* launches_out
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulates the physical orchestration hotpath in pure C++
        long long sum = 0;
        int operations = 25000;
        for (int i = 0; i < operations; ++i) {
            sum += i * step + active_slots;
        }
        
        // Update residency state for all active slots
        for (int i = 0; i < MAX_NATIVE_SLOTS; ++i) {
            if (native_residency_table[i].active) {
                native_residency_table[i].current_step++;
                if (native_residency_table[i].current_step >= native_residency_table[i].max_tokens) {
                    // Auto-expire session natively
                    native_residency_table[i].active = false;
                    native_residency_table[i].session_id_hash = 0;
                    native_residency_table[i].current_step = 0;
                    native_residency_table[i].max_tokens = 0;
                    active_native_slot_count = (active_native_slot_count > 0) ? active_native_slot_count - 1 : 0;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        
        *latency_ms_out = duration.count();
        *launches_out = 1; // Collasped into a single native launch invocation
    }

    // Native status check
    __declspec(dllexport) int get_active_native_slot_count() {
        return active_native_slot_count;
    }
}
