#include <chrono>
#include <vector>
#include <mutex>

struct ResidencySlot {
    int id;
    int session_id_hash;
    bool occupied;
    float admission_time_ms;
};

#define MAX_SCHEDULER_SLOTS 64
static ResidencySlot residency_slots[MAX_SCHEDULER_SLOTS] = {0};
static int scheduled_count = 0;
static std::mutex scheduler_mutex;

extern "C" {
    __declspec(dllexport) void init_native_scheduler() {
        std::lock_guard<std::mutex> lock(scheduler_mutex);
        for (int i = 0; i < MAX_SCHEDULER_SLOTS; ++i) {
            residency_slots[i].id = i;
            residency_slots[i].session_id_hash = 0;
            residency_slots[i].occupied = false;
            residency_slots[i].admission_time_ms = 0.0f;
        }
        scheduled_count = 0;
    }

    __declspec(dllexport) int schedule_native_session(int session_hash) {
        std::lock_guard<std::mutex> lock(scheduler_mutex);
        // Find existing slot or admit new
        for (int i = 0; i < MAX_SCHEDULER_SLOTS; ++i) {
            if (residency_slots[i].occupied && residency_slots[i].session_id_hash == session_hash) {
                return i; // Already scheduled
            }
        }
        for (int i = 0; i < MAX_SCHEDULER_SLOTS; ++i) {
            if (!residency_slots[i].occupied) {
                residency_slots[i].session_id_hash = session_hash;
                residency_slots[i].occupied = true;
                residency_slots[i].admission_time_ms = 120.5f; // Simulated timing
                scheduled_count++;
                return i;
            }
        }
        return -1; // Overloaded
    }

    __declspec(dllexport) void evict_native_session(int slot_id) {
        std::lock_guard<std::mutex> lock(scheduler_mutex);
        if (slot_id >= 0 && slot_id < MAX_SCHEDULER_SLOTS) {
            if (residency_slots[slot_id].occupied) {
                residency_slots[slot_id].occupied = false;
                residency_slots[slot_id].session_id_hash = 0;
                residency_slots[slot_id].admission_time_ms = 0.0f;
                scheduled_count = (scheduled_count > 0) ? scheduled_count - 1 : 0;
            }
        }
    }

    __declspec(dllexport) float get_native_occupancy_rate() {
        std::lock_guard<std::mutex> lock(scheduler_mutex);
        int active = 0;
        for (int i = 0; i < MAX_SCHEDULER_SLOTS; ++i) {
            if (residency_slots[i].occupied) {
                active++;
            }
        }
        return (float)active / (float)MAX_SCHEDULER_SLOTS;
    }
}
