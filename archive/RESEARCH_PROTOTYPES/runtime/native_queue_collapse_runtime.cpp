#include <mutex>
#include <vector>
#include <string>

struct QueueItem {
    int request_id;
    int priority;
    bool active;
};

#define MAX_QUEUE_SIZE 128
static QueueItem native_queue[MAX_QUEUE_SIZE] = {0};
static int queue_head = 0;
static int queue_tail = 0;
static int total_queued = 0;
static std::mutex queue_mutex;

extern "C" {
    __declspec(dllexport) void init_native_queue() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (int i = 0; i < MAX_QUEUE_SIZE; ++i) {
            native_queue[i].request_id = 0;
            native_queue[i].priority = 0;
            native_queue[i].active = false;
        }
        queue_head = 0;
        queue_tail = 0;
        total_queued = 0;
    }

    __declspec(dllexport) int enqueue_native_request(int request_id, int priority) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (total_queued >= MAX_QUEUE_SIZE) {
            return 0; // Overflow
        }
        native_queue[queue_tail].request_id = request_id;
        native_queue[queue_tail].priority = priority;
        native_queue[queue_tail].active = true;
        
        queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
        total_queued++;
        return 1;
    }

    __declspec(dllexport) int dequeue_native_request() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (total_queued == 0) {
            return -1; // Empty
        }
        int req_id = native_queue[queue_head].request_id;
        native_queue[queue_head].active = false;
        
        queue_head = (queue_head + 1) % MAX_QUEUE_SIZE;
        total_queued--;
        return req_id;
    }

    __declspec(dllexport) int get_native_queue_depth() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return total_queued;
    }

    __declspec(dllexport) int arbitrate_next_slot() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (total_queued == 0) {
            return -1;
        }
        // Highest priority arbitration
        int best_idx = -1;
        int max_pri = -9999;
        
        int curr = queue_head;
        for (int i = 0; i < total_queued; ++i) {
            if (native_queue[curr].active && native_queue[curr].priority > max_pri) {
                max_pri = native_queue[curr].priority;
                best_idx = curr;
            }
            curr = (curr + 1) % MAX_QUEUE_SIZE;
        }
        
        if (best_idx != -1) {
            int req_id = native_queue[best_idx].request_id;
            // Shift queue items to preserve ordering or simple pop
            return req_id;
        }
        return -1;
    }
}
