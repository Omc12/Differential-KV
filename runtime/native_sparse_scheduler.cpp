#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>

namespace diffkv {
namespace runtime {

class NativeSparseScheduler {
public:
    NativeSparseScheduler(int num_workers) : stop_(false) {
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&NativeSparseScheduler::worker_loop, this);
        }
    }

    ~NativeSparseScheduler() {
        stop_ = true;
        condition_.notify_all();
        for (auto& thread : workers_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void schedule_retrieval(int query_id, const std::vector<float>& query_vector) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tasks_.push({query_id, query_vector});
        condition_.notify_one();
    }

private:
    struct Task {
        int query_id;
        std::vector<float> query_vector;
    };

    void worker_loop() {
        while (!stop_) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = tasks_.front();
                tasks_.pop();
            }
            // Execute retrieval natively, bypassing Python GIL
            execute_sparse_retrieval(task);
        }
    }

    void execute_sparse_retrieval(const Task& task) {
        // Mock native retrieval logic
        // std::cout << "Native scheduler executing retrieval for query: " << task.query_id << "\n";
    }

    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
};

} // namespace runtime
} // namespace diffkv
