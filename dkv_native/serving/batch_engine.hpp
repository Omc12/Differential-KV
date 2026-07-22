#pragma once

#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "runtime/dkv_model.hpp"
#include "native_core/kv_runtime_manager.hpp"
#include "serving/production_session_manager.hpp"

namespace dkv {

struct BatchRequest {
    std::string session_id;
    std::string prompt;
    int max_tokens = 16384;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float repetition_penalty = 1.15f;

    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    bool is_prefilled = false;
    bool is_finished = false;
    bool cancelled = false;
    bool repetition_loop_detected = false;
    int loop_detection_idx = -1;
    bool sfa_active = false;

    // Output queue / callback state
    std::vector<std::string> output_chunks;
    std::mutex output_mutex;
    std::condition_variable output_cv;
    bool stream_finished = false;
    std::string error_msg = "";

    BatchRequest(
        const std::string& sid,
        const std::string& p,
        int mt,
        float temp,
        float tp,
        float rep
    ) : session_id(sid),
        prompt(p),
        max_tokens(mt),
        temperature(temp),
        top_p(tp),
        repetition_penalty(rep) {}

    void push_chunk(const std::string& chunk) {
        std::lock_guard<std::mutex> lock(output_mutex);
        output_chunks.push_back(chunk);
        output_cv.notify_one();
    }

    void finish_stream() {
        std::lock_guard<std::mutex> lock(output_mutex);
        stream_finished = true;
        output_cv.notify_one();
    }

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(output_mutex);
        error_msg = err;
        stream_finished = true;
        output_cv.notify_one();
    }
};

class DKVBatchEngine {
public:
    DKVBatchEngine(
        DKVModel* model,
        ggml_backend_t backend,
        ggml_backend_sched_t sched,
        KVRuntimeManager* runtime_manager,
        ProductionSessionManager* session_manager
    );
    ~DKVBatchEngine();

    void start();
    void stop();

    std::shared_ptr<BatchRequest> submit(
        const std::string& session_id,
        const std::string& prompt,
        int max_tokens = 16384,
        float temperature = 0.7f,
        float top_p = 0.9f,
        float repetition_penalty = 1.15f
    );

    void cancel(const std::string& session_id);

    DKVModel* get_model() const { return model_; }
    KVRuntimeManager* get_runtime_manager() const { return runtime_manager_; }
    ProductionSessionManager* get_session_manager() const { return session_manager_; }

private:
    void run_loop();
    void process_request(const std::shared_ptr<BatchRequest>& req);

    DKVModel* model_;
    ggml_backend_t backend_;
    ggml_backend_sched_t sched_;
    KVRuntimeManager* runtime_manager_;
    ProductionSessionManager* session_manager_;

    std::queue<std::shared_ptr<BatchRequest>> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::unordered_set<int32_t> stop_token_ids_;
};

} // namespace dkv
