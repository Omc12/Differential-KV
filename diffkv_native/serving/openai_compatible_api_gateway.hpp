#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include "serving/batch_engine.hpp"
#include "serving/production_session_manager.hpp"

namespace diffkv {

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    bool stream = false;
    int max_tokens = 2048;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float repetition_penalty = 1.15f;
    std::string session_id = "";
    
    // SRL config overrides
    bool srl_enabled = true;
    int srl_threshold = 50;
    int srl_k_min = 20;
    int srl_k_max = 200;
    float srl_age_penalty = 0.01f;
};

class DiffKVGateway {
public:
    DiffKVGateway(
        DiffKVBatchEngine* batch_engine,
        ProductionSessionManager* session_manager
    );
    ~DiffKVGateway();

    // Core endpoints
    std::string handle_chat_completion(const ChatCompletionRequest& request);
    
    void handle_chat_completion_stream(
        const ChatCompletionRequest& request,
        std::function<void(const std::string& sse_line)> callback
    );

    std::string create_session();
    void delete_session(const std::string& session_id);
    std::string list_models();
    std::string get_runtime_info();
    std::string get_session_srl_info(const std::string& session_id);
    std::string health_check();

    // Heuristics for Open WebUI & Summarization requests
    bool is_ephemeral_request(const std::vector<ChatMessage>& messages);
    std::string optimize_ephemeral_prompt(const std::string& content);

private:
    std::string format_non_stream_json(
        const std::string& request_id,
        long long created_time,
        const std::string& model,
        const std::string& result_text,
        int prompt_tokens,
        int completion_tokens
    );

    std::string format_stream_chunk_sse(
        const std::string& request_id,
        long long created_time,
        const std::string& model,
        const std::string& chunk_text,
        bool is_final
    );

    std::string match_session_by_history(const std::vector<ChatMessage>& incoming);

    DiffKVBatchEngine* batch_engine_;
    ProductionSessionManager* session_manager_;
};

} // namespace diffkv
