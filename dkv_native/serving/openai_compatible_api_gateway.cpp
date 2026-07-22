#include "serving/openai_compatible_api_gateway.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <random>
#include <ctime>

namespace dkv {

static std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++) ss << dis(gen);
    ss << "-4";
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    ss << dis2(gen);
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << dis(gen);
    return ss.str();
}

static std::string json_escape(const std::string& s) {
    std::string res;
    for (char c : s) {
        if (c == '"') res += "\\\"";
        else if (c == '\\') res += "\\\\";
        else if (c == '\n') res += "\\n";
        else if (c == '\r') res += "\\r";
        else if (c == '\t') res += "\\t";
        else res += c;
    }
    return res;
}

static std::string apply_chat_template(const std::vector<ChatMessage>& messages) {
    std::string prompt;
    for (const auto& msg : messages) {
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

DKVGateway::DKVGateway(
    DKVBatchEngine* batch_engine,
    ProductionSessionManager* session_manager
) : batch_engine_(batch_engine),
    session_manager_(session_manager) {}

DKVGateway::~DKVGateway() {}

bool DKVGateway::is_ephemeral_request(const std::vector<ChatMessage>& messages) {
    if (messages.empty()) return false;
    const auto& last_msg = messages.back();
    if (last_msg.role != "user") return false;
    
    std::string content = last_msg.content;
    std::string content_lower = content;
    std::transform(content_lower.begin(), content_lower.end(), content_lower.begin(), ::tolower);
    
    std::vector<std::string> keywords = {
        "title", "summarize", "summary", "label", "name this", "name the", "name of the"
    };
    bool has_keyword = false;
    for (const auto& kw : keywords) {
        if (content_lower.find(kw) != std::string::npos) {
            has_keyword = true;
            break;
        }
    }
    
    bool has_assistant_marker = (
        content_lower.find("assistant:") != std::string::npos ||
        content_lower.find("assistant\n") != std::string::npos ||
        content_lower.find("<|im_start|>assistant") != std::string::npos ||
        content_lower.find("assistant role") != std::string::npos ||
        content_lower.find("bot:") != std::string::npos ||
        content_lower.find("ai:") != std::string::npos ||
        content_lower.find("response:") != std::string::npos
    );
    
    if (messages.size() == 1 && has_keyword && has_assistant_marker) {
        return true;
    }
    
    if (content.size() < 500 && has_keyword) {
        return true;
    }
    
    std::vector<std::string> keywords_long = {
        "summarizing the chat history",
        "concise, 3-5 word title",
        "concise title with an emoji",
        "emoji summarizing the chat",
        "generate a concise, 3-5 word title",
        "generate a concise title with an emoji",
        "json format: { \"title\":",
        "json format: {\"title\":",
        "guidelines:\n- the title should clearly represent"
    };
    for (const auto& kw : keywords_long) {
        if (content_lower.find(kw) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string DKVGateway::optimize_ephemeral_prompt(const std::string& content) {
    if (content.size() <= 3000) {
        return content;
    }
    
    std::string content_lower = content;
    std::transform(content_lower.begin(), content_lower.end(), content_lower.begin(), ::tolower);
    
    std::vector<std::string> markers = {
        "chat history:", "conversation history:", "history:", "messages:", "context:"
    };
    for (const auto& marker : markers) {
        size_t idx = content_lower.find(marker);
        if (idx != std::string::npos) {
            std::string prefix = content.substr(0, idx + marker.size());
            std::string history = content.substr(idx + marker.size());
            if (history.size() > 2000) {
                history = history.substr(0, 1000) + "\n... [TRUNCATED FOR SPEED] ...\n" + history.substr(history.size() - 1000);
            }
            std::cout << "[DKV Gateway] Optimized long title/summary prompt by truncating embedded chat history." << std::endl;
            return prefix + history;
        }
    }
    
    std::string truncated = content.substr(0, 1500) + "\n... [TRUNCATED FOR SPEED] ...\n" + content.substr(content.size() - 1500);
    std::cout << "[DKV Gateway] Optimized long title/summary prompt by middle-truncation." << std::endl;
    return truncated;
}

std::string DKVGateway::match_session_by_history(const std::vector<ChatMessage>& incoming) {
    if (incoming.size() <= 1 || !session_manager_) {
        return "";
    }
    
    std::vector<ChatMessage> prefix_history(incoming.begin(), incoming.end() - 1);
    
    std::lock_guard<std::mutex> lock(session_manager_->get_mutex());
    const auto& active_sessions = session_manager_->get_active_sessions();
    
    for (auto const& [sid, session] : active_sessions) {
        if (sid.rfind("__ephemeral__", 0) == 0) {
            continue;
        }
        
        if (session->history.size() == prefix_history.size()) {
            bool match = true;
            for (size_t i = 0; i < prefix_history.size(); ++i) {
                std::string h_content = session->history[i].content;
                std::string p_content = prefix_history[i].content;
                
                auto trim = [](std::string& s) {
                    s.erase(0, s.find_first_not_of(" \t\r\n"));
                    s.erase(s.find_last_not_of(" \t\r\n") + 1);
                };
                trim(h_content);
                trim(p_content);
                
                if (session->history[i].role != prefix_history[i].role || h_content != p_content) {
                    match = false;
                    break;
                }
            }
            if (match) {
                std::cout << "[DKV Gateway] Dynamically matched message history prefix to active session: " << sid << std::endl;
                return sid;
            }
        }
    }
    
    ChatMessage last_incoming_assistant;
    bool found_assistant = false;
    for (auto it = prefix_history.rbegin(); it != prefix_history.rend(); ++it) {
        if (it->role == "assistant") {
            last_incoming_assistant = *it;
            found_assistant = true;
            break;
        }
    }
    
    if (found_assistant) {
        for (auto const& [sid, session] : active_sessions) {
            if (sid.rfind("__ephemeral__", 0) == 0) {
                continue;
            }
            
            ChatMessage last_stored_assistant;
            bool found_stored = false;
            for (auto it = session->history.rbegin(); it != session->history.rend(); ++it) {
                if (it->role == "assistant") {
                    last_stored_assistant = *it;
                    found_stored = true;
                    break;
                }
            }
            
            if (found_stored) {
                std::string h_last = last_stored_assistant.content;
                std::string p_last = last_incoming_assistant.content;
                auto trim = [](std::string& s) {
                    s.erase(0, s.find_first_not_of(" \t\r\n"));
                    s.erase(s.find_last_not_of(" \t\r\n") + 1);
                };
                trim(h_last);
                trim(p_last);
                
                if (p_last.size() > 150 && h_last == p_last) {
                    std::cout << "[DKV Gateway] Dynamically matched session " << sid << " using fallback last assistant message content match." << std::endl;
                    return sid;
                }
            }
        }
    }
    
    return "";
}

std::string DKVGateway::handle_chat_completion(const ChatCompletionRequest& request) {
    std::string session_id = request.session_id;
    bool is_ephemeral = is_ephemeral_request(request.messages);
    
    std::vector<ChatMessage> messages = request.messages;
    
    if (is_ephemeral) {
        std::string orig_id = session_id.empty() ? generate_uuid() : session_id;
        session_id = "__ephemeral__" + orig_id;
        for (auto& m : messages) {
            if (m.role == "user") {
                m.content = optimize_ephemeral_prompt(m.content);
            }
        }
        if (session_manager_) {
            session_manager_->create_session_with_id(session_id);
        }
    } else {
        if (session_id.empty()) {
            session_id = match_session_by_history(messages);
        }
        if (session_id.empty()) {
            if (session_manager_) {
                session_id = session_manager_->create_session();
            } else {
                session_id = generate_uuid();
            }
        }
    }
    
    std::vector<ChatMessage> full_messages = messages;
    if (session_manager_ && !is_ephemeral) {
        auto session = session_manager_->get_session(session_id);
        if (session && full_messages.size() == 1 && full_messages[0].role == "user") {
            std::vector<ChatMessage> combined_msgs;
            for (const auto& msg : session->history) {
                combined_msgs.push_back({msg.role, msg.content});
            }
            combined_msgs.push_back(full_messages[0]);
            full_messages = std::move(combined_msgs);
        }
    }
    
    std::string prompt = apply_chat_template(full_messages);
    
    auto start_time = std::chrono::system_clock::now();
    long long created_time = std::chrono::duration_cast<std::chrono::seconds>(start_time.time_since_epoch()).count();
    std::string request_id = "chatcmpl-" + generate_uuid();
    
    if (!batch_engine_) {
        return "{\"error\": \"Batch engine is not initialized.\"}";
    }
    
    auto req_ptr = batch_engine_->submit(
        session_id,
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.repetition_penalty
    );
    
    // Synchronously wait for completion
    std::string result_text;
    {
        std::unique_lock<std::mutex> lock(req_ptr->output_mutex);
        req_ptr->output_cv.wait(lock, [&]() {
            return req_ptr->stream_finished;
        });
        
        if (!req_ptr->error_msg.empty()) {
            return "{\"error\": \"" + json_escape(req_ptr->error_msg) + "\"}";
        }
        
        // Accumulate chunks
        std::stringstream res_ss;
        for (const auto& chunk : req_ptr->output_chunks) {
            res_ss << chunk;
        }
        result_text = res_ss.str();
    }
    
    // Save history and prefix registry updates
    if (session_manager_ && !is_ephemeral) {
        auto session = session_manager_->get_session(session_id);
        if (session) {
            session->history.clear();
            for (const auto& msg : full_messages) {
                session->history.push_back({msg.role, msg.content});
            }
            session->history.push_back({"assistant", result_text});
            session_manager_->save_session(session_id);
        }
    }
    
    if (is_ephemeral && session_manager_) {
        session_manager_->delete_session(session_id);
    }
    
    int prompt_tokens = req_ptr->prompt_tokens.size();
    int completion_tokens = req_ptr->generated_tokens.size();
    
    return format_non_stream_json(request_id, created_time, request.model, result_text, prompt_tokens, completion_tokens);
}

void DKVGateway::handle_chat_completion_stream(
    const ChatCompletionRequest& request,
    std::function<void(const std::string& sse_line)> callback
) {
    std::string session_id = request.session_id;
    bool is_ephemeral = is_ephemeral_request(request.messages);
    
    std::vector<ChatMessage> messages = request.messages;
    
    if (is_ephemeral) {
        std::string orig_id = session_id.empty() ? generate_uuid() : session_id;
        session_id = "__ephemeral__" + orig_id;
        for (auto& m : messages) {
            if (m.role == "user") {
                m.content = optimize_ephemeral_prompt(m.content);
            }
        }
        if (session_manager_) {
            session_manager_->create_session_with_id(session_id);
        }
    } else {
        if (session_id.empty()) {
            session_id = match_session_by_history(messages);
        }
        if (session_id.empty()) {
            if (session_manager_) {
                session_id = session_manager_->create_session();
            } else {
                session_id = generate_uuid();
            }
        }
    }
    
    std::vector<ChatMessage> full_messages = messages;
    if (session_manager_ && !is_ephemeral) {
        auto session = session_manager_->get_session(session_id);
        if (session && full_messages.size() == 1 && full_messages[0].role == "user") {
            std::vector<ChatMessage> combined_msgs;
            for (const auto& msg : session->history) {
                combined_msgs.push_back({msg.role, msg.content});
            }
            combined_msgs.push_back(full_messages[0]);
            full_messages = std::move(combined_msgs);
        }
    }
    
    std::string prompt = apply_chat_template(full_messages);
    
    auto start_time = std::chrono::system_clock::now();
    long long created_time = std::chrono::duration_cast<std::chrono::seconds>(start_time.time_since_epoch()).count();
    std::string request_id = "chatcmpl-" + generate_uuid();
    
    if (!batch_engine_) {
        callback("data: {\"error\": \"Batch engine is not initialized.\"}\n\n");
        callback("data: [DONE]\n\n");
        return;
    }
    
    auto req_ptr = batch_engine_->submit(
        session_id,
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.repetition_penalty
    );
    
    size_t chunk_idx = 0;
    std::string full_generated;
    
    while (true) {
        std::vector<std::string> chunks_to_process;
        bool finished = false;
        std::string err;
        
        {
            std::unique_lock<std::mutex> lock(req_ptr->output_mutex);
            req_ptr->output_cv.wait_for(lock, std::chrono::seconds(5), [&]() {
                return req_ptr->output_chunks.size() > chunk_idx || req_ptr->stream_finished;
            });
            
            // Collect any new chunks
            while (chunk_idx < req_ptr->output_chunks.size()) {
                chunks_to_process.push_back(req_ptr->output_chunks[chunk_idx]);
                full_generated += req_ptr->output_chunks[chunk_idx];
                chunk_idx++;
            }
            
            finished = req_ptr->stream_finished;
            err = req_ptr->error_msg;
        }
        
        // Deliver chunks to SSE callback
        for (const auto& chunk : chunks_to_process) {
            callback(format_stream_chunk_sse(request_id, created_time, request.model, chunk, false));
        }
        
        if (!err.empty()) {
            callback("data: {\"error\": \"" + json_escape(err) + "\"}\n\n");
            break;
        }
        
        if (finished && chunks_to_process.empty()) {
            break;
        }
    }
    
    // Final stop chunk
    callback(format_stream_chunk_sse(request_id, created_time, request.model, "", true));
    callback("data: [DONE]\n\n");
    
    // Save history and prefix registry updates
    if (session_manager_ && !is_ephemeral) {
        auto session = session_manager_->get_session(session_id);
        if (session) {
            session->history.clear();
            for (const auto& msg : full_messages) {
                session->history.push_back({msg.role, msg.content});
            }
            session->history.push_back({"assistant", full_generated});
            session_manager_->save_session(session_id);
        }
    }
    
    if (is_ephemeral && session_manager_) {
        session_manager_->delete_session(session_id);
    }
}

std::string DKVGateway::create_session() {
    if (session_manager_) {
        std::string sid = session_manager_->create_session();
        return "{\"session_id\": \"" + sid + "\"}";
    }
    return "{\"session_id\": \"" + generate_uuid() + "\"}";
}

void DKVGateway::delete_session(const std::string& session_id) {
    if (session_manager_) {
        session_manager_->delete_session(session_id);
    }
    if (batch_engine_) {
        batch_engine_->cancel(session_id);
    }
}

std::string DKVGateway::list_models() {
    std::string model_id = "dkv-serving";
    long long created_time = std::time(nullptr);
    
    std::stringstream ss;
    ss << "{\n"
       << "  \"object\": \"list\",\n"
       << "  \"data\": [{\n"
       << "    \"id\": \"" << model_id << "\",\n"
       << "    \"name\": \"" << model_id << "\",\n"
       << "    \"object\": \"model\",\n"
       << "    \"created\": " << created_time << ",\n"
       << "    \"owned_by\": \"differential-kv\"\n"
       << "  }]\n"
       << "}";
    return ss.str();
}

std::string DKVGateway::get_runtime_info() {
    // Return hardware and model stats mimicking runtime_info in Python gateway
    double mps_allocated_gb = 0.0;
    double process_rss_gb = 0.0;
    
    // Check if we are running Qwen model
    std::string model_name = "dkv-serving";
    if (batch_engine_ && batch_engine_->get_model()) {
        const auto& cfg = batch_engine_->get_model()->get_config();
        model_name = "Qwen2.5-" + std::to_string(cfg.n_layer) + "L";
    }
    
    std::stringstream ss;
    ss << "{\n"
       << "  \"vram_allocated_gb\": " << mps_allocated_gb << ",\n"
       << "  \"cuda_available\": false,\n"
       << "  \"mps_available\": true,\n"
       << "  \"sampling_mode\": \"temperature+top_p+repetition_penalty\",\n"
       << "  \"streaming_mode\": \"phrase_group_chunked\",\n"
       << "  \"serving_mode\": \"balanced\",\n"
       << "  \"model\": \"" << model_name << "\",\n"
       << "  \"device\": \"mps\",\n"
       << "  \"process_rss_gb\": " << process_rss_gb << ",\n"
       << "  \"kv_summary\": {}\n"
       << "}";
    return ss.str();
}

std::string DKVGateway::get_session_srl_info(const std::string& session_id) {
    if (!session_manager_) {
        return "{\"error\": \"Session manager is not initialized.\"}";
    }
    
    auto session = session_manager_->get_session(session_id);
    if (!session) {
        return "{\"error\": \"Session " + session_id + " not found.\"}";
    }
    
    std::stringstream ss;
    ss << "{\n"
       << "  \"session_id\": \"" << session_id << "\",\n"
       << "  \"srl_built\": true,\n"
       << "  \"active_blocks\": " << session->layers_blocks.size() << ",\n"
       << "  \"k_min\": " << session->srl_state.k_min << ",\n"
       << "  \"k_max\": " << session->srl_state.k_max << ",\n"
       << "  \"routing_threshold\": " << session->srl_state.routing_threshold << ",\n"
       << "  \"srl_age_penalty\": " << session->srl_state.srl_age_penalty << "\n"
       << "}";
    return ss.str();
}

std::string DKVGateway::health_check() {
    return "{\"status\": \"ok\"}";
}

std::string DKVGateway::format_non_stream_json(
    const std::string& request_id,
    long long created_time,
    const std::string& model,
    const std::string& result_text,
    int prompt_tokens,
    int completion_tokens
) {
    std::stringstream ss;
    ss << "{\n"
       << "  \"id\": \"" << request_id << "\",\n"
       << "  \"object\": \"chat.completion\",\n"
       << "  \"created\": " << created_time << ",\n"
       << "  \"model\": \"" << json_escape(model) << "\",\n"
       << "  \"choices\": [\n"
       << "    {\n"
       << "      \"index\": 0,\n"
       << "      \"message\": {\n"
       << "        \"role\": \"assistant\",\n"
       << "        \"content\": \"" << json_escape(result_text) << "\"\n"
       << "      },\n"
       << "      \"finish_reason\": \"stop\"\n"
       << "    }\n"
       << "  ],\n"
       << "  \"usage\": {\n"
       << "    \"prompt_tokens\": " << prompt_tokens << ",\n"
       << "    \"completion_tokens\": " << completion_tokens << ",\n"
       << "    \"total_tokens\": " << (prompt_tokens + completion_tokens) << "\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string DKVGateway::format_stream_chunk_sse(
    const std::string& request_id,
    long long created_time,
    const std::string& model,
    const std::string& chunk_text,
    bool is_final
) {
    std::stringstream ss;
    ss << "data: {";
    ss << "\"id\":\"" << request_id << "\",";
    ss << "\"object\":\"chat.completion.chunk\",";
    ss << "\"created\":" << created_time << ",";
    ss << "\"model\":\"" << json_escape(model) << "\",";
    ss << "\"choices\":[{";
    ss << "\"index\":0,";
    ss << "\"delta\":{\"content\":\"" << json_escape(chunk_text) << "\"},";
    if (is_final) {
        ss << "\"finish_reason\":\"stop\"";
    } else {
        ss << "\"finish_reason\":null";
    }
    ss << "}]";
    ss << "}\n\n";
    return ss.str();
}

} // namespace dkv
