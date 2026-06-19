// diffkv_native/include/config.hpp
// Translation of config.py → C++17
// DiffKVConfig: preset-based configuration with environment-variable overrides.
// All methods are inline so this header is self-contained (no .cpp needed).

#pragma once

#include <string>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace diffkv {

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------
#ifdef __APPLE__
    static constexpr bool kIsMacOS = true;
#else
    static constexpr bool kIsMacOS = false;
#endif

// ---------------------------------------------------------------------------
// DiffKVConfig
// ---------------------------------------------------------------------------
class DiffKVConfig {
public:
    // -----------------------------------------------------------------------
    // Fields (public so callers can read them directly)
    // -----------------------------------------------------------------------
    std::string preset;                 // "low" | "mid" | "high"
    bool        decode_cache_enabled;
    int         decode_cache_max_tokens;
    int         prefill_chunk_size;
    int         srl_threshold;
    bool        async_svd;
    float       mps_watermark;
    bool        torch_compile;
    bool        approximate_attn;
    float       srl_age_penalty;
    bool        early_layer_rank_boost;
    int         max_rank_early;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    // config_map may contain any of:
    //   "preset", and/or any field name as string key → string value.
    // Environment variables take precedence over config_map values.
    // -----------------------------------------------------------------------
    explicit DiffKVConfig(
        const std::unordered_map<std::string, std::string>& config_map = {})
    {
        // 1. Determine preset
        preset = "mid"; // default
        {
            const char* env_preset = std::getenv("DIFFKV_PRESET");
            if (env_preset && *env_preset)
                preset = env_preset;
            else {
                auto it = config_map.find("preset");
                if (it != config_map.end() && !it->second.empty())
                    preset = it->second;
            }
        }

        // Normalise preset to lowercase
        std::transform(preset.begin(), preset.end(), preset.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        // 2. Apply preset defaults
        apply_preset_defaults(preset);

        // 3. Apply config_map overrides (lower priority than env vars)
        apply_map_overrides(config_map);

        // 4. Apply environment-variable overrides (highest priority)
        apply_env_overrides();
    }

    // -----------------------------------------------------------------------
    // Static env-parsing helpers (public so external code can reuse them)
    // -----------------------------------------------------------------------

    /// Returns true/false from env var name, or default_val if not set.
    static bool get_bool_env(const char* name, bool default_val) {
        const char* v = std::getenv(name);
        if (!v || !*v) return default_val;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (s == "1" || s == "true" || s == "yes" || s == "on")  return true;
        if (s == "0" || s == "false"|| s == "no"  || s == "off") return false;
        return default_val; // unrecognised → keep default
    }

    /// Returns int from env var name, or default_val if not set / unparseable.
    static int get_int_env(const char* name, int default_val) {
        const char* v = std::getenv(name);
        if (!v || !*v) return default_val;
        try { return std::stoi(v); }
        catch (...) { return default_val; }
    }

    /// Returns float from env var name, or default_val if not set / unparseable.
    static float get_float_env(const char* name, float default_val) {
        const char* v = std::getenv(name);
        if (!v || !*v) return default_val;
        try { return std::stof(v); }
        catch (...) { return default_val; }
    }

private:
    // -----------------------------------------------------------------------
    // Preset defaults
    // -----------------------------------------------------------------------
    void apply_preset_defaults(const std::string& p) {
        if (p == "low") {
            decode_cache_enabled    = false;
            decode_cache_max_tokens = 0;
            prefill_chunk_size      = 512;
            srl_threshold           = 30;
            async_svd               = false;
            mps_watermark           = 0.0f;
            torch_compile           = false;
            approximate_attn        = false;
            srl_age_penalty         = 0.01f;
            early_layer_rank_boost  = false;
            max_rank_early          = 32;
        }
        else if (p == "high") {
            decode_cache_enabled    = true;
            decode_cache_max_tokens = 16384;
            prefill_chunk_size      = 2048;
            srl_threshold           = 100;
            async_svd               = kIsMacOS ? false : true;
            mps_watermark           = 0.0f;
            torch_compile           = kIsMacOS ? false : true;
            approximate_attn        = false;
            srl_age_penalty         = 0.01f;
            early_layer_rank_boost  = true;
            max_rank_early          = 64;
        }
        else {
            // "mid" or anything unrecognised → mid defaults
            if (p != "mid") {
                // warn to stderr but continue
                std::fprintf(stderr,
                    "[DiffKVConfig] Warning: unknown preset '%s', using 'mid'.\n",
                    p.c_str());
                const_cast<std::string&>(preset) = "mid";
            }
            decode_cache_enabled    = true;
            decode_cache_max_tokens = 4096;
            prefill_chunk_size      = 512;
            srl_threshold           = 50;
            async_svd               = kIsMacOS ? false : true;
            mps_watermark           = 0.0f;
            torch_compile           = false;
            approximate_attn        = false;
            srl_age_penalty         = 0.01f;
            early_layer_rank_boost  = false;
            max_rank_early          = 48;
        }
    }

    // -----------------------------------------------------------------------
    // Config-map overrides (applied before env-var overrides)
    // -----------------------------------------------------------------------
    void apply_map_overrides(const std::unordered_map<std::string, std::string>& m) {
        auto parse_bool = [](const std::string& s, bool cur) -> bool {
            std::string t = s;
            std::transform(t.begin(), t.end(), t.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (t == "1" || t == "true" || t == "yes" || t == "on")  return true;
            if (t == "0" || t == "false"|| t == "no"  || t == "off") return false;
            return cur;
        };
        auto parse_int = [](const std::string& s, int cur) -> int {
            try { return std::stoi(s); } catch (...) { return cur; }
        };
        auto parse_float = [](const std::string& s, float cur) -> float {
            try { return std::stof(s); } catch (...) { return cur; }
        };

        for (const auto& [key, val] : m) {
            if (key == "preset") continue; // already handled
            if (key == "decode_cache_enabled")
                decode_cache_enabled = parse_bool(val, decode_cache_enabled);
            else if (key == "decode_cache_max_tokens")
                decode_cache_max_tokens = parse_int(val, decode_cache_max_tokens);
            else if (key == "prefill_chunk_size")
                prefill_chunk_size = parse_int(val, prefill_chunk_size);
            else if (key == "srl_threshold")
                srl_threshold = parse_int(val, srl_threshold);
            else if (key == "async_svd")
                async_svd = parse_bool(val, async_svd);
            else if (key == "mps_watermark")
                mps_watermark = parse_float(val, mps_watermark);
            else if (key == "torch_compile")
                torch_compile = parse_bool(val, torch_compile);
            else if (key == "approximate_attn")
                approximate_attn = parse_bool(val, approximate_attn);
            else if (key == "srl_age_penalty")
                srl_age_penalty = parse_float(val, srl_age_penalty);
            else if (key == "early_layer_rank_boost")
                early_layer_rank_boost = parse_bool(val, early_layer_rank_boost);
            else if (key == "max_rank_early")
                max_rank_early = parse_int(val, max_rank_early);
            // unknown keys are silently ignored
        }
    }

    // -----------------------------------------------------------------------
    // Environment-variable overrides (highest priority)
    // -----------------------------------------------------------------------
    void apply_env_overrides() {
        decode_cache_enabled =
            get_bool_env("DIFFKV_DECODE_CACHE_ENABLED",  decode_cache_enabled);
        decode_cache_max_tokens =
            get_int_env ("DIFFKV_DECODE_CACHE_MAX_TOKENS", decode_cache_max_tokens);
        prefill_chunk_size =
            get_int_env ("DIFFKV_PREFILL_CHUNK_SIZE",    prefill_chunk_size);
        srl_threshold =
            get_int_env ("DIFFKV_SRL_THRESHOLD",         srl_threshold);
        async_svd =
            get_bool_env("DIFFKV_ASYNC_SVD",             async_svd);
        mps_watermark =
            get_float_env("PYTORCH_MPS_HIGH_WATERMARK_RATIO", mps_watermark);
        torch_compile =
            get_bool_env("DIFFKV_USE_TORCH_COMPILE",     torch_compile);
        approximate_attn =
            get_bool_env("DIFFKV_MPS_APPROXIMATE_ATTN",  approximate_attn);
        srl_age_penalty =
            get_float_env("DIFFKV_SRL_AGE_PENALTY",      srl_age_penalty);
        early_layer_rank_boost =
            get_bool_env("DIFFKV_EARLY_LAYER_RANK_BOOST", early_layer_rank_boost);
        max_rank_early =
            get_int_env ("DIFFKV_MAX_RANK_EARLY",        max_rank_early);
    }
};

} // namespace diffkv
