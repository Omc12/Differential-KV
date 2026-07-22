#!/bin/bash
# native_decode_tps.sh — reproducible native DECODE tps measurement.
#
# Forces the DKV sparse decode path at low context (ENGAGE_THRESHOLD=512) and
# uses a long-output prompt so the DKV_PROFILE "Total Step Time" is averaged
# over many decode tokens (decode-only; prefill is excluded from that average).
#
#   tps = 1000 / Total_Step_Time_ms
#
# Usage: tools/native_decode_tps.sh [ctx_tokens] [max_gen] [extra envs...]
#   ctx_tokens : approx prompt length (default 2000)
#   max_gen    : decode token cap (default 150)
# Any trailing NAME=VALUE args are exported (e.g. DKV_DECODE_CACHE=1).
set -e
cd "$(dirname "$0")/.."

CTX="${1:-2000}"; MAX_GEN="${2:-150}"; shift 2 2>/dev/null || true
for kv in "$@"; do export "$kv"; done

BINARY="dkv_native/build/dkv_native"
MODEL="dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"
QUESTION="Ignore the text above. Write an extremely long and detailed essay about the history, present, and future of artificial intelligence. Write at least 600 words and do not stop early."
NEEDLE="(background note) the field has many subfields."

source dkv_venv/bin/activate 2>/dev/null || true
PROMPT=$(python3 dkv_native/tests/make_niah_prompt.py "$CTX" 0.5 "$NEEDLE" "$QUESTION")

export DKV_ENGAGE_THRESHOLD=512   # force sparse at low ctx
export DKV_COMPRESSED_DECODE=1    # sparse from token 1
export DKV_TEMPERATURE=0          # greedy / deterministic
export DKV_MAX_TOKENS="$MAX_GEN"
export DKV_PROFILE=1
export HF_HUB_OFFLINE=1

echo "[native_decode_tps] ctx≈$CTX max_gen=$MAX_GEN  extra: $*"
"$BINARY" "$MODEL" "$PROMPT" 2>&1 | grep -E "Total Step Time|Attention:|Reconstruction:|decode_use_sparse|Averaged over" | tail -8
