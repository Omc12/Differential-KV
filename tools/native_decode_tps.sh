#!/bin/bash
# native_decode_tps.sh — reproducible native DECODE tps measurement.
#
# Forces the DiffKV sparse decode path at low context (ENGAGE_THRESHOLD=512) and
# uses a long-output prompt so the DIFFKV_PROFILE "Total Step Time" is averaged
# over many decode tokens (decode-only; prefill is excluded from that average).
#
#   tps = 1000 / Total_Step_Time_ms
#
# Usage: tools/native_decode_tps.sh [ctx_tokens] [max_gen] [extra envs...]
#   ctx_tokens : approx prompt length (default 2000)
#   max_gen    : decode token cap (default 150)
# Any trailing NAME=VALUE args are exported (e.g. DIFFKV_DECODE_CACHE=1).
set -e
cd "$(dirname "$0")/.."

CTX="${1:-2000}"; MAX_GEN="${2:-150}"; shift 2 2>/dev/null || true
for kv in "$@"; do export "$kv"; done

BINARY="diffkv_native/build/diffkv_native"
MODEL="diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
QUESTION="Ignore the text above. Write an extremely long and detailed essay about the history, present, and future of artificial intelligence. Write at least 600 words and do not stop early."
NEEDLE="(background note) the field has many subfields."

source diffkv_venv/bin/activate 2>/dev/null || true
PROMPT=$(python3 diffkv_native/tests/make_niah_prompt.py "$CTX" 0.5 "$NEEDLE" "$QUESTION")

export DIFFKV_ENGAGE_THRESHOLD=512   # force sparse at low ctx
export DIFFKV_COMPRESSED_DECODE=1    # sparse from token 1
export DIFFKV_TEMPERATURE=0          # greedy / deterministic
export DIFFKV_MAX_TOKENS="$MAX_GEN"
export DIFFKV_PROFILE=1
export HF_HUB_OFFLINE=1

echo "[native_decode_tps] ctx≈$CTX max_gen=$MAX_GEN  extra: $*"
"$BINARY" "$MODEL" "$PROMPT" 2>&1 | grep -E "Total Step Time|Attention:|Reconstruction:|decode_use_sparse|Averaged over" | tail -8
