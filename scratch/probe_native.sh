#!/bin/bash
# Controlled native-binary probe: measures startup RSS, prefill time, peak RSS, and output.
# Usage: probe_native.sh <srl_k_keep> <tag>
set -u
cd "$(dirname "$0")/.."

KKEEP="${1:-16}"
TAG="${2:-default}"
MODEL="diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PROMPT="scratch/test_bigprompt.txt"
OUT="scratch/probe_${TAG}.out"
ERR="scratch/probe_${TAG}.err"
RSS="scratch/probe_${TAG}.rss"
BIN="diffkv_native/build/diffkv_native"

# Replicate cli.py environment
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DIFFKV_PRESET=mid
export DIFFKV_PREFILL_CHUNK_SIZE=512
export DIFFKV_COMPRESSOR_THREADS=4
export DIFFKV_MAX_TOKENS=60
export DIFFKV_USE_GPU=1
export DIFFKV_MICRO_BLOCK_SIZE=256
export DIFFKV_TEMPERATURE=0.7 DIFFKV_TOP_P=0.9 DIFFKV_REPETITION_PENALTY=1.15
export DIFFKV_VERBOSE=1
export DIFFKV_SRL_K_KEEP="$KKEEP"

: > "$RSS"
# Launch binary, feed prompt on stdin
"$BIN" "$MODEL" < "$PROMPT" > "$OUT" 2> "$ERR" &
PID=$!

T0=$(python3 -c 'import time;print(time.time())')
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  R=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$R" ]; then
    NOW=$(python3 -c 'import time;print(round(time.time()-'"$T0"',1))')
    MB=$(( R / 1024 ))
    echo "$NOW $MB" >> "$RSS"
    [ "$MB" -gt "$PEAK" ] && PEAK="$MB"
  fi
  sleep 0.5
done
wait "$PID"
RC=$?
T1=$(python3 -c 'import time;print(round(time.time()-'"$T0"',1))')
echo "=== TAG=$TAG srl_k_keep=$KKEEP wall=${T1}s peak_rss=${PEAK}MB rc=$RC ===" | tee -a "scratch/probe_summary.txt"
