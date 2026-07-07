#!/usr/bin/env bash
# CLEAN DiffKV measurement: one isolated process at a time, nothing else running.
# Config = ACTIVE CLI (mid / balanced / rank 32 / int4) with DiffKV sparse forced
# on at every context; see paper/scripts/cell_worker.py for the exact flags.
set -u
cd "$(dirname "$0")/../.."
source diffkv_venv/bin/activate 2>/dev/null
RES=benchmarks/results
LOG=$RES/clean_${TAG:-sweep}.log
: > "$LOG"
CTXS=${CTXS:-"4096 8192 16384 32768"}
GEN=${GEN:-128}

echo "== CLEAN sweep start $(date) ==" | tee -a "$LOG"
echo "config: active = mid/balanced/rank32/int4, COMPRESSED_DECODE=1 DECODE_CACHE=1 SPARSE_PREFILL=1 SPARSE_BIAS=auto MAX_RESIDUAL=128" | tee -a "$LOG"
for ctx in $CTXS; do
  P=$RES/prompt_$ctx.txt
  [ -f "$P" ] || { echo "MISSING $P" | tee -a "$LOG"; continue; }
  for engine in dense active; do
    echo "--- $engine ctx=$ctx  $(date +%H:%M:%S) ---" | tee -a "$LOG"
    python paper/scripts/cell_worker.py --engine "$engine" --ctx "$ctx" --gen "$GEN" \
        --prompt-file "$P" --result-file "$RES/clean_${engine}_$ctx.json" >> "$LOG" 2>&1
    python - "$RES/clean_${engine}_$ctx.json" <<'PY' | tee -a "$LOG"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    if d.get("status")=="ok":
        print(f"  {d['engine']} {d['ctx_target']}: pf={d['prefill_s']:.1f}s tps={d['decode_tps']:.1f} mx={d['mx_peak_gb']:.3f}GB needle={d['needle_found']}")
    else:
        print("  ERROR", d.get("error"))
except Exception as e:
    print("  parse-fail", e)
PY
  done
done
echo "== CLEAN sweep done $(date) ==" | tee -a "$LOG"
