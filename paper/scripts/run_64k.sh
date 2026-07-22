#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
source dkv_venv/bin/activate 2>/dev/null
RES=benchmarks/results
LOG=$RES/clean_64k.log
: > "$LOG"
export DKV_DECODE_CACHE=1 DKV_SPARSE_PREFILL=1 DKV_SPARSE_BIAS=auto \
       DKV_ROUTER=residual DKV_TOPK_BLOCKS=16 DKV_SVD_SEED=1234 DKV_PRESET=mid \
       DKV_COMPRESSED_DECODE=1 DKV_MAX_RESIDUAL=128
for engine in active dense; do
  echo "--- $engine 64k $(date +%H:%M:%S) ---" | tee -a "$LOG"
  python paper/scripts/cell_worker.py --engine "$engine" --ctx 65536 --gen 128 \
      --prompt-file "$RES/prompt_65536.txt" --result-file "$RES/clean_${engine}_65536.json" >> "$LOG" 2>&1
  python - "$RES/clean_${engine}_65536.json" <<'PY' | tee -a "$LOG"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"  {d.get('engine')} 64k: status={d.get('status')} pf={d.get('prefill_s')} tps={d.get('decode_tps')} mx={d.get('mx_peak_gb')} needle={d.get('needle_found')} {d.get('error','')}")
except Exception as e: print("  no-result", e)
PY
done
echo "== 64k done $(date +%H:%M:%S) ==" | tee -a "$LOG"
