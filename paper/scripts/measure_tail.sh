#!/usr/bin/env bash
# Re-check the volatile dense-32k cell (memory-pressure variance on 8 GB) and
# measure the 64k reach point for both engines. Sequential, isolated (8 GB-safe).
set -u
cd "$(dirname "$0")/../.."
source diffkv_venv/bin/activate 2>/dev/null
RES=benchmarks/results
LOG=$RES/fresh_tail.log
: > "$LOG"
ACTIVE_ENV="DIFFKV_COMPRESSED_DECODE=1 DIFFKV_DECODE_CACHE=1 DIFFKV_SPARSE_PREFILL=1 \
DIFFKV_MAX_RESIDUAL=128 DIFFKV_ROUTER=residual DIFFKV_TOPK_BLOCKS=16 DIFFKV_SVD_SEED=1234"

run() {  # engine ctx outfile
  local engine=$1 ctx=$2 out=$3
  echo "--- $engine ctx=$ctx $(date +%H:%M:%S) ---" | tee -a "$LOG"
  if [ "$engine" = active ]; then
    env $ACTIVE_ENV python benchmarks/bench_worker.py --engine active --ctx "$ctx" --gen 128 \
        --prompt-file "$RES/prompt_$ctx.txt" --result-file "$out" >> "$LOG" 2>&1
  else
    python benchmarks/bench_worker.py --engine dense --ctx "$ctx" --gen 128 \
        --prompt-file "$RES/prompt_$ctx.txt" --result-file "$out" >> "$LOG" 2>&1
  fi
  python - "$out" <<'PY' | tee -a "$LOG"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"  {d.get('engine')} {d.get('ctx_target')}: pf={d.get('prefill_s'):.1f}s tps={d.get('decode_tps'):.1f} mx={d.get('mx_peak_gb')} needle={d.get('needle_found')} status={d.get('status')}")
except Exception as e:
    print("  parse-fail", e)
PY
}

run dense   32768 "$RES/fresh_dense_32768_recheck.json"
run active  65536 "$RES/fresh_active_65536.json"
run dense   65536 "$RES/fresh_dense_65536.json"
echo "== tail done $(date +%H:%M:%S) ==" | tee -a "$LOG"
