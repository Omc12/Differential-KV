#!/usr/bin/env bash
# Fresh DiffKV active-vs-dense sweep on the CURRENT working tree.
# Each (engine, ctx) runs in an isolated process (8 GB-safe; clean mem attribution).
# active = DiffKV in its default serving configuration:
#   compressed sparse decode + decode-cache (bit-exact fast path) + sparse prefill,
#   max_residual=128, rank=16, router=residual, topk=16.
# dense  = mlx_lm full KV, SAME int4 weights.
# Writes benchmarks/results/fresh_{engine}_{ctx}.json  (+ a live log).
set -u
cd "$(dirname "$0")/../.."          # repo root
source diffkv_venv/bin/activate 2>/dev/null
RES=benchmarks/results
LOG=$RES/fresh_sweep.log
: > "$LOG"

CTXS=${CTXS:-"4096 8192 16384 32768"}
GEN=${GEN:-128}

ACTIVE_ENV="DIFFKV_COMPRESSED_DECODE=1 DIFFKV_DECODE_CACHE=1 DIFFKV_SPARSE_PREFILL=1 \
DIFFKV_MAX_RESIDUAL=128 DIFFKV_ROUTER=residual DIFFKV_TOPK_BLOCKS=16 DIFFKV_SVD_SEED=1234"

echo "== fresh sweep $(date) ==" | tee -a "$LOG"
for ctx in $CTXS; do
  P=$RES/prompt_$ctx.txt
  [ -f "$P" ] || { echo "MISSING $P" | tee -a "$LOG"; continue; }

  echo "--- dense ctx=$ctx ---" | tee -a "$LOG"
  python benchmarks/bench_worker.py --engine dense --ctx "$ctx" --gen "$GEN" \
      --prompt-file "$P" --result-file "$RES/fresh_dense_$ctx.json" >> "$LOG" 2>&1
  python - "$RES/fresh_dense_$ctx.json" <<'PY' | tee -a "$LOG"
import json,sys; d=json.load(open(sys.argv[1]))
print(f"  dense {d.get('ctx_target')}: pf={d.get('prefill_s'):.1f}s tps={d.get('decode_tps'):.1f} mx={d.get('mx_peak_gb')} needle={d.get('needle_found')} status={d.get('status')}")
PY

  echo "--- active ctx=$ctx ---" | tee -a "$LOG"
  env $ACTIVE_ENV python benchmarks/bench_worker.py --engine active --ctx "$ctx" --gen "$GEN" \
      --prompt-file "$P" --result-file "$RES/fresh_active_$ctx.json" >> "$LOG" 2>&1
  python - "$RES/fresh_active_$ctx.json" <<'PY' | tee -a "$LOG"
import json,sys; d=json.load(open(sys.argv[1]))
print(f"  active {d.get('ctx_target')}: pf={d.get('prefill_s'):.1f}s tps={d.get('decode_tps'):.1f} mx={d.get('mx_peak_gb')} needle={d.get('needle_found')} status={d.get('status')}")
PY
done
echo "== sweep done $(date) ==" | tee -a "$LOG"
