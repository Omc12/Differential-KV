#!/usr/bin/env bash
# CLEAN ablations + 64k reach, one process at a time, nothing else running.
# Same DiffKV config as the primary sweep (mid/balanced/rank32/int4, decode-cache,
# sparse-prefill, adaptive bias). Each internal cell is its own subprocess.
set -u
cd "$(dirname "$0")/../.."
source diffkv_venv/bin/activate 2>/dev/null
RES=benchmarks/results
LOG=$RES/clean_ablations.log
: > "$LOG"

# Shared DiffKV env (COMPRESSED_DECODE + MAX_RESIDUAL are set per-cell by the drivers).
export DIFFKV_DECODE_CACHE=1 DIFFKV_SPARSE_PREFILL=1 DIFFKV_SPARSE_BIAS=auto \
       DIFFKV_ROUTER=residual DIFFKV_TOPK_BLOCKS=16 DIFFKV_SVD_SEED=1234 DIFFKV_PRESET=mid

echo "== ablations start $(date) ==" | tee -a "$LOG"

echo "--- E6: decode-mode ablation (compressed vs exact), 4k-32k ---" | tee -a "$LOG"
DIFFKV_MAX_RESIDUAL=128 python paper/scripts/measure_active.py \
    --ctx 4096 8192 16384 32768 --modes compressed exact --gen 128 \
    --out paper/generated/active_modes_fresh.json >> "$LOG" 2>&1
echo "  modes done" | tee -a "$LOG"

echo "--- E5: residual-budget sweep @16k (0 8 16 32 64 128) ---" | tee -a "$LOG"
python paper/scripts/measure_residual_sweep.py \
    --ctx 16384 --residuals 0 8 16 32 64 128 --gen 128 \
    --out paper/generated/residual_sweep_fresh.json >> "$LOG" 2>&1
echo "  residual sweep done" | tee -a "$LOG"

echo "--- 64k reach: dense then active ---" | tee -a "$LOG"
for engine in dense active; do
  python paper/scripts/cell_worker.py --engine "$engine" --ctx 65536 --gen 128 \
      --prompt-file "$RES/prompt_65536.txt" --result-file "$RES/clean_${engine}_65536.json" >> "$LOG" 2>&1
  python - "$RES/clean_${engine}_65536.json" <<'PY' | tee -a "$LOG"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"  {d.get('engine')} 64k: pf={d.get('prefill_s'):.1f}s tps={d.get('decode_tps'):.1f} mx={d.get('mx_peak_gb'):.3f} needle={d.get('needle_found')} status={d.get('status')}")
except Exception as e: print("  parse-fail", e)
PY
done
echo "== ablations done $(date) ==" | tee -a "$LOG"
