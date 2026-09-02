#!/usr/bin/env bash
# RULER campaign: every arm, one model, only at lengths the card can hold.
#
# --max-length is the model's MEASURED non-spilling ceiling from
# paper/results/ladder/, not a RULER default. Generated data above that ceiling
# is skipped rather than run: a spilled run's quality numbers are still real,
# but its latency is PCIe bandwidth, and the two cannot share a table.
#
#   granite-4.2-8b   clean to 16,384
#   Qwen3.5-4B       dense clean to 49,152  -> 32,768 is the largest generated
#                    rung both arms can run; DKV additionally reaches 65,536,
#                    which is run as a DKV-only row.
#
# Eviction baselines all get the SAME ~2048-token KV budget, so they are
# compared on equal terms rather than at whatever budget each paper used.
#
# Resumable per item; rerun to continue after an interruption.

set -u
cd "$(dirname "$0")/.." || exit 1

PY="C:/Users/USER/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODEL="${MODEL:-ibm-granite/granite-4.2-8b}"
MAXLEN="${MAXLEN:-16384}"
TAG="$(basename "$MODEL")"
OUTDIR="paper/results/ruler"
mkdir -p "$OUTDIR"

run_arm () {
  local name="$1"; shift
  local out="$OUTDIR/${TAG}_${name}_max${MAXLEN}.jsonl"
  echo ""
  echo "############################################################"
  echo "### RULER ARM: $name   model=$TAG  max_length=$MAXLEN"
  echo "############################################################"
  "$PY" benchmarks/run_ruler_cuda.py \
      --model "$MODEL" --max-length "$MAXLEN" --out "$out" "$@"
  echo "### RULER ARM $name exit=$?"
}

run_arm dense        --arm dense  --quant nf4
run_arm dkv_mid      --arm dkv    --quant nf4 --preset mid
run_arm snapkv       --arm snapkv --quant nf4 --baseline-params '{"budget": 2016, "window": 32}'
run_arm streamingllm --arm streamingllm --quant nf4 --baseline-params '{"n_sink": 4, "recency_window": 2044}'
run_arm dkv_high     --arm dkv    --quant nf4 --preset high
run_arm kivi2        --arm kivi2  --quant nf4
run_arm h2o          --arm h2o    --quant nf4 --baseline-params '{"budget": 2048, "window": 32, "recency_window": 512}'

echo ""
echo "############ RULER CAMPAIGN COMPLETE — scoring ############"
"$PY" benchmarks/run_ruler_cuda.py --score "$OUTDIR/${TAG}_*_max${MAXLEN}.jsonl"
