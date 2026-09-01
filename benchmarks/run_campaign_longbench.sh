#!/usr/bin/env bash
# LongBench campaign: every arm, one model, sized to fit the card.
#
# WHY A FIXED --max-length FOR EVERY ARM
#   LongBench middle-truncates anything longer, so the context budget IS part of
#   the task definition: an arm given more context is answering an easier
#   question. Every arm here gets the same budget, and the budget is chosen from
#   the measured context ladder so that no arm spills to host memory. A spilled
#   arm's latency is PCIe bandwidth and its quality is still valid, but mixing
#   spilled and unspilled timings in one table is not a comparison.
#
#   granite-4.2-8b measured clean to 16,384 tokens (peak 11.30 GB on a 12.28 GB
#   card, DKV arm). 12,000 leaves room for the generation on top of the prompt.
#
# WHY THESE BASELINE BUDGETS
#   The eviction methods are all given the SAME ~2048-token KV budget so they
#   are compared to each other on equal terms rather than at whatever budget
#   each one's paper happened to use. The realized compression ratio is recorded
#   per item, so the memory axis is measured rather than assumed.
#
# Resumable: every arm checkpoints per item and skips what is already on disk,
# so re-running this script after a power cut continues where it stopped.

set -u
cd "$(dirname "$0")/.." || exit 1

PY="C:/Users/USER/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODEL="${MODEL:-ibm-granite/granite-4.2-8b}"
MAXLEN="${MAXLEN:-12000}"
N="${N:-20}"
TASKS="${TASKS:-qasper narrativeqa hotpotqa multifieldqa_en gov_report passage_retrieval_en}"
TAG="$(basename "$MODEL")"
OUTDIR="paper/results/longbench"
mkdir -p "$OUTDIR"

run_arm () {                     # name, extra args...
  local name="$1"; shift
  local out="$OUTDIR/${TAG}_${name}_len${MAXLEN}.jsonl"
  echo ""
  echo "############################################################"
  echo "### ARM: $name   model=$TAG  maxlen=$MAXLEN  n=$N"
  echo "############################################################"
  "$PY" benchmarks/run_longbench_cuda.py \
      --model "$MODEL" --max-length "$MAXLEN" --num-samples "$N" \
      --datasets $TASKS --out "$out" "$@"
  echo "### ARM $name exit=$?"
}

# Ordered so the load-bearing comparison exists early: if the run is cut short,
# dense + DKV + the strongest eviction baseline are already on disk.
run_arm dense          --arm dense  --quant nf4
run_arm dkv_mid        --arm dkv    --quant nf4 --preset mid
run_arm snapkv         --arm snapkv --quant nf4 --baseline-params '{"budget": 2016, "window": 32}'
run_arm streamingllm   --arm streamingllm --quant nf4 --baseline-params '{"n_sink": 4, "recency_window": 2044}'
run_arm dkv_high       --arm dkv    --quant nf4 --preset high
run_arm kivi2          --arm kivi2  --quant nf4
run_arm h2o            --arm h2o    --quant nf4 --baseline-params '{"budget": 2048, "window": 32, "recency_window": 512}'
run_arm kivi4          --arm kivi4  --quant nf4
run_arm int8_kv        --arm int8_kv --quant nf4

echo ""
echo "############ CAMPAIGN COMPLETE — scoring everything ############"
"$PY" benchmarks/run_longbench_cuda.py --score "$OUTDIR/${TAG}_*_len${MAXLEN}.jsonl"
