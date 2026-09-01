#!/usr/bin/env bash
# Generate the official RULER suite for one model, at lengths that FIT the card.
#
# CONTEXT LENGTHS ARE CHOSEN FROM THE MEASURED LADDER, NOT FROM RULER'S DEFAULTS.
#   RULER's standard rungs run to 128k. On a 12 GB card most of those spill into
#   host memory, and a spilled run reports PCIe bandwidth as latency. Each model
#   is therefore generated only up to the largest length its ladder cleared
#   without spilling (paper/results/ladder/):
#
#     granite-4.2-8b   clean to 16,384   -> 4k, 8k, 16k
#     Qwen3.5-4B       dense clean to 49,152, DKV to 98,304
#                      -> 4k, 8k, 16k, 32k for the dense-vs-DKV comparison,
#                         plus 64k which only the DKV arm can run
#
# WHY A SEPARATE, SPACE-FREE WORK DIRECTORY
#   RULER's prepare.py builds its subprocess command by string concatenation
#   without quoting, so any space in the path splits the argument and the child
#   dies with "can't open file 'C:\Users\USER\Desktop\Differential'". It then
#   prints "Prepare <task> with lines: N" ANYWAY, so a failed generation looks
#   exactly like a successful one and you get an empty directory that a later
#   run reads as zero samples. The repo lives under "Differential KV", so
#   generation happens in a copy at a path with no spaces and the output is
#   copied back.
#
#   `python` must also resolve to the real interpreter: bare `python` on this
#   box hits the Windows Store alias stub, which is the other way this silently
#   produces nothing.

set -u

WORK="${WORK:-C:/Users/USER/AppData/Local/Temp/ruler_work}"
REPO="C:/Users/USER/Desktop/Differential KV"
PY="C:/Users/USER/AppData/Local/Programs/Python/Python313/python.exe"
export PATH="/c/Users/USER/AppData/Local/Programs/Python/Python313:$PATH"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODEL="${MODEL:-ibm-granite/granite-4.2-8b}"
TAG="$(basename "$MODEL")"
LENGTHS="${LENGTHS:-4096 8192 16384}"
NSAMP="${NSAMP:-20}"
OUT="${OUT:-$WORK/generated/$TAG}"

TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"

mkdir -p "$OUT"
cd "$WORK/data" || exit 1

for L in $LENGTHS; do
  for T in $TASKS; do
    dest="$OUT/$L"
    if [ -s "$dest/$T/validation.jsonl" ]; then
      echo "  skip $TAG $T @ $L (already generated)"
      continue
    fi
    echo "  generating $TAG $T @ $L ..."
    "$PY" prepare.py \
        --save_dir "$dest" \
        --benchmark synthetic \
        --task "$T" \
        --tokenizer_path "$MODEL" \
        --tokenizer_type hf \
        --max_seq_length "$L" \
        --num_samples "$NSAMP" \
        --model_template_type base \
        > /dev/null 2>&1
    n=$(wc -l < "$dest/$T/validation.jsonl" 2>/dev/null || echo 0)
    if [ "$n" -lt 1 ]; then
      echo "    !! $T @ $L produced NO samples -- generation failed"
    else
      echo "    $T @ $L: $n samples"
    fi
  done
done

echo "copying generated data back into the repo ..."
mkdir -p "$REPO/paper/results/ruler_data/$TAG"
cp -r "$OUT/." "$REPO/paper/results/ruler_data/$TAG/"
echo "done: $REPO/paper/results/ruler_data/$TAG"
