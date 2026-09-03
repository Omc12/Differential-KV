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
      --model "$MODEL" --max-length "$MAXLEN" ${MINLEN:+--min-length "$MINLEN"}       --out "$out" "$@"
  echo "### RULER ARM $name exit=$?"
}

# ARMS is deliberately SHORTER than the LongBench campaign's nine. RULER runs
# 1,040 items per arm and the dense arm alone took ~2 h, so seven arms is
# 10-12 h at 32k and considerably worse at 64k -- for information the 12k
# LongBench table already carries. What is kept is what answers a question that
# is still open:
#
#   dense, dkv_mid, dkv_high   the core quality-at-length comparison
#   snapkv                     the one competitor that beat DKV at 12k
#
# Dropped at RULER lengths, with the reason:
#   streamingllm  characterised at 12k (-8.96), consistently the weakest
#                 eviction arm; nothing about length changes that reading
#   kivi2         catastrophic everywhere (-27.83 at 12k); a second
#                 confirmation is not worth 2 h
#   h2o           tracks snapkv closely at 12k (-0.78 vs +0.34) and shares
#                 its observation-window mechanism, so snapkv already
#                 represents attention-observation eviction here
#
# At 64k, RULER_ARMS should be dense+dkv only: the context ladder already
# measured that snapkv and streamingllm spill at 65,536 exactly where dense
# does, so running them there measures paging, not the method.
RULER_ARMS="${RULER_ARMS:-dense dkv_mid dkv_high snapkv}"

case " $RULER_ARMS " in *" dense "*)        run_arm dense    --arm dense --quant nf4 ;; esac
case " $RULER_ARMS " in *" dkv_mid "*)      run_arm dkv_mid  --arm dkv --quant nf4 --preset mid ;; esac
case " $RULER_ARMS " in *" dkv_high "*)     run_arm dkv_high --arm dkv --quant nf4 --preset high ;; esac
case " $RULER_ARMS " in *" snapkv "*)       run_arm snapkv   --arm snapkv --quant nf4 --baseline-params '{"budget": 2016, "window": 32}' ;; esac
case " $RULER_ARMS " in *" streamingllm "*) run_arm streamingllm --arm streamingllm --quant nf4 --baseline-params '{"n_sink": 4, "recency_window": 2044}' ;; esac
case " $RULER_ARMS " in *" kivi2 "*)        run_arm kivi2    --arm kivi2 --quant nf4 ;; esac
case " $RULER_ARMS " in *" h2o "*)          run_arm h2o      --arm h2o --quant nf4 --baseline-params '{"budget": 2048, "window": 32, "recency_window": 512}' ;; esac

echo ""
echo "############ RULER CAMPAIGN COMPLETE — scoring ############"
"$PY" benchmarks/run_ruler_cuda.py --score "$OUTDIR/${TAG}_*_max${MAXLEN}.jsonl"
