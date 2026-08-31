#!/bin/bash
set -e
# Is the srl_k_keep clamp's GROWTH term live, and tied to the generation budget?
#
# The clamp bounds K at min(n_comp_blocks + growth, n_slots), where growth is
# ceil(max_generate / micro_block_size) -- the blocks that can still be compressed
# while the answer is being generated, as the dense window flushes.
#
# This exists because the growth term was first implemented as the pool's
# headroom_slots, which LOOKS right (the pool reserves it for exactly this) but is
# capped at 512 tokens and does not bound how many blocks generation can create.
# That form yields growth=1 for every generation budget, so K would sit short
# mid-answer at any budget above one block.
#
# test_niah_native.sh cannot catch that: it sets DKV_MAX_TOKENS=40, where both
# forms give 1 and agree. This sweeps the budget instead, so the two forms give
# visibly different K:
#
#     DKV_MAX_TOKENS   correct (ceil/mbs)   headroom form
#     40               n_comp + 1           n_comp + 1     <- agree, useless
#     2048             n_comp + 2           n_comp + 1
#     8192             n_comp + 8           n_comp + 1
#
# Recall is asserted too: a K that grew must not have broken retrieval.

BINARY="${DKV_TEST_BINARY:-../build/dkv_native}"
MODEL="${DKV_TEST_MODEL:-../qwen2.5-1.5b-instruct-q8_0.gguf}"
if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    echo "set DKV_TEST_MODEL=/path/to/model.gguf" >&2
    exit 2
fi

MBS="${DKV_MICRO_BLOCK_SIZE:-1024}"
NEEDLE="The secret passcode is OMEGA-7741-DELTA."
QUESTION="What is the secret passcode? Repeat it exactly."

export DKV_ENGAGE_THRESHOLD=1024 DKV_NATIVE_ATTN=0 DKV_FORCE_CPU_ATTN=0
export DKV_MPS_APPROXIMATE_ATTN=1 DKV_DENSE_DIRECT=1 DKV_POOL_ABS_ROT=1
export DKV_TEMPERATURE=0 DKV_DISABLE_VSL=1 DKV_ENABLE_FACTUAL=0
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1

source ../../dkv_venv/bin/activate 2>/dev/null || true
prompt=$(python3 make_niah_prompt.py 4000 0.5 "$NEEDLE" "$QUESTION")

fail=0
printf "%-10s %-10s %-10s %-7s %s\n" MAX_TOKENS predicted observed match recall
for mt in 40 2048 8192; do
    export DKV_MAX_TOKENS=$mt
    out=$("$BINARY" "$MODEL" "$prompt" 2>k_growth_stderr.log) || true
    line=$(grep -o "lowered from [0-9]* → [0-9]* ([0-9]* compressed blocks" k_growth_stderr.log | tail -1)
    if [ -z "$line" ]; then
        echo "  clamp never fired at DKV_MAX_TOKENS=$mt -- K was not bounded"; fail=1; continue
    fi
    k=$(echo "$line" | awk '{print $5}')
    nc=$(echo "$line" | grep -o "([0-9]*" | tr -d '(')
    growth=$(( (mt + MBS - 1) / MBS ))
    pred=$(( nc + growth ))
    recall=$(echo "$out" | grep -qi "OMEGA-7741-DELTA" && echo PASS || echo FAIL)
    match=$([ "$k" = "$pred" ] && echo yes || echo NO)
    [ "$match" = "yes" ] || fail=1
    [ "$recall" = "PASS" ] || fail=1
    printf "%-10s %-10s %-10s %-7s %s\n" "$mt" "$pred" "$k" "$match" "$recall"
done

if [ "$fail" -eq 0 ]; then
    echo "growth term is live and scales with the generation budget: OK"
else
    echo "growth term WRONG -- K does not track ceil(max_generate/${MBS})" >&2
fi
exit $fail
