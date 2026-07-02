#!/bin/bash
set -e

echo "================================================================================"
echo "    DIFFERENTIAL KV: NATIVE C++ CLI 1.5B Q4 BENCHMARK (4K - 32K)"
echo "================================================================================"

export DIFFKV_MPS_APPROXIMATE_ATTN=0  # Exact attention on MPS/Metal!
export DIFFKV_NATIVE_ATTN=1
export DIFFKV_MAX_TOKENS=32
export DIFFKV_GPU_BUDGET_GB=1.5       # Budget GPU VRAM to prevent allocator hangs!
export DIFFKV_VERBOSE=0

# Define context lengths
contexts=(4096 8192 16384 32768)

# Log file headers
echo "Context | TTFT (ms) | Speed (tok/s) | Duration (s)" > summary.txt
echo "--------------------------------------------------" >> summary.txt

for ctx in "${contexts[@]}"
do
    echo "--------------------------------------------------------------------------------"
    echo "Running Context Length: $ctx tokens..."
    
    # Generate a prompt of approximately $ctx tokens (each word is roughly 1.3 tokens)
    num_words=$((ctx * 3 / 4))
    python3 -c "print('The quick brown fox jumps over the lazy dog. ' * ($num_words // 9))" > temp_prompt.txt
    
    # Run using the DiffKV CLI and capture output
    cd diffkv_native
    cat ../temp_prompt.txt | ../diffkv_venv/bin/python3 serving/cli.py \
        --model 1.5b \
        --preset high \
        --context $ctx \
        --max-tokens 32 > ../run_native_$ctx.log 2>&1 || true
    cd ..
    
    # Extract metrics from log file
    ttft=$(grep -oE "TTFT: [0-9]+\.[0-9]+ms" run_native_$ctx.log | head -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    speed=$(grep -oE "Speed: [0-9]+\.[0-9]+ tok/s" run_native_$ctx.log | head -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    duration=$(grep -oE "Duration: [0-9]+\.[0-9]+s" run_native_$ctx.log | head -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    
    echo "$ctx | $ttft ms | $speed tok/s | $duration s" >> summary.txt
    echo "Done. TTFT: $ttft ms | Speed: $speed tok/s | Duration: $duration s"
    
    # Clean up intermediate logs
    rm -f run_native_$ctx.log
done

# Clean up temp prompt
rm -f temp_prompt.txt

echo ""
echo "================================================================================"
echo "                            BENCHMARK SUMMARY TABLE"
echo "================================================================================"
cat summary.txt
echo "================================================================================"

rm -f summary.txt
