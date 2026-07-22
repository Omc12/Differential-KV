#!/bin/bash

# Optimized configuration for dkv_native with long prompts
# For ~20k token contexts like Pride and Prejudice Wikipedia article

echo "🚀 Starting dkv_native with optimized settings for long contexts"
echo ""

# 1. Set context budget to actual needs (prompt + generation)
export DKV_MAX_CTX_TK=24576  # 24k tokens (20k prompt + 4k generation)
echo "✓ Context budget: 24576 tokens"

# 2. Use mid preset for memory efficiency
export DKV_PRESET=mid
export DKV_PREFILL_CHUNK_SIZE=512
echo "✓ Preset: mid (8k token pool limit will be overridden by MAX_CTX_TK)"

# 3. Conservative GPU budget
export DKV_GPU_BUDGET_GB=1.5
echo "✓ GPU budget: 1.5GB"

# 4. Micro block size 16 (default, but explicit)
export DKV_MICRO_BLOCK_SIZE=16
echo "✓ Micro block size: 16 tokens"

# 5. Rank 16 (default, but explicit)
export DKV_RANK=16
echo "✓ SVD rank: 16"

# 6. Keep approximate attention on macOS
export DKV_MPS_APPROXIMATE_ATTN=1
echo "✓ MPS approximate attention: enabled"

# 7. Generation limit
export DKV_MAX_TOKENS=512
echo "✓ Max generation tokens: 512"

echo ""
echo "📊 Expected memory usage: ~2.0-2.5GB (down from 3.8GB)"
echo "📊 Expected TPS: 0.8-1.2 tok/s on 20k context (up from 0.3-0.4)"
echo ""

# Determine model path
if [ -f "dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    MODEL="dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"
elif [ -f "dkv_native/qwen2.5-1.5b-instruct-q8_0.gguf" ]; then
    MODEL="dkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
elif [ -f "dkv_native/qwen2.5-0.5b-instruct.gguf" ]; then
    MODEL="dkv_native/qwen2.5-0.5b-instruct.gguf"
else
    echo "❌ Model file not found in dkv_native/"
    echo "Looking for: qwen2.5-1.5b-instruct-q4_k_m.gguf, qwen2.5-1.5b-instruct-q8_0.gguf or qwen2.5-0.5b-instruct.gguf"
    exit 1
fi

echo "📦 Using model: $MODEL"
echo ""

# Check if binary exists
if [ ! -f "dkv_native/build/dkv_native" ]; then
    echo "❌ Binary not found at dkv_native/build/dkv_native"
    echo "Please build first:"
    echo "  cd dkv_native"
    echo "  cmake -B build && cmake --build build"
    exit 1
fi

echo "▶️  Launching CLI..."
echo ""

cd dkv_native
../dkv_venv/bin/python3 serving/cli.py \
    --model "../$MODEL" \
    --binary-path build/dkv_native \
    --preset mid \
    --max-tokens 512 \
    --temperature 0.7 \
    --top-p 0.9 \
    --repetition-penalty 1.15

echo ""
echo "✅ Session ended"
