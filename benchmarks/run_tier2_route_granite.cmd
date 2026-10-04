@echo off
rem Tier 2, batch C on granite (all 40 layers attend): routing beyond the
rem exact limit, on the hybrid in streaming mode, top-16 vs all blocks.
rem 8 RULER prompts at 24k and 32k. The dense control pages to host memory at
rem these lengths (slow, but its logits are exact).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set H=DKV_KEY_QUANT=pc4,DKV_RESID_ATTN=1,DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb ^
  --ruler 24576/niah_multikey_2:0-0 24576/niah_multiquery:0-0 24576/qa_1:0-0 24576/niah_single_3:0-0 32768/niah_multikey_2:0-0 32768/niah_multiquery:0-0 32768/qa_1:0-0 32768/niah_single_3:0-0 ^
  --arms "hyb_top16:%H%" "hyb_all:%H%,DKV_TOPK_BLOCKS=0,DKV_BLOCKS_PER_CHUNK=256" "hyb_top32:%H%,DKV_TOPK_BLOCKS=32"
echo ROUTEG COMPLETE
