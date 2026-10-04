@echo off
rem Tier 2: hybrid design parameters (value residuals, value rank) and the
rem streaming baseline on the same 24 prompts. Reuses dense control and base.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set H=DKV_KEY_QUANT=pc4,DKV_MAX_RESIDUAL_TOKENS=1024,DKV_RESIDUAL_QUANT=none,DKV_RESIDUAL_TIER_Q=1
"%PY%" benchmarks\decode_fidelity.py --gen 24 ^
  --lb qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4 ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "stream:DKV_STREAMING_COMPRESS=1" "hyb4_v0:%H%,DKV_KEY_QUANT_VRES=0" "hyb4_r16:%H%,DKV_KEY_QUANT_VRES=64,DKV_RSVD_MAX_RPROJ=16" "hyb4_r16_v0:%H%,DKV_KEY_QUANT_VRES=0,DKV_RSVD_MAX_RPROJ=16"
echo HYBRID2 COMPLETE
