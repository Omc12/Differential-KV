@echo off
rem Tier 2, batch A, enlarged: 24 prompts x 24 forced steps, paired against base.
rem Resumable: finished arms are reused on a rerun.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
"%PY%" benchmarks\decode_fidelity.py --gen 24 ^
  --lb qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4 ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "att:DKV_RESID_ATTN=1" "u4r64vg0:DKV_U_BITS=4,DKV_RANK=64,DKV_RSVD_MAX_RPROJ=64,DKV_V_SCALE=0" "win:DKV_U_BITS=4,DKV_RANK=64,DKV_RSVD_MAX_RPROJ=64,DKV_V_SCALE=0,DKV_RESID_ATTN=1"
echo TIER2 COMPLETE
