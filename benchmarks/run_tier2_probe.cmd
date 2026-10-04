@echo off
rem Tier 2 probe: which side limits real decode? One side keeps every row as an
rem 8-bit residual (near exact), the other its usual 128 top rows. Same 24
rem prompts as run_tier2_batchA.cmd, so the dense control and base are reused.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set P=DKV_MAX_RESIDUAL_TOKENS=1024,DKV_RESIDUAL_TIER_Q=1,DKV_PROBE_OTHER_RES=128
"%PY%" benchmarks\decode_fidelity.py --gen 24 ^
  --lb qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4 ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "probeK:%P%,DKV_PROBE_EXACT_SIDE=K" "probeV:%P%,DKV_PROBE_EXACT_SIDE=V"
echo PROBE COMPLETE
