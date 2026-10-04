@echo off
rem Tier 2: hybrid store with the trims (value-only factor, key residuals from
rem the codes) + attention-ranked residuals. Same prompts; controls reused.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "fast_att:DKV_KEY_QUANT=pc4,DKV_RESID_ATTN=1" "fast_att_stream:DKV_KEY_QUANT=pc4,DKV_RESID_ATTN=1,DKV_STREAMING_COMPRESS=1" "stream:DKV_STREAMING_COMPRESS=1"
echo FAST COMPLETE
