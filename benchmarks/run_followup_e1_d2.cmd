@echo off
rem 1) D2 v3 (reserved-aware elastic gate) at 65k, granite hybrid streaming.
rem 2) E1 (DKV_PREFILL_SDPA=1) on the SHIPPED streaming store: tier 2 vs plain
rem    streaming, and reach at 65k / 131k (shipped tiled streaming: 131k at 11.00 GB).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set GR=ibm-granite/granite-4.2-8b
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "stream:DKV_STREAMING_COMPRESS=1" "sdpa_base_stream:DKV_STREAMING_COMPRESS=1,DKV_PREFILL_SDPA=1"
setlocal
set DKV_KEY_QUANT=pc4
set DKV_RESID_ATTN=1
set DKV_STREAMING_COMPRESS=1
set DKV_STREAM_ELASTIC=1
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 65536 --out paper\results\diag\TEST_granite_hybrid_elastic3_65k.jsonl
endlocal
setlocal
rem same streaming setting as the shipped ceiling ladder (TILED_granite_dkv_auto)
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
set DKV_PREFILL_SDPA=1
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 65536 131072 --out paper\results\diag\TEST_granite_sdpa_stream_ladder.jsonl
endlocal
echo FOLLOWUP COMPLETE
