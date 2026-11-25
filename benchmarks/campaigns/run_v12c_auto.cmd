@echo off
rem V12 p4 -- the card-derived exact-history budget (DKV_STREAM_ELASTIC_POLICY
rem =auto, the default since 2026-11-25), checked before it ships.
rem V12 p2 (token window, 8192) lost 7.2 points to p1 (memory window) on the
rem shared 39 items and ran ~50% slower (141 vs 91 s/item). auto reads the
rem budget from the card ONCE at load and converts it per model from the live
rem blocks, so no setting is needed. Same items and environment as p1 with no
rem policy set: it should tie p1.
rem   g1  granite-4.2-8b, auto, one niah item at 114688 (the longest granite
rem       RULER length generated; 131072 has no items): reach (no spill) check.
rem   bs  block-size logit-fidelity sweep, Qwen3.5-4B nf4, 32k.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set DKV_ROUTE_TOPP_STATS=1
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set DKV_DBG_ELASTIC=0

if not "%1"=="" goto %1

:p4
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 3 --out %R%\ruler\V12_Qwen3.5-4B_dkv_mid_resid_ada_auto_131k.jsonl
echo V12 P4 COMPLETE
if not "%1"=="" goto :eof

:g1
"%PY%" benchmarks\run_ruler_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset mid --min-length 114688 --max-length 114688 --tasks niah_single_1 --per-task 1 --out %R%\ruler\V12_granite_dkv_mid_auto_115k_reach.jsonl
echo V12 G1 COMPLETE
if not "%1"=="" goto :eof

:bs
"%PY%" colab\logit_fidelity.py --model Qwen/Qwen3.5-4B --quant nf4 --ctx 32768 --arms dense_true baseline bs512 bs256 --json %R%\fidelity\V12_blocksize_Qwen3.5-4B_32k.json
echo V12 BS COMPLETE
