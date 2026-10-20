@echo off
rem V4-SHORT -- the review's follow-ups that fit in ~3 GPU hours, in priority
rem order. The full V4 re-run (every RULER store on the fixed prompt, ~4 days)
rem did not fit. Instead, new RULER arms use the SAME prompt as the existing
rem stores (RULER_ANSWER_PREFIX=0) so they pair item-for-item with them; the
rem paper excludes vt (zeroed for every arm by the missing prefix) and checks,
rem with the prefix on, that DKV's gap to dense does not move (step 3).
rem Resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set Q32=--min-length 32768 --max-length 32768 --per-task 5

rem -- 1. routing ablation: DKV attending every block at 32k (pairs with dkv_mid 32k)
setlocal
set RULER_ANSWER_PREFIX=0
set DKV_STREAMING_COMPRESS=0
set DKV_TOPK_BLOCKS=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q32% --out %R%\ruler\V4S_Qwen3.5-4B_dkv_mid_attendall_32k.jsonl
endlocal

rem -- 2. tiled KIVI-2, quality at 32k (pairs with dkv_mid and kivi4_tiled 32k)
setlocal
set RULER_ANSWER_PREFIX=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi2_tiled %Q32% --out %R%\ruler\V4S_Qwen3.5-4B_kivi2_tiled_32k.jsonl
endlocal

rem -- 3. prompt check: dense and DKV/mid at 32k WITH the answer prefix
setlocal
set RULER_ANSWER_PREFIX=1
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense %Q32% --out %R%\ruler\V4S_Qwen3.5-4B_dense_prefix_32k.jsonl
set DKV_STREAMING_COMPRESS=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q32% --out %R%\ruler\V4S_Qwen3.5-4B_dkv_mid_prefix_32k.jsonl
endlocal

rem -- 4. exact KV offloaded to host memory: reach and time at the long rungs
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 131072 262144 --baseline-params "{\"offload\": true}" --timeout 3600 --out %R%\ladder\V4_Qwen3.5-4B_dense_offload.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 32768 65536 --baseline-params "{\"offload\": true}" --timeout 3600 --out %R%\ladder\V4_granite_dense_offload.jsonl

rem -- 5. tiled KIVI-2 reach
"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi2_tiled --gen 128 --contexts 98304 131072 --timeout 3600 --out %R%\ladder\V4_granite_kivi2_tiled.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms kivi2_tiled --gen 128 --contexts 262144 --timeout 3600 --out %R%\ladder\V4_Qwen3.5-4B_kivi2_tiled.jsonl

rem -- 6. eviction at DKV's own bytes at 131k (23,808 tokens; pairs with the 131k stores)
setlocal
set RULER_ANSWER_PREFIX=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked --min-length 131072 --max-length 131072 --per-task 5 --baseline-params "{\"n_sink\": 4, \"recency_window\": 23804}" --out %R%\ruler\V4S_Qwen3.5-4B_sl_chunked_eqbytes_131k.jsonl
endlocal

rem -- 7. SnapKV with the allocator fix dense gets
"%PY%" benchmarks\context_ladder.py --model %QW% --arms snapkv --gen 128 --contexts 65536 98304 122880 --baseline-params "{\"budget\": 2016, \"window\": 32, \"defrag\": true}" --timeout 3600 --out %R%\ladder\V4_Qwen3.5-4B_snapkv_defrag.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms snapkv --gen 128 --contexts 20480 24576 --baseline-params "{\"budget\": 2016, \"window\": 32, \"defrag\": true}" --timeout 3600 --out %R%\ladder\V4_granite_snapkv_defrag.jsonl

echo V4-SHORT COMPLETE
