@echo off
rem THE ONE-HOUR SUBSET of run_midnight.cmd -- everything except the 5.5 h
rem DKV/high RULER run and the rungs already known to spill. Same output stores
rem as run_midnight.cmd, so running that later skips whatever finished here.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set ST={\"static\": true}
set DF={\"defrag\": true}
set DKV_TRITON_STRICT=1

rem 1. preallocated dense, 128-token answers
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 49152 65536 98304 131072 --baseline-params "%ST%" --out %R%\ladder\GEN128_Qwen3.5-4B_dense_static.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 16384 24576 32768 49152 --baseline-params "%ST%" --out %R%\ladder\GEN128_granite_dense_static.jsonl

rem 2. deciding rungs at 128-token answers
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 49152 65536 98304 --baseline-params "%DF%" --out %R%\ladder\GEN128_Qwen3.5-4B_dense_defrag.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 16384 24576 --baseline-params "%DF%" --out %R%\ladder\GEN128_granite_dense_defrag.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 98304 131072 --out %R%\ladder\GEN128_Qwen3.5-4B_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 16384 32768 --out %R%\ladder\GEN128_granite_dkv_auto.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

rem 3. sessions resident at 16k with the decode-memory fixes
set DKV_REMAT_CACHE=0
"%PY%" benchmarks\bench_concurrency_cuda.py --model %QW% --arms dkv --ctx 16384 --out %R%\concurrency\Qwen3.5-4B_ctx16384_dkv_noremat.jsonl
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\bench_concurrency_cuda.py --model %QW% --arms dkv --ctx 16384 --out %R%\concurrency\Qwen3.5-4B_ctx16384_dkv_streamprofile.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_REMAT_CACHE=

rem 4. timing: preallocated dense; streaming DKV forced at shared lengths
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dense --contexts 32768 65536 --baseline-params "%ST%" --out %R%\systems\MIDNIGHT_Qwen3.5-4B_dense_static.jsonl
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --contexts 32768 --out %R%\systems\MIDNIGHT_Qwen3.5-4B_dkv_forced_stream_32k.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --contexts 16384 --out %R%\systems\MIDNIGHT_granite_dkv_forced_stream_16k.jsonl
set DKV_STREAMING_COMPRESS=

rem 5. SnapKV on granite: ceiling and timing
"%PY%" benchmarks\context_ladder.py --model %GR% --arms snapkv --contexts 16384 24576 32768 --out %R%\ladder\granite-4.2-8b_mid_nf4_snapkv_aligned.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms snapkv --contexts 16384 24576 --out %R%\systems\MIDNIGHT_granite_snapkv_aligned.jsonl

echo HOUR CAMPAIGN COMPLETE
