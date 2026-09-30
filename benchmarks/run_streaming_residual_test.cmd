@echo off
rem Does a larger exact-residual budget stop the compounding error of streaming
rem prefill? narrativeqa + hotpotqa (40 items, where streaming mid lost the most)
rem at 256 and 512 exact tokens per block, then the memory each costs.
rem Opt-in Track A settings; separate stores; resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_STREAMING_COMPRESS=1
set DKV_PREFILL_LOWMEM=1
set DKV_REMAT_CACHE=0
set DKV_TRITON_STRICT=1

rem Quality: preset high (rank 32 on granite, 256 exact tokens per block)
"%PY%" benchmarks\run_longbench_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset high --max-length 12000 --num-samples 20 --datasets narrativeqa hotpotqa --out %R%\longbench\RESID_granite_dkv_high_streamlow.jsonl
rem Quality: 512 exact tokens per block
set DKV_MAX_RESIDUAL_TOKENS=512
"%PY%" benchmarks\run_longbench_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset high --max-length 12000 --num-samples 20 --datasets narrativeqa hotpotqa --out %R%\longbench\RESID_granite_dkv_high_resid512_streamlow.jsonl
rem Quality: 512 exact tokens per block stored int4 -- roughly the bytes of 256
rem at int8. Coverage vs precision: int4 blurs digits (CUDA NIAH lost 6/96 at
rem EQUAL count, 2026-08-31) but doubles how much earlier text stays exact.
set DKV_RESIDUAL_QUANT=int4
"%PY%" benchmarks\run_longbench_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset high --max-length 12000 --num-samples 20 --datasets narrativeqa hotpotqa --out %R%\longbench\RESID_granite_dkv_high_resid512_int4_streamlow.jsonl
set DKV_RESIDUAL_QUANT=
set DKV_MAX_RESIDUAL_TOKENS=

rem Memory: what each budget costs on the ladder
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --preset high --contexts 16384 24576 32768 49152 --timeout 3600 --out %R%\ladder\RESID_granite_high_streamlow.jsonl
set DKV_MAX_RESIDUAL_TOKENS=512
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --preset high --contexts 16384 24576 32768 49152 --timeout 3600 --out %R%\ladder\RESID_granite_high_resid512_streamlow.jsonl
set DKV_RESIDUAL_QUANT=int4
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --preset high --contexts 16384 24576 32768 49152 --timeout 3600 --out %R%\ladder\RESID_granite_high_resid512_int4_streamlow.jsonl
set DKV_RESIDUAL_QUANT=
set DKV_MAX_RESIDUAL_TOKENS=

echo RESIDUAL TEST COMPLETE
