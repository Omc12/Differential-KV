@echo off
rem BISECT THE CEILINGS between the ladder's rungs, 128-token answers. The
rem ladder steps 98k -> 131k on Qwen3.5-4B and 16k -> 24k -> 32k -> 49k on
rem granite; these fill the gaps at 8k (Qwen) and 4k (granite) spacing, so
rem the paper can state each ceiling to within one step. Each ladder stops at
rem its first spill. Resumable like every other campaign.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ladder
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set ST={\"static\": true}
set DKV_TRITON_STRICT=1

"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 106496 114688 122880 --baseline-params "%ST%" --out %R%\BISECT_Qwen3.5-4B_dense_static.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 20480 --baseline-params "%ST%" --out %R%\BISECT_granite_dense_static.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 139264 147456 155648 --out %R%\BISECT_Qwen3.5-4B_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 36864 40960 45056 --out %R%\BISECT_granite_dkv_auto.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

echo BISECT COMPLETE
