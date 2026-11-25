@echo off
rem V13d -- Phi-4-mini second-token bisect. V13 panel: first token clean (KL 0.008)
rem but the second, the first COMPRESSED-decode step, KL 0.679, top-5 overlap 1.0/5.
rem Phi-4-mini is the panel's only fused-qkv_proj, partial-rotary-0.75, LongRoPE model.
rem   dense_arm  compressed decode off (prefill side only)
rem   allblocks  every block attended (routing out of the picture)
rem   noremat    decode without the remat cache
rem   unrot      unrotated pool (keys re-rotated at read)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true baseline dense_arm allblocks noremat unrot
echo V13D COMPLETE
