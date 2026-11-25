@echo off
rem V13g -- Phi-4-mini decode defect, two splits.
rem   a) ctx 3500: below LongRoPE's 4096-token switch (original_max_position_embeddings)
rem   b) 32k with the codes store instead of lowrank
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
echo === V13G a: ctx 3500 ===
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 3500 --arms dense_true baseline dense_arm
echo === V13G b: 32k codes store ===
set DKV_STORE=codes
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true baseline
echo V13G COMPLETE
