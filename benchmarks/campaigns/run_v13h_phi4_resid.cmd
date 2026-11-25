@echo off
rem V13h -- Phi-4-mini lowrank decode defect. V13g: codes store clean (0.0017),
rem lowrank 0.679, and more residual rows made it worse (1.165). resid0 = no
rem residual rows; DKV_DBG_RESIDUAL_ERRORS surfaces swallowed capture failures.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set DKV_DBG_RESIDUAL_ERRORS=1
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true baseline resid0
echo V13H COMPLETE
