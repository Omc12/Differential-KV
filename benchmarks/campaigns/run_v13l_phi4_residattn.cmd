@echo off
rem V13l -- Phi-4-mini: decode-step per-layer diff localised the gap to layer 9,
rem which is clean with exact history (8.9%) and explodes with the lowrank store
rem (104%); rope scaling forced to 1 did not change it. Attention-sink signature.
rem The lowrank store picks residual rows by reconstruction error only;
rem DKV_RESID_ATTN=1 ranks them by error x attention received (on by default only
rem for the hybrid/codes stores, and the codes store is clean on Phi-4: 0.0017).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set DKV_RESID_ATTN=1
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true baseline
echo V13L COMPLETE
