V3 part A streaming ladders measured 2026-10-17 before the incremental-prefill fix:
the incremental prefill branch attended raw history eagerly (whole [H, Q, raw]
score matrix). With the elastic window holding ~113k raw tokens on Qwen3.5-4B
this spilled at 131k (18.58 GB). Superseded by the re-run after the fix; outputs
(quality) were unaffected, only memory and time.
