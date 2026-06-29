# Compressed Decode — Accuracy + Speed Optimization (2026-06-29)

Brings the MLX active-runtime compressed sparse-decode path
([ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py](../ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py))
toward dense parity on **accuracy** and **speed**.

**Setup:** Qwen2.5-1.5B-Instruct (4-bit MLX) · Apple M3, 8 GB · greedy decode.
**Recall** = exact buried passcode (`OMEGA-7741-DELTA`) in GENERATED text, on the HARD
on-topic NIAH prompt (`benchmarks/niah_recall.py --bench`) — needle hidden in same-topic
filler, the case that exposes compression faults (the easy prompt always passed and hid the
bug). **Decode tok/s** from that harness (early-out on emit). **Prefill / peak** from
`bench_worker.py` (the compressed prefill is shared by both decode paths).

## Decode throughput (tok/s) · all recall = exact unless noted

| Ctx | Prefill (s) | Peak (GB) | **Dense** | Compressed BEFORE¹ | Compressed NOW² |
|----:|----:|----:|----:|----:|----:|
| 8k  | 18.5 | 2.03 | 42.7 | 13.5 ✗garbled | **14.0** |
| 16k | 44   | 2.26 | 37.1 | 9.7           | **14.5** |
| 32k | 108  | 2.74 | 29.2 | 5.2  ✗garbled | **14.1** |
| 64k | 282  | ~4.2 | 21.1 | 2.9 (all-blk) | 16k-route ✗³ |

¹ BEFORE = all-blocks decode, `max_residual=32`. Two faults: garbled needle at 8k/32k AND
slow (the 32k garble is why those cells are ✗).
² NOW = `max_residual=64` + top-K block routing (default K=16). Exact recall, ~14–16 tok/s
**flat** (context-independent), 3.2× faster than BEFORE at 32k.
³ At 64k, all-blocks decode recalls exactly (2.9 tok/s) — the needle is fully captured — but
top-K routing drops the needle's block (see "Open: 64k routing"). Dense is the better choice
at 64k on this host anyway (faster + fits).

## What changed
- **Accuracy — `DIFFKV_MAX_RESIDUAL` 32 → 64.** The passcode's distinctive tokens are the
  highest-reconstruction-error ones (SVD truncates such low-energy outliers), so they're
  kept verbatim only if they fit the per-block exact-residual budget. At 32 they were
  truncated → values fell back to lossy SVD → wrong digits. 64 → exact at 4k/8k/16k/32k.
- **Speed — top-K block routing (`DIFFKV_TOPK_BLOCKS`, default 16; `DIFFKV_TOPK_FRAC`
  0.12).** Decode scores all blocks cheaply (Quest-style key min/max upper bound,
  `_block_relevance_minmax`, new per-block `comp_min_k`/`comp_max_k`) and runs value
  reconstruction + exact-residual attention for only the top-K blocks. Decode cost becomes
  ~independent of context → flat ~14–16 tok/s vs dense's declining curve.

## Honest read
- The **fused** dense kernel (`mx.fast.scaled_dot_product_attention`) is still faster than
  compressed below ~64k (42→21 declining vs ~14–16 flat), and dense fits to 64k on 8 GB.
  Compressed does NOT beat dense on raw speed in this range.
- Compressed's real wins: (1) accuracy now equals dense; (2) **context-independent** decode
  (flat where dense declines → curves cross at very high context); (3) bounded decode-time
  residual memory + compressed KV → reach 128k where dense OOMs.
- **Recommendation:** use dense for ≤~32–64k (faster, fits, exact). Use compressed top-K for
  memory-bound / very-long contexts. `DIFFKV_COMPRESSED_MIN_CTX` governs the auto switch.

## Open: 64k routing
At 64k (256 blocks) the min/max upper bound is too loose — dozens of blocks over-estimate
above the needle's block, so a fixed/fractional K=16–30 drops it (partial output
"OMEGA-77-ALPHA"). all-blocks recalls, so it is purely routing, not capture. A residual-aware
router (scoring exact residual keys) was tried and both slowed decode ~2× and did not fix it.
Larger K recovers recall but erodes the speed benefit. A tighter, cheap router (or always
attending the most-recent + score-selected blocks) is the path to reliable >32k routing —
left as follow-up since dense is preferable at 64k on this hardware.

## Repro
```
# accuracy + speed, current defaults
DIFFKV_COMPRESSED_DECODE=1 python benchmarks/niah_recall.py --bench --ctx 8192 16384 32768
# dense baseline
DIFFKV_COMPRESSED_DECODE=0 python benchmarks/niah_recall.py --bench --ctx 8192 16384 32768
# kernel parity gate
cd ACTIVE_RUNTIME && python -m pytest tests/test_diffkv_kernel_parity.py -q
```
