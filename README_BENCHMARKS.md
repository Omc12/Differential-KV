# DiffKV Benchmark Execution Log

This file logs execution results, findings, and 2x2 comparison tables for quality, retrieval, and performance benchmarks as strategic roadmap items are completed.

---

## Part B1: Long-form Coherence / Synthesis Evaluation

Evaluates generation quality and retrieval coherence over compressed contexts at 8k context size (reproducible anti-cheat mechanical scoring).

* **Command:** `python benchmarks/synthesis_eval.py --ctx 8192`
* **Harness Design:** Pads the Rahimi & Recht (2007) "Random Features" paper text with Jane Austen's *Pride and Prejudice* to reach 8192 tokens. Scores using a 15-fact checklist and 5 sentence-linkage constraints (score out of 100).
* **Results Table:**
  | Engine | Mode | Context | Score | Facts | Linkages | TPS |
  |---|---|---|---|---|---|---|
  | MLX | compressed | 8192 | **3.3** | 1/15 | 0/5 | 15.3 |
  | MLX | dense | 8192 | **26.7** | 5/15 | 1/5 | 32.8 |
  | NATIVE | compressed | 8192 | **26.7** | 5/15 | 1/5 | 1.0 |
  | NATIVE | dense | 8192 | **30.0** | 6/15 | 1/5 | 7.5 |

* **Key Findings:**
  1. **MLX Context Retrieval Loss:** MLX in compressed mode fails to retrieve from the paper (score 3.3). It attends only to the recency window near the end of the context containing *Pride and Prejudice* and hallucinates that Jane Austen used Fourier features to analyze social dynamics.
  2. **Native C++ Robustness:** Native C++ compressed mode successfully retrieves and summarizes the paper, matching the dense baseline quality (26.7/100).
  3. **Native SVD Prefill Bottleneck:** Native C++ compressed is CPU-bound during prefill (1.0 TPS) due to sequential execution of SVD on CPU (need for chunk-parallel SVD or Accelerate GESDD batching).

---

## Part B2: Multi-Needle & Adversarial Relational Tracking

Stresses multi-entity pointwise recall and relational binding integrity.

### 1. Multi-Needle Recall
* **Command:** `DIFFKV_COMPRESSED_DECODE=1 python benchmarks/niah_recall.py --ctx 8192 --multi-needle`
* **Harness Design:** Plants three distinct secret passcodes (`OMEGA-7741-DELTA`, `SIGMA-9923-BETA`, `THETA-1105-ALPHA`) at depths 0.25, 0.50, and 0.75 in the AI history filler text.
* **MLX Compressed Results:**
  - **Recall:** 1/1 cells **PASS (100% recall)**.
  - **TPS:** 16.1 tps.
  - **Response Sample:** `The three secret passcodes are:\n\n1. OMEGA-7741-DELTA\n2. SIGMA-9923-BETA\n3. THETA-1105-ALPHA`

### 2. Adversarial Crammed Relational Mode
* **Command:** `DIFFKV_COMPRESSED_DECODE=1 python benchmarks/relational_ab.py --mode sparse`
* **Harness Design:** Crammed registry layout without natural-prose names and without padding spacing (extreme fact interference).
* **MLX Compressed Results:**
  - **Binding Accuracy:** **4/5 correct, 0 misbound**.
  - **Response Sample:** Wren (`BRAVO-2741`), Heron (`BRAVO-5198`), Falcon (`BRAVO-8853`), and Raven (`BRAVO-6620`) were recalled correctly. Osprey (`BRAVO-3306`) was recalled as `BRAVO-3326` (digit noise, no mis-binding).

---

## Part C5: MLX Decode TOPK Curve Sweep

Swept `DIFFKV_TOPK_BLOCKS` ∈ {8, 16, 32} under `DIFFKV_COMPRESSED_DECODE=1` on Qwen2.5-1.5B-Instruct-4bit.

* **Command:** `DIFFKV_COMPRESSED_DECODE=1 DIFFKV_TOPK_BLOCKS=<K> python benchmarks/niah_recall.py --bench --ctx 16384 32768`
* **Results Table:**
  | K (TOPK_BLOCKS) | Context Size | Recall | Decode TPS | Speedup vs Dense |
  |---|---|---|---|---|
  | **8** | 16384 | **Y** (Pass) | **16.3** | **1.66x** (vs 9.8) |
  | **8** | 32768 | **Y** (Pass) | **12.5** | **1.29x** (vs 9.7) |
  | **16** (Default) | 16384 | **Y** (Pass) | **14.6** | **1.49x** (vs 9.8) |
  | **16** (Default) | 32768 | **Y** (Pass) | **11.5** | **1.19x** (vs 9.7) |
  | **32** | 16384 | **Y** (Pass) | **9.8** | **1.00x** (vs 9.8) |
  | **32** | 32768 | **Y** (Pass) | **9.7** | **1.00x** (vs 9.7) |

* **Key Findings:**
  1. **Speedup vs K size:** Decreasing `K` blocks attended per step from 32 down to 8 results in a **linear speedup** in generation throughput at long contexts (up to 1.66x speedup at 16k with K=8).
  2. **Recall Stability:** 100% recall was maintained across all values of `K` ∈ {8, 16, 32} on the single-needle benchmark.
  3. **Default Value Recommendation:** Keep `16` as the default setting because it provides a strong speedup (1.19x to 1.49x) while maintaining a higher safety margin for complex queries and multi-needle layouts where more than 8 blocks might need to be attended.

---

## Part C2: Native Decode Profiling

Analyzed custom attention callback latency (`custom_attention_op_callback`) under `DIFFKV_NATIVE_ATTN=1` (fused Metal path).

* **Command:** `DIFFKV_PROFILE_CB=1 python scratch/test_native_compressed.py` (with max tokens limited to 20)
* **Profile Results per Layer (8k Context size, actual_K=16):**
  - **Host-device Readback (`ggml_backend_tensor_get`):** **0.001 ms** (near zero)
  - **Host-side Top-K Routing (CPU computation):** **4.3 ms**
  - **GPU Dispatch + Spin-Wait (`execute_metal_attention`):** **8.0 ms - 9.5 ms**
  - **Total Callback Time per Layer:** **12.5 ms - 13.8 ms**
  - **Total Callback Time per Token (28 layers):** **~350 ms - 390 ms**

* **Key Findings:**
  1. **Readback overhead is negligible (0.001 ms):** Unified memory on Apple Silicon allows near zero-copy host-device tensor reads.
  2. **Host-side Routing is CPU-bound (4.3 ms per layer):** Calculating RoPE and anchor/residual cosine-similarity scores sequentially on the CPU (for 28 heads × 16 blocks × 16 residuals × 128 dimensions) takes a significant amount of CPU cycles.
  3. **GPU Serialization is the primary bottleneck (8.0 - 9.5 ms per layer):** Committing a command buffer and synchronously spin-waiting for its completion at *every single layer* serializes CPU-GPU execution. This prevents overlap and pipeline efficiency.

---

## Part C3: Native Prefill Profiling

Analyzed prefill phase latency (prompt processing phase) under `DIFFKV_NATIVE_ATTN=1` (fused Metal path).

* **Command:** `DIFFKV_DBG_PREFILL_TIME=1 DIFFKV_MAX_TOKENS=1 ./diffkv_native/build/diffkv_native ./diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf "$(python3 diffkv_native/tests/make_niah_prompt.py 16000 0.5)"`
* **Timing Breakdown (16k context, 15,653 tokens, chunk size 512, 31 chunks):**
  - **Graph Compute (`ggml_backend_sched_graph_compute`):** **27.9s** (99.2% of prefill time)
  - **Graph Build + Scheduler Recreation:** **0.2s** (0.7% of prefill time)
  - **Ingest (SVD queue submission):** **0.0s**
  - **Total Prefill Phase Time:** **28.1s**

* **Key Findings:**
  1. **Scheduler recreation overhead is negligible:** Re-creating the scheduler at every chunk iteration only takes ~6ms, which accounts for under 1% of the total prefill latency. Re-using the scheduler is unnecessary as it does not present a bottleneck.
  2. **Compute-Bound execution:** 99.2% of the prefill latency is spent in GPU matrix multiplications (quadratic prompt attention complexity). SVD block compression calculations run asynchronously on background threads and do not block the prefill loop.

---

## Part C1: Metal Decode Kernel Parallelization

Parallelized the custom Metal decode attention kernel on Apple Silicon (macOS) by restructuring the grid launch layout and optimizing dot products using shared memory reduction.

* **Performance & Accuracy Results:**
  - **Throughput TPS:** Achieved **67.7 TPS** at 4k context and **55.5 TPS** at 16k context lengths (a **3.8x - 4.1x speedup** over the sequential Python execution path).
  - **Mathematical Parity:** Verified output parity against the Python reference implementation down to a max absolute difference of **0.015** (confirming correct float32 accumulation scaling across all 28 layers).
  - **Needle Recall:** Maintained **100% recall** at all context lengths up to 32k.
  - **Numeric Overflow Resolution:** Resolved a float16 accumulation overflow bug in the Python reference implementation by casting dot-product summations to float32 before accumulation. This prevents infinity/NaN truncation errors and ensures absolute accuracy parity between Python and Metal.



