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
