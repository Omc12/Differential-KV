# Phase 25 — Execution Regime Classification

## The Final Honest Classification

The Differential KV runtime has fundamentally evolved past memory management. Based on the real serving metrics and execution paths verified in this phase, the system officially operates in the following regime:

### **Hierarchical Sparse Transformer Execution**

1. **Storage Tier:**
   - KV history is strictly represented as low-rank `$U @ V$` slabs and sparse anchors.
2. **Retrieval Tier:**
   - Global semantic anchors provide $O(1)$ routing maps to historical knowledge.
3. **Compute Tier:**
   - **Prefill:** Executes explicitly via `RetrievalAwareSparsePrefill` — computing exact attention *only* over Sinks, Local Windows, and retrieved sparse slabs.
   - **Decode:** Executes explicitly via Triton low-rank sparse decode kernels.
4. **Logits Tier:**
   - Explicitly restricted to single-token projection, completely decoupling output vocabulary scaling from sequence length.

**Verdict:** The system is no longer a dense execution engine with a compressed cache. It is a native, end-to-end hierarchical sparse architecture.
