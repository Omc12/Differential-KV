# Phase 24.8 — Compressed Prefill Execution

## The Core Frontier Question
> Can prefill attention execute *directly* over compressed historical blocks without dense reconstruction?

## The Mathematical Proof

Assume a historical block of length $T$ (e.g., 64 tokens) is compressed into a low-rank form (rank $R$, e.g., 8).
The keys $K$ and values $V$ are approximated as:
$$ K \approx U_k V_k $$
$$ V_{orig} \approx U_v V_v $$

A new query $Q$ of length $L$ needs to attend to this block. 
Standard attention requires:
$$ S = Q K^T $$
$$ O = \text{softmax}(S) V_{orig} $$

If we substitute the low-rank forms:
$$ S = Q (U_k V_k)^T = Q V_k^T U_k^T $$
$$ O = \text{softmax}(S) (U_v V_v) $$

### The Execution Order
To avoid dense reconstruction (i.e., avoiding $U_k V_k \to T \times D$), we manipulate the matrix multiplication order:

1. **Low-Rank Projection:**
   $$ P = Q V_k^T $$ 
   - Size: $L \times R$. (Extremely small).
2. **Score Computation:**
   $$ S = P U_k^T $$
   - Size: $L \times T$. (This matches the chunk dimensions, bounded to e.g. $512 \times 64$).
3. **Softmax Application:**
   Compute softmax over $S$. (Requires maintaining running max/sum for global normalization across all blocks, identical to the FlashAttention algorithm).
4. **Value Projection:**
   $$ O_{temp} = \text{softmax}(S) U_v $$
   - Size: $L \times R$. 
5. **Final Output:**
   $$ O_{block} = O_{temp} V_v $$
   - Size: $L \times D$.

## Conclusion

**YES.** Prefill attention can execute natively and directly over compressed historical blocks.
By maintaining a running maximum $m$ and denominator $l$ (using the online softmax trick from FlashAttention), we can loop through the list of `KVBlock` objects, execute the sequence above, and accumulate the final $O$ output without EVER materializing the dense $K$ or $V$ tensors, and without ever running `TritonDKV.reconstruct_lowrank`.

This proves that Differential KV can be an end-to-end sparse memory runtime across both decoding AND multi-turn prefill.
