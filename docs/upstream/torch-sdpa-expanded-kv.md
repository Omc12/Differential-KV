# torch SDPA (memory-efficient backend) returns wrong output on stride-0 K/V with a boolean mask

Found 2026-11-25 while debugging DKV (KV-cache compression) on gemma-4. One PyTorch
issue; transformers contributes the trigger (see 4). Everything below was measured
on one machine; items marked UNVERIFIED were not checked.

## 1. Title, component, location

**Title:** `scaled_dot_product_attention` memory-efficient backend gives wrong results
when K/V are broadcast views (stride 0 on the head axis) and an `attn_mask` is passed,
at some query lengths.

- Component: `torch.nn.functional.scaled_dot_product_attention`, CUDA,
  `SDPBackend.EFFICIENT_ATTENTION` (the mem-efficient / CUTLASS-derived kernel).
- PyTorch: `2.11.0+cu130`, git `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
- Kernel and dispatcher source locations: UNVERIFIED. They were not opened from this
  session. The usual places are `aten/src/ATen/native/transformers/cuda/attention.cu`
  (`_scaled_dot_product_efficient_attention_cuda` / `_efficient_attention_forward`),
  `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`, and the
  backend selection in `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`.
  Confirm line numbers against the commit above before citing them.
- The trigger comes from transformers 5.14.1, `transformers/integrations/sdpa_attention.py`:
  `repeat_kv` (line 16) expands K/V and the `torch.nn.functional.scaled_dot_product_attention`
  call is at line 154. For a layer with exactly ONE KV head, `repeat_kv`'s
  `expand(...).reshape(...)` is a free view, so the kernel receives K/V with
  stride 0 on the batch and head axes (observed `k.stride() == (0, 0, 256, 1)`).

## 2. What goes wrong

- Expected: same output as `SDPBackend.MATH` on the same inputs, within normal bf16
  rounding (~1.7e-3 relative error in this setup).
- Actual: relative error 4e-2 to 8e-2 at some query lengths, with no error or warning.
- Severity: **silent wrong results.** No crash, no performance or memory impact.
  In a real model (gemma-4-e2b-it, nf4 weights, transformers SDPA path, chunked prefill
  in 1025-token chunks) it pushed the residual stream ~7.6% off from the first
  affected global layer onward. On gemma-4-e2b at 32k this bug accounted for
  first-token KL 2.57 → 0.96. CORRECTION: the remaining ~0.9-1.3 on gemma
  e2b/e4b/12B was a separate DKV harness artifact, not torch: the DKV arm hooked
  lm_head before gemma's final logit softcap and the control read capped logits.
  e4b, whose sliding layers have 2 KV heads, never hit the torch bug at all. Top-1
  mostly still agreed, which is why it went unnoticed.

## 3. Minimal repro

Environment: Windows 11 Pro 10.0.26300, Python 3.13.13, torch 2.11.0+cu130 (CUDA 13.0,
cuDNN 91900), NVIDIA GeForce RTX 4070 SUPER (sm_89), driver 621.01. No transformers
needed.

```python
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.manual_seed(0)

def run(qlen, past=0, D=256, H=8, W=512, expand=True):
    kv = past + qlen
    q = torch.randn(1, qlen, H, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k1 = torch.randn(1, 1, kv, D, device="cuda", dtype=torch.bfloat16) * 0.3
    v1 = torch.randn(1, 1, kv, D, device="cuda", dtype=torch.bfloat16)
    k, v = k1.expand(1, H, kv, D), v1.expand(1, H, kv, D)          # stride 0 on heads
    if not expand:
        k, v = k.contiguous(), v.contiguous()
    qp = torch.arange(past, past + qlen, device="cuda")[:, None]
    kp = torch.arange(kv, device="cuda")[None, :]
    mask = ((kp <= qp) & (qp - kp < W))[None, None]                 # sliding-window causal, bool
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask, scale=1.0)
    return float((out.float() - ref).norm() / ref.norm())

for L in (513, 832, 1024, 1025, 1026, 2048, 2050):
    print(L, f"expanded {run(L):.1e}  contiguous {run(L, expand=False):.1e}  "
             f"expanded, 511 past {run(L, past=511):.1e}")
```

Output on the machine above:

| qlen | expanded K/V | contiguous K/V | expanded, 511 cached |
|---|---|---|---|
| 513 | **8.2e-2** | 1.7e-3 | **4.9e-2** |
| 832 | 1.7e-3 | 1.7e-3 | 1.7e-3 |
| 1024 | 1.7e-3 | 1.7e-3 | 1.7e-3 |
| 1025 | **5.9e-2** | 1.7e-3 | **4.1e-2** |
| 1026 | 1.7e-3 | 1.7e-3 | 1.7e-3 |
| 2048 | 1.7e-3 | 1.7e-3 | 1.7e-3 |
| 2050 | 1.7e-3 | 1.7e-3 | 1.7e-3 |

Making only K/V contiguous (Q still the transposed view) also gives 1.7e-3 at every
length, so the transposed Q is not required.

## 4. Root-cause analysis (what is and is not known)

Established:
- The math backend is correct on identical inputs. Selecting only the efficient
  backend reproduces the error.
- Contiguous K/V removes it; it needs the stride-0 head axis.
- The mask is involved. In an in-model spy (wrapping every SDPA call and comparing
  against math), calls with `attn_mask=None, is_causal=True` on the same stride-0 K/V
  were correct (1.7e-3). Every bad call had a bool mask. A pure-repro check with
  `attn_mask=None` was not done.
- Bad query lengths seen: 513, 1025 (and 1537 in-model). Good: 512, 832, 1024, 1026,
  1028, 1536, 2048.

NOT established:
- Why only some lengths fail. 513 and 1025 are both one past a multiple of 512, but
  so is 1537 while 2049 was not tested. A "length ≡ 1 mod 512" rule was hypothesised
  and is unconfirmed.
- **A possible second trigger:** plain transformers at chunk 2050 on gemma-4-e2b was
  also wrong (from the first sliding layer). That case was fixed by forcing the math
  backend or eager attention. But the minimal repro above is correct at L=2050. So
  there is another shape or stride condition not yet captured; mask stride/layout or
  kv-length alignment are candidates. Use the in-model spy (below) to get its exact
  shapes.
- Which code path in the kernel mishandles the stride-0 batch/head stride together
  with the attn_bias. One plausible area is the attn_bias alignment/padding path
  combined with zero K/V strides; this is a hypothesis only.

## 5. Workaround and proposed upstream fixes

DKV workaround (in the DKV repo, `ACTIVE_RUNTIME/runtime/dkv_attention.py`,
`install_sdpa_expanded_kv_guard()`, installed once at model load from
`ACTIVE_RUNTIME/serving/hf_dkv_wrapper.py`). It wraps
`torch.nn.functional.scaled_dot_product_attention`: when `attn_mask is not None`, K is
4-D with more than one head and `key.stride(1) == 0` (likewise value), it calls
`.contiguous()` first. All other calls pass through untouched. Uncommitted as of this
writing; unit test `ACTIVE_RUNTIME/tests/test_sdpa_expanded_kv_guard.py`.

Proposed upstream fixes (none tried):
- PyTorch: in SDPA backend selection, reject the efficient kernel (or materialise
  K/V) when K/V have a zero stride on batch/head axes and a mask is given, until the
  kernel handles it. A real kernel fix needs the root cause above.
- transformers (optional, defensive): avoid handing a free stride-0 view to SDPA
  for one-KV-head layers. Use `enable_gqa=True` where supported, or make the
  repeated K/V contiguous. Note the efficient kernel rejected `enable_gqa=True` in
  this setup (it fell to math), so the second option is the safe one.

## 6. Measured numbers

Kernel level: the table in section 3 (bad 4e-2 to 8e-2 → 1.7e-3 with contiguous K/V).

Model level, gemma-4-e2b-it nf4 at 8k (last-position hidden state per decoder layer,
relative error against a plain-transformers reference):

| run | first global DKV layer (9) | last layer (34) |
|---|---|---|
| plain transformers, chunk 512 (reference) | - | - |
| plain transformers, chunk 1025, SDPA default | 7.6e-2 | ~8e-2 |
| plain transformers, chunk 1025, SDPA math only | 1.5e-2 | noise |
| plain transformers, chunk 1025, eager attention | 1.8e-2 | 2.4e-2 |
| DKV, chunk 1025, with the guard | 1.8e-2 | 1.8e-2 |

The ~1-2% floor is bf16/fp16 and chunking noise; a one-shot forward sits at the same
level.

Fidelity metric (32k, 5 prompts, first generated token, KL vs plain transformers):
gemma-4-e2b 2.57 before the guard, 0.96 after. e4b 1.29 → 1.29 (2 KV heads, not
affected). 12B 0.90 → 0.85. The remainder is the softcap harness artifact described
in section 2, being re-measured in the DKV session (V13f). Only e2b's 2.57 → 0.96
should be attributed to the torch bug.

## 7. Existing upstream reports

**Not checked.** No search of pytorch/pytorch or huggingface/transformers issues or
PRs was done, and no nightly build was tested (that needs a download the user has not
approved yet). Do both before filing.

## Tools in the DKV repo

- `benchmarks/diag_prefill_layer_diff.py`: per-decoder-layer diff of the last-position
  hidden state, plain transformers vs DKV. Options: `--chunk`, `--attn eager`,
  `--sdpa-backend math|efficient|flash`, `--unpatch` (put a DKV layer back on the
  native forward). It isolated this bug in four steps: DKV layer native → unchanged;
  plain HF chunk 1025 → reproduces; eager / math → correct; efficient only → wrong.
- In-model spy (scratch, not in repo): monkeypatches
  `torch.nn.functional.scaled_dot_product_attention` during a plain-transformers
  chunked prefill. For each distinct (shapes, mask, strides) it prints the relative
  error against the math backend. Useful for capturing the unexplained chunk-2050
  shapes.

## Blog notes (story order)

1. A cross-architecture fidelity panel showed gemma-4 first-token KL 0.9-2.6 while
   top-1 still agreed. The same gap had been accepted as a pass twice before.
2. A per-layer diff named the first DKV-patched layer.
3. Putting that layer back on the native forward left the error unchanged, so it
   was not DKV.
4. Plain transformers at the same chunk size reproduced it. A chunk-size sweep
   suggested "≡ 1 mod the sliding window", and chunk 2050 broke that rule.
5. Eager attention, then the math SDPA backend, were correct. Efficient-only was
   wrong.
6. A spy on SDPA calls found the bad calls: one-KV-head layers with stride-0 K/V and
   a bool mask. A 25-line pure-torch repro followed.
