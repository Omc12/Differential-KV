# BENCHMARK CLASSIFICATION

MODE:
PRODUCTION

INCLUDED:
autoregressive_pressure, batching, concurrency, concurrent_decode, embeddings, heavy_serving, kv_virtualization, logits, model_residency, multi_user_decode, queue_contention, sampling, serialization_overhead, streaming, tokenizer, triton_kernels, vram_pressure

EXCLUDED:
cuda_graphs, hf_dispatch, mlp, serving_overhead, sparse_attention

REAL MODEL WEIGHTS:
NO

REAL LOGITS:
YES

REAL SAMPLING:
YES

REAL WALL CLOCK:
NO

REAL VRAM:
NO

SYNTHETIC ACCOUNTING:
YES
