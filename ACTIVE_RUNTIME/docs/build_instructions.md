# Build Instructions

## Prerequisites

```bash
pip install torch transformers triton accelerate
```

## Native C++ Extension (optional, for NativeBlockPool)

```bash
cd ACTIVE_RUNTIME/native_core/diffkv_core
python setup.py build_ext --inplace
```

The pre-built `.pyd` is included for Windows (Python 3.13, CUDA 12.x).

## Running the Server

```bash
cd ACTIVE_RUNTIME
python serving/openai_compatible_api_gateway.py     --model Qwen/Qwen2-7B-Instruct     --host 0.0.0.0 --port 8000
```

## Running Tests

```bash
cd ACTIVE_RUNTIME
python tests/test_4k.py
python tests/test_25k.py
python tests/benchmark.py
```
