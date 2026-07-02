# Building & running Differential-KV on a fresh machine

These are the canonical build steps derived from `diffkv_native/CMakeLists.txt` and
`ACTIVE_RUNTIME/native_core/diffkv_core/setup.py`. They contain **no absolute/author-specific paths** —
everything resolves relative to the checkout or to env vars.

> Status note: these steps are transcribed from the build scripts; verify against your toolchain
> versions. The repo previously shipped some prebuilt artifacts (see the cleanup section) — remove
> those first so you're testing a real build, not a stale binary.

---

## 1. Prerequisites

| Platform | Needs |
|---|---|
| All | Python 3.10+ (repo dev used 3.14), CMake ≥ 3.20, a C++17 compiler, git |
| macOS (Apple Silicon) | Xcode Command Line Tools (`xcode-select --install`) for `clang`/`xcrun metal`; the native engine uses Metal + Accelerate |
| Linux/CUDA | CUDA Toolkit (nvcc, cuSOLVER, cuBLAS) if building the GPU paths |

The `llama.cpp` dependency is a git submodule:

```bash
git submodule update --init --recursive
```

> **IMPORTANT — fresh clones:** the pinned submodule commit (branch
> `diffkv-fused-op`, the DiffKV fused-attention ggml/Metal kernels) exists
> only in this repo, **not** on upstream `ggerganov/llama.cpp`, so the plain
> `submodule update` above will fail to find it. The commits are vendored in
> `diffkv_native/third_party/diffkv-fused-op.bundle` (exact SHAs preserved).
> Restore them with:
>
> ```bash
> git submodule init
> git -C diffkv_native/third_party/llama.cpp fetch origin d2462f8f7ac6d80070a587ffebf6cd73730f4280 || \
>     git submodule update --init diffkv_native/third_party/llama.cpp || true
> git -C diffkv_native/third_party/llama.cpp fetch \
>     "$(git rev-parse --show-toplevel)/diffkv_native/third_party/diffkv-fused-op.bundle" \
>     diffkv-fused-op:diffkv-fused-op
> git submodule update diffkv_native/third_party/llama.cpp
> ```
>
> Long-term fix (TODO): push `diffkv-fused-op` to a fork of llama.cpp under
> your GitHub account and point `.gitmodules` at the fork. Regenerate the
> bundle after any new submodule commits:
> `git -C diffkv_native/third_party/llama.cpp bundle create ../diffkv-fused-op.bundle $(git -C diffkv_native/third_party/llama.cpp merge-base origin/master diffkv-fused-op)..diffkv-fused-op`

---

## 2. Python environment (the "active" runtime)

```bash
python3 -m venv diffkv_venv
source diffkv_venv/bin/activate
pip install -r ACTIVE_RUNTIME/requirements.txt

# Apple Silicon: the active runtime is MLX-only — MLX is REQUIRED here, not optional:
pip install mlx            # macOS / M-series only

# CUDA machines: install the GPU extras that are commented out in requirements.txt:
pip install "triton>=2.1.0"
```

Run the active (Python) runtime from `ACTIVE_RUNTIME/` (it puts `native_core`, `runtime`, `serving` on
the path). On macOS `DiffKVHFWrapper` auto-selects the MLX backend; on CUDA it uses the PyTorch path.

---

## 3. `diffkv_core` CPython extension (used by the PyTorch/MPS active path)

`setup.py` is platform-aware: on macOS it builds a `CppExtension` (Accelerate, no CUDA) and
auto-generates the embedded Metal library (`xcrun metal` → `.metallib` → `diffkv_metallib.hpp`); on
CUDA it builds a `CUDAExtension` (cuSOLVER/cuBLAS).

```bash
cd ACTIVE_RUNTIME/native_core/diffkv_core
python setup.py build_ext --inplace
cd -
```

This produces `diffkv_core.cpython-<ver>-<platform>.so` **for your interpreter/arch** — it is not
portable, which is exactly why it must not be committed (see §6).

---

## 4. `diffkv_native` C++ engine (GGUF / llama.cpp)

```bash
cd diffkv_native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     # add -DGGML_CUDA=ON on CUDA boxes
cmake --build build -j
# → build/diffkv_native
cd -
```

The Metal shader path is now portable (fixed 2026-07-02): `runtime/diffkv_attention.mm` resolves
`diffkv_decode.metal` via (0) the `DIFFKV_METAL_DIR` env var, (1) executable-relative, (2) CWD-relative,
(3) a CMake-injected `DIFFKV_METAL_SOURCE_DIR` pointing at *this build's* source tree. No home-dir path.

---

## 5. Models

GGUF weights are **not** in git (they're gitignored). Place them under `diffkv_native/`, e.g.
`qwen2.5-0.5b-instruct.gguf`, `qwen2.5-1.5b-instruct-q4_k_m.gguf`. Override locations with:

```bash
export DIFFKV_MODEL_PATH=/path/to/model.gguf
export DIFFKV_BINARY_PATH=/path/to/diffkv_native
```

The native serving gateway (`diffkv_native/serving/openai_compatible_api_gateway.py`) now defaults these
relative to the checkout and honors the env vars above (fixed 2026-07-02).

---

## 6. Untracking committed build artifacts (one-time cleanup)

Some build outputs were committed before the `.gitignore` rules existed. `.gitignore` does not untrack
already-indexed files, so run this once (from a clean/committed working tree, so it doesn't get mixed
into other in-progress changes). Files stay on disk — only the git tracking is removed:

```bash
git rm -r --cached \
  ACTIVE_RUNTIME/native_core/diffkv_core/diffkv_core.cpython-314-darwin.so \
  ACTIVE_RUNTIME/native_core/diffkv_core/build \
  'diffkv_native/native_core/**/__pycache__'
git commit -m "chore: stop tracking per-platform build artifacts"
```

After this, everyone rebuilds §3/§4 locally and the arch-locked `.so` never masks a failed build again.

---

## 7. Smoke test

```bash
# Active (MLX/PyTorch) — kernel parity is the ground-truth oracle:
diffkv_venv/bin/python -m pytest ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py

# Native — needle recall:
bash diffkv_native/tests/test_niah_native.sh
```
