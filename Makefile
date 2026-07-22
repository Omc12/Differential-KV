# Differential-KV — one-command setup & run.
#
# macOS / Apple Silicon (MLX) works out of the box:  make setup && make chat
# Linux / CUDA: `make setup` installs the base deps; see BUILD.md for the CUDA
# extras (triton, cuSOLVER/cuBLAS) and the native `-DGGML_CUDA=ON` build.
#
# The default MODEL is a HuggingFace id; on macOS the wrapper auto-selects the
# matching mlx-community 4-bit build. Override with:  make chat MODEL=<id>

PY    := dkv_venv/bin/python
PIP   := dkv_venv/bin/pip
MODEL ?= Qwen/Qwen2.5-1.5B-Instruct

.DEFAULT_GOAL := help

help:
	@echo "Differential-KV — common commands:"
	@echo "  make setup    create venv + install Python deps (run this first)"
	@echo "  make chat     interactive DKV chat CLI       (MODEL=$(MODEL))"
	@echo "  make serve    OpenAI-compatible API on :8000    (MODEL=$(MODEL))"
	@echo "  make test     NIAH recall guardrail (8k + 16k)"
	@echo "  make native   build the C++ native engine (submodule + cmake)"
	@echo ""
	@echo "  Override the model:  make chat MODEL=Qwen/Qwen2.5-0.5B-Instruct"

setup:
	python3 -m venv dkv_venv
	$(PIP) install --upgrade pip
	$(PIP) install -r ACTIVE_RUNTIME/requirements.txt
	@echo ""
	@echo "Setup complete. Next:  make chat"

chat:
	$(PY) ACTIVE_RUNTIME/serving/cli.py --model $(MODEL) --preset mid

serve:
	$(PY) ACTIVE_RUNTIME/serving/openai_compatible_api_gateway.py --model $(MODEL) --port 8000

test:
	$(PY) benchmarks/niah_recall.py --bench --ctx 8192 16384

native:
	git submodule update --init --recursive
	cd dkv_native && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

.PHONY: help setup chat serve test native
