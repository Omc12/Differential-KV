#!/usr/bin/env bash
# Turn a fresh Linux GPU machine into campaign workers, one per GPU.
#
#   export GH_TOKEN=github_pat_...      # fine-grained, Contents: read+write on the repo
#   export HF_TOKEN=hf_...              # optional (all three models are public)
#   curl -fsSL https://raw.githubusercontent.com/Omc12/Differential-KV/main/benchmarks/cloud/bootstrap.sh | bash
#
# Idempotent: run it again after a reboot and it only restarts the workers.
# Everything lives under DKV_ROOT (default /workspace/dkv-cloud), which MUST be
# the same path on every machine: RULER stores record their data directory, and
# a job resumed on another machine is refused if that path differs.
#
# Knobs: DKV_ROOT, DKV_GPUS ("0 1 2 3"; default all), DKV_MACHINE (a short
# name; default hostname), REPO_URL, DKV_RESULTS_BRANCH, TORCH_INDEX (override
# the CUDA wheel index), DKV_SKIP_PREFETCH=1, DKV_NO_START=1 (set up only).
set -euo pipefail

: "${GH_TOKEN:?set GH_TOKEN (fine-grained token, Contents read/write on the repo)}"
REPO_URL="${REPO_URL:-https://github.com/Omc12/Differential-KV.git}"
BRANCH="${DKV_RESULTS_BRANCH:-cloud-results}"
ROOT="${DKV_ROOT:-/workspace/dkv-cloud}"
MACHINE="${DKV_MACHINE:-$(hostname -s)}"
PYVER="${DKV_PYTHON:-3.13}"
export HF_HOME="${HF_HOME:-$ROOT/hf}"
export DKV_RULER_DATA="$ROOT/ruler_data"
mkdir -p "$ROOT"/{logs,q,ruler_data} "$HF_HOME"
say() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }

# ---- system packages: git, a compiler for Triton/Inductor, tmux for humans
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  if ! command -v gcc >/dev/null || ! command -v git >/dev/null; then
    say "installing git, build-essential"
    $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
      git build-essential curl ca-certificates tmux >/dev/null
  fi
fi
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: not a GPU machine?"; exit 1; }
command -v gcc >/dev/null || { echo "no gcc: Inductor cannot compile; install build-essential"; exit 1; }

# ---- GitHub auth from the environment: the token is never written to disk
git config --global credential.helper \
  '!f() { test "$1" = get && echo username=x-access-token && echo "password=${GH_TOKEN}"; }; f'
git config --global advice.detachedHead false

# ---- code: one clone, a worktree per GPU at the campaign's code_ref
if [ ! -d "$ROOT/code/.git" ]; then
  say "cloning $REPO_URL"
  git clone -q "$REPO_URL" "$ROOT/code"
fi
git -C "$ROOT/code" fetch -q origin "+refs/heads/*:refs/remotes/origin/*"
REF="$(git -C "$ROOT/code" show "origin/$BRANCH:control.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["code_ref"])')"
say "campaign code_ref $REF"
git -C "$ROOT/code" checkout -q --detach "$REF"

if [ -z "${DKV_GPUS:-}" ]; then
  DKV_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
for g in $DKV_GPUS; do
  if [ ! -d "$ROOT/w$g" ]; then
    git -C "$ROOT/code" worktree add -q --detach "$ROOT/w$g" "$REF"
  fi
done

# ---- python: uv-managed interpreter + venv, torch from the driver's CUDA line
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "installing uv"
  python3 -m pip install -q --user uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x "$ROOT/venv/bin/python" ]; then
  say "creating venv (python $PYVER)"
  uv venv -q --python "$PYVER" "$ROOT/venv"
fi
PY="$ROOT/venv/bin/python"
if ! "$PY" -c "import torch, transformers, bitsandbytes" 2>/dev/null; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  if [ -z "${TORCH_INDEX:-}" ]; then
    if   [ "$DRV" -ge 580 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu130
    elif [ "$DRV" -ge 570 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu128
    else                          TORCH_INDEX=https://download.pytorch.org/whl/cu126; fi
  fi
  say "driver $DRV -> torch 2.11.0 from $TORCH_INDEX"
  uv pip install -q --python "$PY" "torch==2.11.0" --index-url "$TORCH_INDEX"
  uv pip install -q --python "$PY" -r "$ROOT/code/benchmarks/cloud/requirements.txt"
fi
"$PY" -m nltk.downloader -q punkt punkt_tab >/dev/null 2>&1 || true
"$PY" - <<'EOF'
import torch, triton, transformers
assert torch.cuda.is_available(), "torch sees no GPU"
print("[bootstrap] torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__,
      "transformers", transformers.__version__, "|", torch.cuda.get_device_name(0))
EOF

# ---- models and LongBench, once per machine (workers then run offline)
if [ -z "${DKV_SKIP_PREFETCH:-}" ]; then
  say "downloading models and LongBench into $HF_HOME"
  "$PY" - <<'EOF'
import os
from huggingface_hub import snapshot_download, hf_hub_download
for m in ("Qwen/Qwen3.5-4B", "ibm-granite/granite-4.2-8b", "Qwen/Qwen3.5-9B"):
    # everything the loader may read (chat_template.jinja included), not the
    # alternative weight formats some repos also carry
    snapshot_download(m, ignore_patterns=["*.gguf", "*.pth", "*.bin", "original/*", "*.onnx"])
    print("  ok", m, flush=True)
hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
print("  ok LongBench", flush=True)
EOF
  # RULER items are generated here, not shipped: deterministic, and every file
  # is checked against the hashes of the items the paper used. All sets at
  # once in the background; a worker that needs one first waits on its lock.
  for spec in "Qwen/Qwen3.5-4B 8192" "Qwen/Qwen3.5-4B 131072" "ibm-granite/granite-4.2-8b 8192" \
              "ibm-granite/granite-4.2-8b 24576" "ibm-granite/granite-4.2-8b 32768" \
              "ibm-granite/granite-4.2-8b 131072" "Qwen/Qwen3.5-4B 4096" "Qwen/Qwen3.5-4B 16384" \
              "Qwen/Qwen3.5-4B 32768" "Qwen/Qwen3.5-4B 65536" "ibm-granite/granite-4.2-8b 4096" \
              "ibm-granite/granite-4.2-8b 16384" "Qwen/Qwen3.5-9B 8192" "Qwen/Qwen3.5-9B 131072" \
              "Qwen/Qwen3.5-4B 262144"; do
    set -- $spec
    tag="${1##*/}"
    ( cd "$ROOT/code" && HF_HUB_OFFLINE=1 nohup "$PY" benchmarks/cloud/ruler_data.py ensure \
        --model "$1" --length "$2" --root "$DKV_RULER_DATA" \
        > "$ROOT/logs/ruler_${tag}_$2.log" 2>&1 & )
  done
fi

# ---- workers: one per GPU, restarted if they crash, done when they exit 0
cat > "$ROOT/run_worker.sh" <<EOF
#!/usr/bin/env bash
# One GPU's worker, restarted after a crash; exit 0 means finished or stopped.
g="\$1"
export HF_HOME="$HF_HOME" DKV_RULER_DATA="$DKV_RULER_DATA" HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$ROOT/w\$g"
while true; do
  "$PY" benchmarks/cloud/worker.py --gpu "\$g" --name "$MACHINE-g\$g" --manage-code \\
      --repo-url "$REPO_URL" --branch "$BRANCH" --qdir "$ROOT/q/g\$g" \\
      --ruler-data "$DKV_RULER_DATA" && break
  echo "[run_worker] worker \$g exited \$?; restarting in 60 s" ; sleep 60
done
EOF
chmod +x "$ROOT/run_worker.sh"
if [ -n "${DKV_NO_START:-}" ]; then
  say "DKV_NO_START set: not starting workers ($ROOT/run_worker.sh <gpu> starts one)"
  exit 0
fi
for g in $DKV_GPUS; do
  if pgrep -f "worker.py --gpu $g " >/dev/null; then
    say "worker g$g already running"
  else
    setsid nohup "$ROOT/run_worker.sh" "$g" >> "$ROOT/logs/worker_g$g.log" 2>&1 < /dev/null &
    say "started worker g$g -> $ROOT/logs/worker_g$g.log"
  fi
done
say "done. watch: tail -f $ROOT/logs/worker_g*.log   (stop: ctl.py stop, or pkill -f run_worker.sh)"
