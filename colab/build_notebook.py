#!/usr/bin/env python3
"""Generator for run_a100_paper_experiments.ipynb (single source of truth).

Run:  python colab/build_notebook.py
Emits colab/run_a100_paper_experiments.ipynb consistent with REWRITE 16.0 of
run_a100_paper_experiments.py (corrected measurement core, faithful baselines,
RULER suite, accuracy-vs-memory Pareto).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l if l.endswith("\n") else l + "\n" for l in lines]}


CELLS = [
    md(
        "# 🚀 Differential-KV (DiffKV) — A100 Research Paper Benchmark Suite (REWRITE 16.0)",
        "",
        "Runs the DiffKV evaluation suite on an **NVIDIA A100 (40/80 GB)** in Lightning AI, Colab, or Jupyter.",
        "",
        "### Models",
        "- Primary: `Qwen/Qwen2.5-7B-Instruct` (FP16 weights for ALL methods — KV memory is isolated from weight quantization).",
        "- Second **family** (generalization, not just a bigger Qwen): `meta-llama/Llama-3.1-8B-Instruct`.",
        "",
        "### What this measures (and what changed from 15.0)",
        "- **Correct DiffKV KV memory** via the pool's real allocation (`native_pool._pool_mb()`), not the empty legacy `session_blocks` dict — the honest ~1.4–2.5× ratio, not the old fake counter.",
        "- **Correct decode throughput** — prefill / compress / decode are measured on ONE session with a token-by-token loop (no hidden second prefill from `generate()`); the dense + baseline paths use the SAME loop and chunk size.",
        "- **One consistent memory metric** for every method + true peak VRAM (weights identical across methods).",
        "- **Faithful baselines**: Dense · INT8-KV · KIVI-2/3/4-bit · StreamingLLM · **SnapKV** (real accumulated-attention eviction) · KeyNorm-HH (H2O-style key-norm proxy, labeled honestly) · DiffKV.",
        "- **RULER-style suite** (multi-key / multi-value / multi-query NIAH, variable tracking, freq-words) with Wilson 95% CIs.",
        "- **exp22 accuracy/KV-memory frontier** + **exp23 quality-vs-context curve** — the two experiments that decide whether the 'less RAM at dense quality' claim is un-dominated. Run these FIRST (`--only 22,23`).",
        "",
        "> ⚠️ Quant baselines (INT8/KIVI) report the *theoretical* KV footprint; the harness dequantizes to FP16 for decode (no fused quant-attention kernel), so their peak-VRAM is FP16-optimistic. DiffKV/dense/eviction footprints are realized.",
    ),
    md("## Step 1: Environment & GPU diagnostics"),
    code(
        "!nvidia-smi",
        "import torch",
        "print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())",
        "if torch.cuda.is_available():",
        "    p = torch.cuda.get_device_properties(0)",
        "    print('Device:', p.name, '| VRAM: %.1f GB' % (p.total_memory / 1e9))",
    ),
    md(
        "> ⚠️ **transformers must be 4.x.** DiffKV's CUDA attention interception uses the transformers 4.x",
        "> `Qwen2Attention.forward` signature. On 4.48+/5.x, `position_embeddings` became a positional arg and the",
        "> patch silently produces **garbage output** (and dense falls to eager → 16k OOM). The pin below fixes it.",
    ),
    code("!pip install -q 'transformers==4.46.3' accelerate bitsandbytes triton matplotlib seaborn tabulate psutil requests urllib3 datasets"),
    code(
        "import transformers",
        "assert transformers.__version__.startswith('4.4'), (",
        "    f'transformers {transformers.__version__} will garble DiffKV — pin 4.x: pip install \"transformers==4.46.3\" then RESTART the kernel')",
        "print('transformers', transformers.__version__, 'OK (4.x)')",
    ),
    md(
        "## Step 2a: Decision run FIRST — exp22 (frontier) + exp23 (quality-vs-context)",
        "",
        "These two are cheap and tell you whether the claim survives before you spend hours on the full grid.",
        "If DiffKV does NOT track dense while KIVI/SnapKV peel away, reframe before running everything else.",
    ),
    code(
        "import os",
        "os.environ.setdefault('DIFFKV_FRONTIER_SAMPLES', '3')",
        "os.environ.setdefault('DIFFKV_CURVE_SAMPLES', '3')",
        "# DIFFKV_ISOLATE_WORKERS=1 runs each config in a fresh subprocess so the GPU",
        "# is fully freed between model loads — required on a 40GB A100 or you OOM.",
        "!DIFFKV_ISOLATE_WORKERS=1 python colab/run_a100_paper_experiments.py \\",
        "    --model Qwen/Qwen2.5-7B-Instruct --only 22,23 \\",
        "    --out diffkv_decision.json",
    ),
    md(
        "## Step 2b: Full run (long — many model loads)",
        "Knobs: `--only 1,5,5b,21` for subsets; `DIFFKV_BENCH_TRIALS` / `DIFFKV_RULER_SAMPLES` / `DIFFKV_NIAH_SAMPLES` set statistical N.",
    ),
    code(
        "os.environ.setdefault('DIFFKV_BENCH_TRIALS', '3')",
        "os.environ.setdefault('DIFFKV_RULER_SAMPLES', '5')",
        "os.environ.setdefault('DIFFKV_NIAH_SAMPLES', '3')",
        "# Isolate each config in a subprocess (frees GPU between loads → no 40GB OOM).",
        "!DIFFKV_ISOLATE_WORKERS=1 python colab/run_a100_paper_experiments.py \\",
        "    --model Qwen/Qwen2.5-7B-Instruct \\",
        "    --model-14b meta-llama/Llama-3.1-8B-Instruct \\",
        "    --out diffkv_paper_results.json",
    ),
    md(
        "## Step 2c: The make-or-break figures (from the decision run)",
        "Frontier: does DiffKV sit on the accuracy/memory frontier the strong baselines can't dominate?",
        "Curve: does DiffKV track dense out to long context while KIVI/SnapKV/Streaming peel away?",
    ),
    code(
        "import json, matplotlib.pyplot as plt, seaborn as sns",
        "sns.set_theme(style='whitegrid')",
        "dec = json.load(open('diffkv_decision.json'))",
        "",
        "# ---- exp22 frontier: recall vs KV memory ----",
        "fr = dec.get('exp22_memory_quality_frontier', {}).get('points', [])",
        "if fr:",
        "    fam = lambda l: l.split('-')[0]",
        "    fams = sorted({fam(p['label']) for p in fr})",
        "    cmap = {f: c for f, c in zip(fams, sns.color_palette('tab10', len(fams)))}",
        "    fig, ax = plt.subplots(figsize=(8, 5))",
        "    for p in fr:",
        "        ax.scatter(p['kv_physical_gb'], p['recall_pct'], color=cmap[fam(p['label'])], s=80,",
        "                   marker='o' if p['kv_footprint_realized'] else 'x')",
        "        ax.annotate(p['label'], (p['kv_physical_gb'], p['recall_pct']), xytext=(4, 3),",
        "                    textcoords='offset points', fontsize=7)",
        "    ax.set_xlabel('KV footprint (GB, lower=better)  [x = theoretical quant footprint]')",
        "    ax.set_ylabel('niah_multikey recall (%, higher=better)')",
        "    ax.set_title('exp22: accuracy / KV-memory frontier @ 32K')",
        "    plt.tight_layout(); plt.savefig('plot_frontier.png', dpi=300, bbox_inches='tight'); plt.show()",
        "",
        "# ---- exp23 quality vs context ----",
        "cv = dec.get('exp23_quality_vs_context', {}).get('curves', {})",
        "if cv:",
        "    fig, ax = plt.subplots(figsize=(8, 5))",
        "    for label, pts in cv.items():",
        "        xs = sorted(int(c) for c in pts)",
        "        ys = [pts[str(x)]['recall_pct'] if str(x) in pts else pts[x]['recall_pct'] for x in xs]",
        "        ax.plot(xs, ys, marker='o', linewidth=2.5 if label == 'dense' else 1.8,",
        "                linestyle='--' if label == 'dense' else '-', label=label)",
        "    ax.set_xscale('log', base=2); ax.set_xticks(xs); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())",
        "    ax.set_xlabel('context length'); ax.set_ylabel('recall (%)')",
        "    ax.set_title('exp23: quality vs context — does DiffKV track dense?'); ax.legend()",
        "    plt.tight_layout(); plt.savefig('plot_quality_vs_context.png', dpi=300, bbox_inches='tight'); plt.show()",
    ),
    md(
        "## Step 3: Headline figure — Accuracy vs KV Memory Pareto",
        "All methods on identical axes and the same measurement. Up-and-to-the-left is better.",
    ),
    code(
        "import json, matplotlib.pyplot as plt, seaborn as sns",
        "sns.set_theme(style='whitegrid')",
        "data = json.load(open('diffkv_paper_results.json'))",
        "",
        "pareto = data.get('pareto_accuracy_vs_memory', {})",
        "if pareto:",
        "    fig, ax = plt.subplots(figsize=(8, 5))",
        "    for name, r in pareto.items():",
        "        ax.scatter(r['kv_physical_gb'], r['recall_pct'], s=90)",
        "        ax.annotate(name, (r['kv_physical_gb'], r['recall_pct']),",
        "                    xytext=(5, 5), textcoords='offset points', fontsize=9)",
        "    ax.set_xlabel('KV footprint (GB, lower better)')",
        "    ax.set_ylabel('NIAH recall @16K (%, higher better)')",
        "    ax.set_title('DiffKV vs baselines — accuracy / KV-memory Pareto')",
        "    plt.tight_layout(); plt.savefig('plot_pareto.png', dpi=300, bbox_inches='tight'); plt.show()",
        "else:",
        "    print('Run exp21 to populate the Pareto frontier.')",
    ),
    md("## Step 4: RULER-style recall heatmap (DiffKV-mid)"),
    code(
        "import numpy as np",
        "ruler = data.get('exp5b_ruler', {}).get('mid', {})",
        "if ruler:",
        "    tasks = list(ruler.keys())",
        "    ctxs = sorted({int(c) for t in ruler.values() for c in t})",
        "    M = np.array([[ruler[t].get(str(c), ruler[t].get(c, {})).get('pass_rate_pct', np.nan) for c in ctxs] for t in tasks])",
        "    fig, ax = plt.subplots(figsize=(7, 4))",
        "    sns.heatmap(M, annot=True, fmt='.0f', xticklabels=ctxs, yticklabels=tasks, cmap='viridis', vmin=0, vmax=100, ax=ax)",
        "    ax.set_title('RULER-style pass-rate % (DiffKV-mid)'); ax.set_xlabel('context length')",
        "    plt.tight_layout(); plt.savefig('plot_ruler.png', dpi=300, bbox_inches='tight'); plt.show()",
        "else:",
        "    print('Run exp 5b to populate RULER results.')",
    ),
    md(
        "## Step 4b: Prefill-compress lever — run the SEPARATE Gram-eigh decision test",
        "",
        "This is a standalone test (not part of the suite). It CPU-verifies Gram-eigh ≡ SVD, then A/Bs",
        "compress time + NIAH recall on the A100 and tells you whether it's safe to make Gram-eigh the default.",
    ),
    code(
        "!python colab/gram_eigh_decision.py --gpu-ab --model Qwen/Qwen2.5-7B-Instruct --ctx 16384 --samples 3",
    ),
    md(
        "## Step 5: Camera-ready Markdown report (rendered inline)",
        "Written to the working directory AND displayed here — on Lightning the file browser may not",
        "show freshly-written files, so we render it inline and print its absolute path.",
    ),
    code(
        "import os, json",
        "from IPython.display import Markdown, display, Image",
        "data = json.load(open('diffkv_paper_results.json'))  # self-contained: reload results",
        "report_path = 'diffkv_a100_paper_report.md'",
        "base = data.get('exp21_external_baselines', {})",
        "lines = ['# DiffKV A100 Paper Report', '', '## Baselines @16K NIAH (consistent metric)', '',",
        "         '| Method | KV footprint (GB) | Compression | Recall % | Decode TPS | Realized? |',",
        "         '|---|---|---|---|---|---|']",
        "for name, r in base.items():",
        "    if not isinstance(r, dict) or r.get('status') != 'success':",
        "        continue",
        "    lines.append('| %s | %.3f | %.2fx | %.0f | %.1f | %s |' % (",
        "        name, r.get('kv_physical_gb', 0), r.get('compression_ratio', 0),",
        "        r.get('recall', 0), r.get('decode_tps', 0),",
        "        'yes' if r.get('kv_footprint_realized', True) else 'theoretical'))",
        "report = '\\n'.join(lines)",
        "open(report_path, 'w').write(report)",
        "print('report file →', os.path.abspath(report_path))",
        "display(Markdown(report))                       # render the report inline",
        "for _p in ['plot_pareto.png', 'plot_frontier.png', 'plot_quality_vs_context.png', 'plot_ruler.png']:",
        "    if os.path.exists(_p):",
        "        print('figure →', os.path.abspath(_p)); display(Image(_p))",
    ),
]

NB = {"cells": CELLS, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}

if __name__ == "__main__":
    out = os.path.join(HERE, "run_a100_paper_experiments.ipynb")
    with open(out, "w") as f:
        json.dump(NB, f, indent=1)
    print("Wrote", out)
