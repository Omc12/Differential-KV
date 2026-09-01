# Vendored LongBench evaluation assets

These files are copied **unmodified** from the official LongBench repository so
that the evaluation protocol in `benchmarks/run_longbench_cuda.py` is the
authors' own and not a reimplementation of it.

| file                        | source |
|-----------------------------|--------|
| `config/dataset2prompt.json` | `THUDM/LongBench` → `LongBench/config/dataset2prompt.json` |
| `config/dataset2maxlen.json` | `THUDM/LongBench` → `LongBench/config/dataset2maxlen.json` |
| `metrics.py`                 | `THUDM/LongBench` → `LongBench/metrics.py` |
| `eval.py`                    | `THUDM/LongBench` → `LongBench/eval.py` (reference; scoring is driven from `run_longbench_cuda.py`) |
| `pred.py`                    | `THUDM/LongBench` → `LongBench/pred.py` (reference for the truncation and chat-template rules) |

Retrieved from `https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/`
on 2026-09-01. LongBench is released under the MIT License; see the upstream
repository for the full text.

## The dataset itself is NOT vendored

The examples come from the official dataset repo `zai-org/LongBench` (the
renamed `THUDM/LongBench`), downloaded on demand into the shared Hugging Face
cache by `run_longbench_cuda.py::longbench_data_dir`.

The dataset ships as a loading SCRIPT plus `data.zip`. `datasets` 4.x refuses to
execute dataset scripts (`RuntimeError: Dataset scripts are no longer
supported`), so the archive is fetched with `hf_hub_download` and its per-task
JSONL read directly. That is the same content the loading script would have
returned — it only unzips and yields those files — so this is a transport
detail, not a change of data.

## Dependencies

`metrics.py` imports `jieba`, `fuzzywuzzy` and `rouge` at module scope, so all
three must be installed even for an English-only run:

```bash
pip install rouge fuzzywuzzy python-Levenshtein jieba
```
