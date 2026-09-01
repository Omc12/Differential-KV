# Vendored RULER generation and scoring

Copied **unmodified** from `NVIDIA/RULER` (Apache-2.0), retrieved 2026-09-01
from `https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/`.

RULER is legitimately a synthetic *generator* — "synthetic" is not the defect
this replaces. The defect is that this repo previously carried a HOMEMADE
approximation of RULER (`colab/run_a100_paper_experiments.py`, "RULER-style
synthetic, dataset-free"), whose numbers are not comparable to the published
tables. These are the authors' own generators, so they are.

| local path | upstream |
|---|---|
| `data/synthetic/niah.py` | `scripts/data/synthetic/niah.py` |
| `data/synthetic/qa.py` | `scripts/data/synthetic/qa.py` |
| `data/synthetic/variable_tracking.py` | `scripts/data/synthetic/variable_tracking.py` |
| `data/synthetic/common_words_extraction.py` | `scripts/data/synthetic/common_words_extraction.py` |
| `data/synthetic/freq_words_extraction.py` | `scripts/data/synthetic/freq_words_extraction.py` |
| `data/synthetic/constants.py` | `scripts/data/synthetic/constants.py` |
| `data/template.py`, `data/tokenizer.py`, `data/prepare.py`, `data/manifest_utils.py` | `scripts/data/` |
| `synthetic.yaml` | `scripts/synthetic.yaml` — the 13 task definitions |
| `config_tasks.sh` | `scripts/config_tasks.sh` |
| `eval/evaluate.py`, `eval/synthetic/constants.py` | `scripts/eval/` — `string_match_all` / `string_match_part` |

## The 13 tasks and what each needs

| task | generator | external data needed |
|---|---|---|
| `niah_single_1` | niah (`noise` haystack) | none |
| `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multivalue`, `niah_multiquery` | niah (`essay` haystack) | **Paul Graham essays** |
| `niah_multikey_2`, `niah_multikey_3` | niah (`needle` haystack) | none |
| `vt` | variable_tracking (`noise`) | none |
| `cwe` | common_words_extraction | `english_words.json` ✔ present |
| `fwe` | freq_words_extraction | `english_words.json` ✔ present |
| `qa_1` | qa | SQuAD ✔ already in the HF cache |
| `qa_2` | qa | HotpotQA |

`data/synthetic/json/english_words.json` is present with its real content
(8.56 MB, 370,101 entries). Note that `raw.githubusercontent.com` serves a Git
LFS *pointer* for this file, not the data; it has to come from
`media.githubusercontent.com/media/...` or the pipeline silently builds its word
tasks out of a 132-byte pointer stub.

## Not yet fetched: the essay haystack

`data/synthetic/json/download_paulgraham_essay.py` builds `PaulGrahamEssays.json`
by scraping ~200 pages from `paulgraham.com` and
`github.com/gkamradt/LLMTest_NeedleInAHaystack`. That is a lot of requests to a
third party's personal site, so it has been left for an explicit decision rather
than run automatically. Six of the thirteen tasks need it; the other seven can
be generated now.

It also needs `html2text` and `beautifulsoup4`, which are not installed here.
