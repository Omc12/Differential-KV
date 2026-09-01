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

## The essay haystack (fetched)

`data/synthetic/json/download_paulgraham_essay.py` builds `PaulGrahamEssays.json`
by scraping ~200 pages from `paulgraham.com` and
`github.com/gkamradt/LLMTest_NeedleInAHaystack`. Six of the thirteen tasks need
it. It was run once, on an explicit decision, and produced 3,048,891 characters
/ 525,469 words — enough haystack for contexts past 128k.

The QA tasks additionally need `json/squad.json` (SQuAD v2 dev, 4.4 MB) and
`json/hotpotqa.json` (HotpotQA dev-distractor, 61.1 MB), fetched from the
sources named in the upstream `download_qa_dataset.sh`.

None of these three data files is committed: they are large, and they are
reproducible from the vendored scripts.

## The one local modification: `data/prepare.py`

Everything else is byte-for-byte upstream. `prepare.py` carries a **local port
fix**, marked in the file with `LOCAL PORT FIX`, because upstream builds the
child-generator command as a **shell string** and that breaks three ways here:

1. `python <script>` — bare `python` on this box resolves to the Windows Store
   alias stub, which prints an ad and exits.
2. `{script}` and `{save_dir}` are unquoted, so the space in `Differential KV`
   splits the path: `can't open file 'C:\Users\USER\Desktop\Differential'`.
3. `--template "{...}"` contains **literal newlines**. Through `cmd.exe` the
   argument is torn apart and the generator runs with a collapsed template that
   never substitutes `{context}`.

(3) is the dangerous one because it does not fail. It produced **26-token
samples for a 4096-token request** — a full RULER suite of essentially empty
haystacks that every task would have scored on.

The fix launches the child with an **argument list** (`shell=False`) instead:
same script, same arguments, same values, only the launch mechanism changes.
No generator, template, task definition or scorer is touched.

A second change makes failure visible: upstream prints
`Prepare <task> with lines: N` unconditionally after its `except` block, so a
dead child reads exactly like a live one. It now counts the lines actually
written and raises if there are none.

Verified after the fix: `niah_single_1 @ 4096` on granite-4.2-8b yields 3,941 /
3,939 / 3,939 tokens (the shortfall from 4096 is RULER's own reserved
`tokens_to_generate`), with distinct needle values per sample.

`prepare.py` also needs `tenacity`, `nltk` (punkt + punkt_tab), `pyyaml`,
`wonderwords`, and `html2text`/`beautifulsoup4` for the essay downloader.
