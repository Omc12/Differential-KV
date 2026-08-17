"""Exact recall of DIGITS from a table — the case the residual budget exists for.

WHY THIS EXISTS
---------------
`max_residual_tokens` keeps N tokens per compressed block EXACT, uncompressed, to
correct the lossy SVD. config.py calls it "the main quality dial" and ladders it
40/128/128 across the presets, at a real VRAM cost: res128 pool = 2.8 GB against
res40 = 1.5 GB at 13.4k.

Three measurements now say it changes NOTHING:

    linkbench @32k, 24 seeds, `low`, 40 vs 128     18/24 both
    linkbench @32k, 48 seeds, `mid`, 40 vs 128     21/48 both
    prose synthesis @13.4k, 40 vs 128              unchanged

But all three are PROSE, and prose is not what residuals are for. The comment in
remat_cache.py is explicit: the residuals "carry the exact values of the tokens
the SVD reconstructs worst (codes, digits)". A low-rank approximation of a block
of flowing text is a good approximation — that is why energy targets work. A
block of unrelated 4-digit numbers is close to full-rank, and the residuals are
the only thing standing between the model and a corrupted digit.

So the ladder was never tested on the content it was designed for, and lowering
`mid` on prose evidence alone would be lowering it on the wrong evidence.

WHAT IT MEASURES
----------------
A ledger of ROWS, each an unrelated code and amount:

    QX-4471   Harbour Logistics    8842 credits

The question asks for one row's amount by its code. Getting it right requires
the exact digits back, not a plausible reconstruction — "8842" and "8942" are
equally fluent and only one is a hit. Rows are digit-dense and mutually
unpredictable, so the blocks holding them are the high-rank case.

SCORING is exact string match on the amount, with the row's OWN code required to
appear too, so a model that emits a fluent wrong number scores 0.

ALWAYS RUN THE DENSE ARM. Every absolute score in this project was recorded
without a same-environment control, and when the environment later shifted, a
change that moved dense by the same amount looked like a DKV regression for an
afternoon. dense is the ruler; DKV is only ever "at parity" or "behind" relative
to a dense number taken on the same machine on the same day.

    ENGINE=dense                       python colab/tablebench_cuda.py
    ENGINE=dkv DKV_MAX_RESIDUAL_TOKENS=128 python colab/tablebench_cuda.py
    ENGINE=dkv DKV_MAX_RESIDUAL_TOKENS=40  python colab/tablebench_cuda.py

Env: MODEL, ENGINE, CTX, SEEDS, N_ROWS, MAX_NEW, DKV_PRESET, RANK, BLOCK.
"""
import os
import random
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
ENGINE = os.environ.get("ENGINE", "dkv")
CTX = int(os.environ.get("CTX", "32000"))
N_ROWS = int(os.environ.get("N_ROWS", "60"))
CHUNK = int(os.environ.get("CHUNK", "1024"))
MAX_NEW = int(os.environ.get("MAX_NEW", "24"))
SEED = int(os.environ.get("SEED", "11"))

DEPTS = ["Harbour Logistics", "Meteorology", "Signal Relay", "Archive Services",
         "Hull Inspection", "Freight Routing", "Water Reclamation",
         "Power Distribution", "Medical Supply", "Crew Rotation",
         "Beacon Maintenance", "Cargo Assay"]
PREFIXES = ["QX", "RM", "TB", "ZL", "VN", "KD", "PW", "HS"]

# Filler is PROSE, so the table rows are the only high-rank content in the
# context. If the residual budget matters anywhere it must matter here; if the
# filler were also digit-dense the effect would be spread over every block
# instead of concentrated in the ones being queried.
FILLER = ("Routine station activity continued without incident during the "
          "reporting period, and no exceptions were logged by the duty officer. ")


def build(tokenizer, seed):
    """One ledger, scattered through prose filler, plus a single-row question."""
    rng = random.Random(seed)
    codes, rows = set(), []
    while len(rows) < N_ROWS:
        code = f"{rng.choice(PREFIXES)}-{rng.randint(1000, 9999)}"
        if code in codes:
            continue
        codes.add(code)
        # Amounts are 4-digit and drawn independently, so no low-rank structure
        # across rows can reconstruct one from the others.
        rows.append((code, rng.choice(DEPTS), rng.randint(1000, 9999)))

    per_filler = len(tokenizer(FILLER).input_ids)
    n_filler = max(64, int(CTX / max(per_filler, 1)))
    body = [FILLER for _ in range(n_filler)]

    # Scatter the rows evenly across the whole context rather than as one table.
    # A contiguous table would land in a couple of blocks, so the result would
    # measure those two blocks rather than the setting.
    inserts = []
    for i, (code, dept, amt) in enumerate(rows):
        frac = (i + 0.5) / len(rows)
        pos = int(len(body) * frac) + rng.randint(-len(body) // 40, len(body) // 40)
        line = (f"\nLEDGER ENTRY {code}: department {dept}, "
                f"allocation {amt} credits.\n")
        inserts.append((max(0, min(len(body) - 1, pos)), line))
    for pos, line in sorted(inserts, key=lambda x: -x[0]):
        body.insert(pos, line)

    # Query a row from the MIDDLE. The first and last rows sit in the prefill
    # edge and the dense window respectively, and neither is compressed the way
    # the interior is -- querying them would test the paths that are exact
    # anyway and report a difference that does not exist.
    tgt = rows[len(rows) // 2]
    question = (f"\n\nQuestion: What is the allocation, in credits, of ledger "
                f"entry {tgt[0]}? Reply with only the number.")
    return " ".join(body) + question, question, tgt


class DenseWrapper:
    """Plain HF attention with chunked prefill, so the control is DKV-free."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="cuda", dtype=torch.float16).eval()

    def generate(self, prompt, max_new_tokens, **_kw):
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            past = None
            for s in range(0, ids.shape[1], CHUNK):
                out = self.model(input_ids=ids[:, s:s + CHUNK],
                                 past_key_values=past, use_cache=True)
                past = out.past_key_values
            nxt = int(out.logits[:, -1, :].argmax())
            got = [nxt]
            for _ in range(max_new_tokens - 1):
                out = self.model(input_ids=torch.tensor([[nxt]], device="cuda"),
                                 past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[:, -1, :].argmax())
                got.append(nxt)
        return self.tokenizer.decode(got, skip_special_tokens=True)


def main():
    if ENGINE == "dense":
        w = DenseWrapper()
    else:
        os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
        from serving.decode_config import BEST_DECODE_DEFAULTS
        for k, v in BEST_DECODE_DEFAULTS.items():
            os.environ.setdefault(k, v)
        from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
        cfg = {"mode": "fp16"}
        if os.environ.get("RANK"):
            cfg["rank"] = int(os.environ["RANK"])
        if os.environ.get("BLOCK"):
            cfg["micro_block_size"] = int(os.environ["BLOCK"])
        w = PyTorchDKVHFWrapper(model_id=MODEL, config=cfg, device="cuda")
        w.ensure_loaded()

    seeds = [int(x) for x in os.environ.get("SEEDS", str(SEED)).split(",")]
    hits, ntok = 0, 0
    for seed in seeds:
        body, question, tgt = build(w.tokenizer, seed)
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False,
            add_generation_prompt=True)
        ntok = len(w.tokenizer(prompt).input_ids)
        t0 = time.perf_counter()
        out = w.generate(prompt=prompt, max_new_tokens=MAX_NEW, temperature=0.0,
                         top_p=1.0, repetition_penalty=1.0)
        dt = time.perf_counter() - t0
        if isinstance(out, dict):
            out = out.get("text", str(out))
        out = str(out)
        # The DKV wrapper returns prompt+completion, the dense arm returns only
        # new tokens. Cut at the question so both are graded on the same text --
        # otherwise the echoed prompt contains the answer and DKV scores 100%.
        ans = out.split(question)[-1] if question in out else out
        want = str(tgt[2])
        hit = want in ans
        hits += bool(hit)
        print(f"  seed={seed} code={tgt[0]} want={want} hit={hit} "
              f"got={ans.strip()[:40]!r} ({dt:.0f}s)", flush=True)
    print(f"TABLEBENCH engine={ENGINE} model={MODEL} ctx={ntok} rows={N_ROWS} "
          f"residual={os.environ.get('DKV_MAX_RESIDUAL_TOKENS', 'preset')} "
          f"HITS={hits}/{len(seeds)}", flush=True)


if __name__ == "__main__":
    main()
