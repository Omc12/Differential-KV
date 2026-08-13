"""Multi-hop linking at long context: DKV vs dense.

WHY THIS EXISTS
---------------
multifact_eval_cuda saturates. At 16k on Qwen3.5-2B it scores 9/15 facts and 3/5
links whether the model is fed dense full attention or DKV's routed subset, and
it scores the same at every edge-propagation strength from beta 0.25 to 4.0 --
even when the propagation rewrites 9 of 12 routing decisions. The ceiling there
is the MODEL, not retrieval, and dense never fails, so DKV can at best tie.

A retrieval architecture can only beat dense where dense degrades. That means:

  * LONG context, so dense attention is spread thin over many tokens.
  * MULTI-HOP facts, so the answer is not a single lexical match the model can
    find by looking in one place. "Dr. Quillfeather -> 4193" is one hop and both
    engines get it. "Quillfeather works at Kestrel; Kestrel is in Fairhaven;
    which city does Quillfeather work in?" needs two, and the two facts are
    thousands of tokens apart.
  * DISTRACTORS, so a model cannot score by guessing the only number present.

WHAT IT MEASURES
----------------
Chains of the form A -> B -> C (2 hops) and A -> B -> C -> D (3 hops), with the
links deliberately scattered: each link of a chain is placed in a different
region of the context, so no single retrieved block contains the answer. The
question names A and asks for the last element. Grading is exact containment of
the target token, which is a unique invented word, so a near-miss cannot pass.

Distractor chains use the same shape and vocabulary, so the answer cannot be
recovered from surface form.

ARMS
----
    ENGINE=dkv    the wrapper (routing, compression, graph)
    ENGINE=dense  plain HF attention, chunked prefill, no DKV loaded

Both arms see byte-identical prompts, built from one seeded generator, so a
difference is the engine and not the data.

USAGE
    ENGINE=dkv   CTX=32000 HOPS=2 python colab/linkbench_cuda.py
    ENGINE=dense CTX=32000 HOPS=2 python colab/linkbench_cuda.py
"""
import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
ENGINE = os.environ.get("ENGINE", "dkv")
CTX = int(os.environ.get("CTX", "32000"))
HOPS = int(os.environ.get("HOPS", "2"))
N_CHAINS = int(os.environ.get("N_CHAINS", "6"))
N_DISTRACT = int(os.environ.get("N_DISTRACT", "10"))
QUANT = os.environ.get("QUANT", "")
CHUNK = int(os.environ.get("CHUNK", "1024"))
MAX_NEW = int(os.environ.get("MAX_NEW", "160"))
SEED = int(os.environ.get("SEED", "11"))

# Invented, single-sense tokens. Real words would let the model answer from
# world knowledge instead of from the context, which is the thing being tested.
PEOPLE = ["Quillfeather", "Braxanible", "Morrowind", "Vantablack", "Ashgrove",
          "Pellucid", "Windermere", "Thornbury", "Calloway", "Ravensmoor",
          "Halthorne", "Underwold", "Fenwick", "Silbrook",
          "Marchetti", "Oakhurst"]
ORGS = ["Kestrel", "Halcyon", "Verdigris", "Sableton", "Ironvale", "Nightjar",
        "Coppermill", "Larkspur", "Duskwater", "Brightmoor", "Fernhollow",
        "Ambergate", "Stonecroft", "Wexford", "Mallowdeep", "Ryecliff"]
CITIES = ["Fairhaven", "Portwick", "Glassmere", "Ashford", "Bellcastle",
          "Northgate", "Riverton", "Summerlin", "Cadwick", "Millbrook",
          "Thornfield", "Eastmere", "Highwater", "Larkhill", "Oldbridge", "Westvale"]
SECTORS = ["Meridian", "Solstice", "Equinox", "Zenith", "Perigee", "Apogee",
           "Nadir", "Vertex", "Aphelion", "Syzygy", "Penumbra", "Antumbra",
           "Umbral", "Nodal", "Azimuth", "Declin"]

FILLER = [
    "The morning fog rolled over the hills before the sun broke through the clouds.",
    "Researchers published a new dataset covering climate trends across five continents.",
    "The old library smelled of dust and aging paper, a comfort to regular visitors.",
    "Markets fluctuated throughout the week as investors weighed new economic data.",
    "A gentle breeze carried the scent of pine through the quiet mountain trail.",
    "The committee reviewed dozens of proposals before selecting a final design.",
    "Local farmers reported a strong harvest season despite the unpredictable weather.",
    "The orchestra rehearsed late into the evening, perfecting the final movement.",
]


def build(tokenizer):
    """Return (prompt, question, answer) with the chain's links scattered.

    Each link of the target chain goes into a DIFFERENT region of the context, so
    no single block holds two of them and no single retrieval can shortcut the
    hop. Distractor chains are laid down the same way and interleaved, so the
    regions are not identifiable by density.
    """
    rng = random.Random(SEED)
    people = rng.sample(PEOPLE, N_CHAINS + N_DISTRACT)
    orgs = rng.sample(ORGS, N_CHAINS + N_DISTRACT)
    cities = rng.sample(CITIES, N_CHAINS + N_DISTRACT)
    sectors = rng.sample(SECTORS, N_CHAINS + N_DISTRACT)

    def chain_sentences(i):
        s = [f"{people[i]} works at the {orgs[i]} Institute.",
             f"The {orgs[i]} Institute is located in {cities[i]}."]
        if HOPS >= 3:
            s.append(f"{cities[i]} belongs to the {sectors[i]} sector.")
        return s

    target = 0
    answer = cities[target] if HOPS == 2 else sectors[target]
    question = (f"Question: {people[target]} works at an institute. "
                + ("Which city is that institute located in? "
                   if HOPS == 2 else
                   "Which sector is that institute's city in? ")
                + "Reply with only the name.")
    # DIRECT mode names the intermediate entity outright, collapsing the chain to
    # a single lookup over the SAME context. It separates "the second fact is
    # unreachable" from "both facts are reachable but the model cannot chain
    # them" -- the two have completely different fixes, and the multi-hop result
    # alone cannot tell them apart.
    if os.environ.get("QMODE") == "direct":
        answer = cities[target]
        question = (f"Question: Which city is the {orgs[target]} Institute "
                    f"located in? Reply with only the name.")

    # Enough filler to reach CTX tokens; measured once on the real tokenizer
    # rather than guessed, because a prompt that lands short would silently make
    # this a short-context test.
    per_filler = len(tokenizer(FILLER[0]).input_ids)
    n_filler = max(64, int(CTX / max(per_filler, 1)))
    body = [rng.choice(FILLER) for _ in range(n_filler)]

    # Scatter: link h of chain i goes near fraction (h+1)/(hops+1) of the way
    # through, jittered, so the hops are far apart and not co-located.
    inserts = []
    for i in range(N_CHAINS + N_DISTRACT):
        for h, sent in enumerate(chain_sentences(i)):
            frac = (h + 1) / (HOPS + 1.0)
            pos = int(len(body) * frac) + rng.randint(-len(body) // 12, len(body) // 12)
            inserts.append((max(0, min(len(body) - 1, pos)), sent))
    for pos, sent in sorted(inserts, key=lambda x: -x[0]):
        body.insert(pos, sent)

    # Candidate pool is what a correct answer competes against: the same field
    # the answer was drawn from, so "first candidate named" is a fair test.
    pool = cities if HOPS == 2 else sectors
    return (" ".join(body) + "\n\n" + question, question, answer,
            pool[:N_CHAINS + N_DISTRACT])


class DenseWrapper:
    """Plain HF attention with chunked prefill, so the control is DKV-free."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        kw = {}
        if QUANT == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            kw["dtype"] = torch.bfloat16
        else:
            kw["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda", **kw)
        self.model.eval()

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
        if QUANT == "nf4":
            os.environ["DKV_QUANTIZATION"] = "nf4"
        from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
        w = PyTorchDKVHFWrapper(model_id=MODEL,
                                config={"mode": "nf4" if QUANT == "nf4" else "fp16"},
                                device="cuda")
        w.ensure_loaded()

    body, question, answer, candidates = build(w.tokenizer)
    prompt = w.tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                             tokenize=False, add_generation_prompt=True)
    ntok = len(w.tokenizer(prompt).input_ids)
    t0 = time.perf_counter()
    out = w.generate(prompt=prompt, max_new_tokens=MAX_NEW, temperature=0.0,
                     top_p=1.0, repetition_penalty=1.0)
    dt = time.perf_counter() - t0
    if isinstance(out, dict):
        out = out.get("text", str(out))
    # Grade on ATTRIBUTION, not presence. A model that simply echoes context can
    # emit the right name by accident -- the dense arm was observed reciting
    # filler with the answer buried in it. Requiring the FIRST candidate name in
    # the response to be the correct one tests that the model connected the hops,
    # and a wrong-but-confident answer scores 0 instead of passing on a substring.
    # Isolate the COMPLETION first. The DKV wrapper returns prompt+completion
    # while the dense arm returns only new tokens, so scanning the raw string
    # would read candidate names out of DKV's echoed prompt and score it against
    # the distractors it was merely quoting -- the two arms would not be graded
    # on the same text. The question is the last thing in the prompt, so whatever
    # follows its final occurrence is the model's own output.
    ans = out.split(question)[-1] if question in out else out
    low = ans.lower()
    firsts = [(low.find(c.lower()), c) for c in candidates if c.lower() in low]
    firsts = [f for f in firsts if f[0] >= 0]
    said = min(firsts)[1] if firsts else None
    hit = (said is not None and said.lower() == answer.lower())
    print(f"LINKBENCH engine={ENGINE} model={MODEL} ctx={ntok} hops={HOPS} "
          f"answer={answer} said={said} hit={hit} ({dt:.0f}s)", flush=True)
    print(f"  tail={out[-120:]!r}", flush=True)


if __name__ == "__main__":
    main()
