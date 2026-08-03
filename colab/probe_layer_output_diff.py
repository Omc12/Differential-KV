#!/usr/bin/env python3
"""Per-layer DKV-vs-dense attention output diff on the failing 32k@depth0.9 prompt.

WHY THIS IS THE RIGHT INSTRUMENT NOW, AND WAS NOT BEFORE
--------------------------------------------------------
Everything upstream of the decode math is measured correct for this case:

  storage      WRITE MAP == ROUTE TRACE slot, |anc| bit-identical across repeats
  routing      needle's block at rank 0-1 of 122, kept=True, held all generation
  routing (2)  DKV_TOPK_BLOCKS=0 DKV_BLOCKS_PER_CHUNK=256 (attend ALL 122 blocks,
               num_chunks=1 so the reduction path is unchanged) -> STILL 'None'
  model        dense control, fp16, same prompts: ALL 9 PASS incl 32k@0.9 3/3

So the model can do it, the data is there and reachable, and DKV still answers
'None'. The only region left is what the decode kernel computes from it.

The earlier handoff proposed this same diff while believing the needle's block
was routed -- it was not, and that premise had to be fixed first (the block was
being dropped, and separately the decode block cache was serving the previous
generation's pool slots). Running it then would have measured a path that was
already broken upstream.

HYPOTHESIS IT TESTS
-------------------
res_max for the needle's block decays with depth while the competition does not:

    32k@0.0  s_anc 11.9-15.6   res_max 15.6-20.3   top ~16.9-22.1   pass
    32k@0.5  s_anc  2.7-6.6    res_max 13.6-19.9   top ~16.8-21.8   pass
    32k@0.9  s_anc -1.1-1.6    res_max 11.5-15.0   top  12.8-17.6   FAIL

At depth 0.9 the needle's true q.k is barely above the field, so a low-rank
reconstruction that OVERSHOOTS on non-residual (filler) tokens would steal
softmax mass from it. Dense faces the same competition and wins; the low-rank
reconstruction is the only thing DKV adds. That predicts the first big per-layer
divergence to be an attention OUTPUT that has drifted toward the filler mean.

USAGE (two processes -- one model each, this card has 7.81 GiB)
    python colab/probe_layer_output_diff.py --mode dense
    python colab/probe_layer_output_diff.py --mode dkv
    python colab/probe_layer_output_diff.py --mode compare

--mode dense needs the same chunking the dense control needs; single-shot 32k
OOMs in torch_chunk_gated_delta_rule on LINEAR-ATTENTION ACTIVATIONS (not the KV
cache). Default 512 matches what worked on this box.
"""
import argparse
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BOTH, because the two entry points in this repo disagree: mlx_needle_parity
# imports `serving.*` (ACTIVE_RUNTIME on the path) while validate_cuda_dkv
# imports `ACTIVE_RUNTIME.serving.*` (repo root on the path). Putting both on
# sys.path makes either spelling work instead of picking one and being wrong.
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

import torch

NEEDLE = "ZEBRA-4471-QUARTZ"

# verbatim from validate_cuda_dkv.py
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


def build(n_filler, depth):
    filler = [random.choice(FILLER) for _ in range(n_filler)]
    at = int(len(filler) * depth)
    needle = (f"Remember this important code: {NEEDLE}. "
              "This is the only code you need to remember.")
    parts = filler[:at] + [needle] + filler[at:]
    parts.append("Question: What was the important code mentioned in this "
                 "text? Reply with only the code.")
    return " ".join(parts)


def make_prompt(tokenizer, label, depth):
    """Replay the validator's FULL RNG sequence.

    It seeds random ONCE and builds all nine cases in order, so each case's
    filler depends on every draw before it. Building one case in isolation
    produces a DIFFERENT PROMPT -- a trap this investigation already paid for.
    """
    cases = [("2k", 200, 0.0), ("2k", 200, 0.5), ("2k", 200, 0.9),
             ("8k", 800, 0.0), ("8k", 800, 0.5), ("8k", 800, 0.9),
             ("32k", 2400, 0.0), ("32k", 2400, 0.5), ("32k", 2400, 0.9)]
    random.seed(5)
    ctx = None
    for lab, n_filler, d in cases:
        c = build(n_filler, d)
        if lab == label and abs(d - depth) < 1e-9:
            ctx = c
    assert ctx is not None, f"no case {label}@{depth}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": ctx}], tokenize=False,
        add_generation_prompt=True)


class Capture:
    """Record each attention module's output for the LAST position only.

    Hooks fire on the module, so they see DKV's monkey-patched forward too --
    module(...) still routes through nn.Module._call_impl.
    """

    def __init__(self):
        self.out = {}
        self.handles = []
        self.enabled = False

    def attach(self, model):
        layers = model.model.layers if hasattr(model, "model") else model.layers
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:                     # linear-attention layer
                continue
            self.handles.append(
                attn.register_forward_hook(self._mk(i)))
        return len(self.handles)

    def _mk(self, idx):
        def hook(_mod, _inp, output):
            if not self.enabled:
                return
            t = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(t) or t.dim() < 2:
                return
            # [B, L, H*D] -> last position, as fp32 on CPU
            self.out[idx] = t[0, -1].detach().float().cpu().clone()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _stem(mode):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"_layerdiff_{mode}.pt")


def run_dense(model_id, prompt, chunk):
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda")
    model.eval()
    cap = Capture()
    n = cap.attach(model)
    print(f"[probe] dense: hooked {n} attention layers")

    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    with torch.inference_mode():
        for i in range(0, len(ids), chunk):
            ch = ids[i:i + chunk]
            # Enable capture only on the FINAL chunk's last token -- that is the
            # position whose attention output decides the answer.
            cap.enabled = (i + len(ch) >= len(ids))
            out = model(input_ids=torch.tensor([ch], device="cuda"),
                        position_ids=torch.tensor(
                            [list(range(i, i + len(ch)))], device="cuda"),
                        past_key_values=cache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
    cap.remove()
    torch.save({"out": cap.out, "first_token": nxt,
                "decoded": tok.decode([nxt])}, _stem("dense"))
    print(f"[probe] dense first generated token: {nxt} {tok.decode([nxt])!r}")
    print(f"[probe] saved {len(cap.out)} layer vectors -> {_stem('dense')}")


def run_dkv(model_id, prompt):
    from serving.decode_config import BEST_DECODE_DEFAULTS
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("DKV_SYNC_COMPRESS", "1")
    from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    w = PyTorchDKVHFWrapper(model_id=model_id, config={"mode": "fp16"},
                            device="cuda")
    w.ensure_loaded()
    cap = Capture()
    n = cap.attach(w.model)
    print(f"[probe] dkv: hooked {n} attention layers")
    cap.enabled = True
    out = w.generate(prompt=prompt, max_new_tokens=1, temperature=0.0,
                     top_p=1.0, repetition_penalty=1.0)
    cap.remove()
    torch.save({"out": cap.out, "text": out[-80:]}, _stem("dkv"))
    print(f"[probe] dkv tail: {out[-60:]!r}")
    print(f"[probe] saved {len(cap.out)} layer vectors -> {_stem('dkv')}")


def compare():
    d = torch.load(_stem("dense"), weights_only=False)
    k = torch.load(_stem("dkv"), weights_only=False)
    da, ka = d["out"], k["out"]
    common = sorted(set(da) & set(ka))
    if not common:
        print("[probe] NO COMMON LAYERS — hooks did not fire on one side. "
              "That is a coverage failure, not a result.")
        return
    print(f"[probe] dense first token: {d.get('decoded')!r}")
    print(f"[probe] {len(common)} common layers\n")
    print(f"{'layer':>6}  {'cos':>9}  {'rel_err':>9}  {'|dense|':>9}  {'|dkv|':>9}")
    worst = None
    worst_rel = None
    for i in common:
        a, b = da[i], ka[i]
        if a.shape != b.shape:
            print(f"{i:>6}  shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).norm() / a.norm().clamp(min=1e-9)).item()
        print(f"{i:>6}  {cos:9.5f}  {rel:9.5f}  {a.norm():9.4f}  {b.norm():9.4f}")
        if worst is None or cos < worst[1]:
            worst = (i, cos)
        if worst_rel is None or rel > worst_rel[1]:
            worst_rel = (i, rel)
    # BOTH, because they catch different failures: a pure MAGNITUDE error leaves
    # cosine at 1.0 (seen while testing this probe -- a layer scaled by 0.5 read
    # cos=1.00000, rel_err=0.5). Direction and scale have to be reported
    # separately or a whole class of divergence is invisible.
    print(f"\n[probe] lowest cosine   layer {worst[0]}: {worst[1]:.5f}")
    print(f"[probe] largest rel_err layer {worst_rel[0]}: {worst_rel[1]:.5f}")
    print("[probe] READ THE FIRST BIG DROP, NOT THE WORST LAYER. Divergence")
    print("        compounds through the stack, so every layer after the first")
    print("        bad one is downstream of it and proves nothing on its own.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dense", "dkv", "compare"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--label", default="32k")
    ap.add_argument("--depth", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=512,
                    help="dense only; single-shot 32k OOMs on linear-attention "
                         "activations")
    args = ap.parse_args()

    if args.mode == "compare":
        compare()
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = make_prompt(tok, args.label, args.depth)
    ntok = len(tok(prompt).input_ids)
    nc = prompt.find(NEEDLE)
    print(f"[probe] {args.label}@depth{args.depth} — {ntok} tok, needle at "
          f"token ~{len(tok(prompt[:nc]).input_ids)}")

    if args.mode == "dense":
        run_dense(args.model, prompt, args.chunk)
    else:
        run_dkv(args.model, prompt)


if __name__ == "__main__":
    main()
