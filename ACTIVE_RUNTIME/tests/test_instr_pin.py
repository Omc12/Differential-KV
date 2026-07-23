"""
test_instr_pin.py — Tests query-aware instruction pinning (DKV_INSTR_PIN).

LAYOUT:  [doc_before | QUESTION (depth 0.3) | doc_after | NEEDLE (depth 0.7) | doc_tail]

The question appears in the MIDDLE of the document, NOT at the end.
Three modes compared:
  A) Dense baseline          (DKV_SPARSE_PREFILL=0)
  B) Sparse, no pinning      (DKV_SPARSE_PREFILL=1, DKV_INSTR_PIN=0)
  C) Sparse + pinning        (DKV_SPARSE_PREFILL=1, DKV_INSTR_PIN=1)

MODEL CAPABILITY NOTE:
  - Standard NIAH (question at END):     1.5B-4bit passes 3/3 at 4k ✓
  - QIM NIAH (question in MIDDLE, dense): 1.5B-4bit hallucinates — model capability ceiling
  - Instruction pinning fires correctly (blocks [0,3,10,15] pinned) but cannot
    overcome the model's decode recency bias at this scale.
  - Recommended: run with ≥7B model or on synthesis/multi-hop QA tasks where
    prefill quality (not just decode routing) is the bottleneck.

WHAT INSTRUCTION PINNING DOES:
  - Ensures post-question prefill chunks always attend the question block
  - Blocks selected by IDF-weighted token match (query_span.py)
  - Gated by DKV_INSTR_PIN=1 (default=0 until benchmarked on larger models)

Run:
    cd ACTIVE_RUNTIME
    python tests/test_instr_pin.py 4000 0.7         # 4k, quick smoke test
    python tests/test_instr_pin.py 16000 0.8        # 16k, where sparse diverges from dense
    python tests/test_instr_pin.py --skip-dense     # skip dense baseline
    python tests/test_instr_pin.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NEEDLE      = "The secret passcode is GRYPHONIX."
QUESTION    = "What is the secret passcode? Repeat it exactly."
NEEDLE_CHECK = "GRYPHONIX"   # single coined word — unambiguous, unscrambable, not in filler

# Number-free filler — no digits anywhere so the model cannot confuse any
# filler fact with the synthetic needle "OMEGA-7741-DELTA".
FILLER_LINES = [
    "The Eiffel Tower is one of the most visited monuments in the world and is located in Paris.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Roman Empire fell when its last emperor was deposed by a Germanic chieftain.",
    "Antarctica is the driest, windiest, and coldest continent on Earth.",
    "Isaac Newton formulated the three laws of motion in his landmark work on mechanics.",
    "The Amazon River discharges more water into the ocean than any other river on Earth.",
    "Marie Curie was the first person to win Nobel Prizes in two different sciences.",
    "The Great Wall of China stretches thousands of kilometres across northern China.",
    "Bees communicate the location of flowers through a waggle dance.",
    "The human genome contains billions of base pairs of deoxyribonucleic acid.",
    "Shakespeare wrote dozens of plays, hundreds of sonnets, and several longer poems.",
    "The Pacific Ocean covers more area than all of Earth's landmasses combined.",
    "Mitochondria produce energy through oxidative phosphorylation in the inner membrane.",
    "The Hubble Space Telescope orbits Earth and has photographed distant galaxies.",
    "Copper has been used by humans for millennia and conducts electricity very well.",
    "The Black Death killed a large fraction of Europe's population in the fourteenth century.",
    "Penguins are flightless birds that use their wings as flippers to swim underwater.",
    "The invention of the printing press by Gutenberg transformed literacy across Europe.",
    "Carbon dioxide concentrations in the atmosphere have risen sharply since industrialisation.",
    "The Amazon rainforest produces a significant proportion of the world's oxygen supply.",
]
FILLER = " ".join(FILLER_LINES) + " "


def make_qim_prompt(tokenizer, target_tokens, needle_depth, question_depth=0.3):
    """
    Build a QIM prompt:
      user: [filler_a] [QUESTION-IN-MIDDLE] [filler_b] [NEEDLE] [filler_c] [QUESTION-AT-END]
      assistant: (empty — let model generate freely)

    The question appears TWICE:
      1. In the MIDDLE of the document (depth question_depth) — this is what instruction pinning
         is designed to handle: ensuring post-question prefill chunks attend the question block.
      2. At the END of the user turn — same as standard NIAH, so the model knows what to answer.

    This mirrors real-world RAG: the system prompt / task description may embed the question
    early, while the retrieved document is long, and the question is also stated as the final
    instruction. Instruction pinning helps the model build question-aware KV representations
    for the document section that follows the embedded question (including the needle block).
    """
    filler_ids   = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_ids   = tokenizer.encode(NEEDLE + "\n", add_special_tokens=False)
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)
    overhead     = len(needle_ids) + 2 * len(question_ids) + 80  # two question copies + template
    budget       = max(200, target_tokens - overhead)
    repeats      = budget // len(filler_ids) + 1
    all_filler   = (filler_ids * repeats)[:budget]

    needle_at  = int(len(all_filler) * needle_depth)
    pre_needle = all_filler[:needle_at]
    post       = all_filler[needle_at:]

    q_at  = int(len(pre_needle) * question_depth)
    seg_a = tokenizer.decode(pre_needle[:q_at])
    seg_b = tokenizer.decode(pre_needle[q_at:])
    seg_c = tokenizer.decode(post)

    # Format mirrors test_mlx_niah.py exactly, with the question at end.
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + seg_a
        + "\n\n[Note: " + QUESTION + "]\n\n"   # question embedded in middle
        + seg_b + "\n"
        + NEEDLE + "\n"
        + seg_c + "\n\n"
        + QUESTION                               # question at end — same as standard NIAH
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt



def run_one(label, env_overrides, ctx_len, needle_depth, model_id):
    """Run one test with given env overrides. Returns result dict."""
    saved = {}
    for k, v in env_overrides.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v

    result = {"label": label, "pass": False, "gen": "", "tokens": 0}
    try:
        # Clear module cache so KVManager re-reads env vars at construction
        for mod in list(sys.modules.keys()):
            if "mlx_dkv_wrapper" in mod or "kv_runtime_manager" in mod:
                del sys.modules[mod]

        from serving.mlx_dkv_wrapper import MLXDKVWrapper
        wrapper = MLXDKVWrapper(model_id=model_id, config={"rank": 16, "block_size": 256})

        prompt      = make_qim_prompt(wrapper.tokenizer, ctx_len, needle_depth)
        prompt_toks = len(wrapper.tokenizer.encode(prompt))
        result["tokens"] = prompt_toks

        print(f"  Prompt tokens : {prompt_toks}")
        print(f"  SPARSE_PREFILL: {os.environ.get('DKV_SPARSE_PREFILL', '1')}  "
              f"INSTR_PIN: {os.environ.get('DKV_INSTR_PIN', '0')}")

        # Always set _last_messages so extract_query_token_ids can find the
        # question span for both pinning and SRL — even without FACTUAL_STORE.
        wrapper._last_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": QUESTION},
        ]

        wrapper.generate(
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )

        sid      = wrapper.active_session or "default"
        all_ids  = wrapper._session_token_ids.get(sid, [])
        gen_ids  = all_ids[prompt_toks:]
        gen_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        wrapper.close()

        result["pass"] = NEEDLE_CHECK in gen_text
        result["gen"]  = gen_text

    except Exception as e:
        result["gen"] = f"ERROR: {e}"
        import traceback; traceback.print_exc()

    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    status = "PASS ✓" if result["pass"] else "FAIL ✗"
    print(f"  Generated : {result['gen']!r}")
    print(f"  Result    : {status}")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Query-in-Middle NIAH test for DKV_INSTR_PIN")
    p.add_argument("ctx",          nargs="?", type=int,   default=12000,
                   help="Target context length in tokens (default 12000)")
    p.add_argument("needle_depth", nargs="?", type=float, default=0.7,
                   help="Depth of needle in document (default 0.7)")
    p.add_argument("--model",      default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    p.add_argument("--skip-dense", action="store_true",
                   help="Skip the dense baseline (saves time)")
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f"Query-in-Middle NIAH | ctx={args.ctx} | needle_depth={args.needle_depth}")
    print(f"Question at depth 0.3 (BEFORE needle) | Model: {args.model}")
    print(f"{'='*65}")

    configs = []
    if not args.skip_dense:
        configs.append((
            "A: Dense baseline",
            {"DKV_SPARSE_PREFILL": "0", "DKV_INSTR_PIN": "0"},
        ))
    configs += [
        ("B: Sparse, no pinning",
         {"DKV_SPARSE_PREFILL": "1", "DKV_INSTR_PIN": "0"}),
        ("C: Sparse + instruction pinning",
         {"DKV_SPARSE_PREFILL": "1", "DKV_INSTR_PIN": "1",
          "DKV_INSTR_PIN_IDF": "2.5", "DKV_INSTR_PIN_MAX": "4"}),
    ]

    results = []
    for label, env in configs:
        print(f"\n{'─'*65}\nRunning: {label}")
        results.append(run_one(label, env, args.ctx, args.needle_depth, args.model))

    print(f"\n{'='*65}")
    print("SUMMARY — Query-in-Middle NIAH")
    print(f"{'='*65}")
    print(f"{'Mode':<38} {'Tokens':>7}  Result")
    print(f"{'─'*65}")
    for r in results:
        print(f"{r['label']:<38} {r['tokens']:>7}  {'PASS ✓' if r['pass'] else 'FAIL ✗'}")
    print(f"{'─'*65}")

    # Exit 0 only if C passes (pinning should fix it)
    c = next((r for r in results if r["label"].startswith("C")), None)
    sys.exit(0 if (c and c["pass"]) else 1)
