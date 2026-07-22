"""
Shared helpers for the DKV multi-engine context-length benchmark.

The only job of this module (when imported by the orchestrator) is to build a
single, deterministic Needle-In-A-Haystack (NIAH) chat prompt whose total token
count, measured by the Qwen2.5 tokenizer, equals a requested target. The SAME
prompt text is fed verbatim to every engine in "raw" mode (no further chat
templating), so prompt-token counts are identical across engines (each engine's
own tokenizer may differ by a handful of tokens; we record the actual count per
engine).

No model weights are loaded here — only the (cached) Qwen2.5 tokenizer.
"""

import os

# Canonical HF model id whose tokenizer defines the reference token count.
REF_TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# The needle: a unique, unguessable fact placed at ~50% depth in the haystack.
# Recording whether the generation reproduces the passcode is a free correctness
# signal (not the focus of this perf benchmark, but useful for the writeup).
NEEDLE_PASSCODE = "OMEGA-7741-DELTA"
NEEDLE_SENTENCE = (
    f" Important: the secret passcode hidden in this document is {NEEDLE_PASSCODE}. "
    "Remember it. "
)
QUESTION = (
    "\n\nQuestion: First, state the secret passcode hidden in the document above. "
    "Then write a thorough, multi-paragraph explanation of consistency models in "
    "distributed systems, covering linearizability, sequential consistency, causal "
    "consistency, and eventual consistency, with concrete examples for each. Be "
    "detailed and keep writing."
)

# Deterministic technical filler. Repeated to build the haystack.
FILLER_PARAGRAPH = (
    "In the study of distributed systems, consistency models define the contract "
    "between a data store and the processes that read from and write to it. A "
    "linearizable system behaves as if every operation takes effect atomically at "
    "some instant between its invocation and its response, which makes reasoning "
    "about correctness straightforward but is expensive to implement at scale. "
    "Weaker models such as causal consistency, eventual consistency, and "
    "read-your-writes trade global ordering guarantees for lower latency and higher "
    "availability under network partitions. Engineers select a consistency model by "
    "weighing the application's tolerance for stale reads against its throughput and "
    "latency requirements, and by considering how conflicts will be detected and "
    "resolved when concurrent writes diverge. "
)


def _load_ref_tokenizer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from mlx_lm.utils import load_tokenizer
        return load_tokenizer("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(REF_TOKENIZER_ID, local_files_only=True)


def build_niah_prompt(target_tokens, tokenizer=None, tolerance=24):
    """Return (prompt_text, actual_token_count) for a chat prompt that tokenizes
    to within `tolerance` tokens of `target_tokens` under the Qwen2.5 tokenizer.

    The needle is inserted at ~50% depth. The chat template (with an assistant
    generation prompt) is applied exactly once here; downstream engines must
    consume the text raw.
    """
    if tokenizer is None:
        tokenizer = _load_ref_tokenizer()

    def n_tokens(text):
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def templ(content):
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    # Fixed overhead: chat template wrapper + needle + question.
    overhead = n_tokens(templ(NEEDLE_SENTENCE + QUESTION))
    filler_tok = n_tokens(FILLER_PARAGRAPH)

    budget = target_tokens - overhead
    if budget < filler_tok:
        raise ValueError(
            f"target_tokens={target_tokens} too small (overhead={overhead}, "
            f"one filler paragraph={filler_tok})"
        )

    reps = max(1, budget // filler_tok)

    def make(reps):
        half = reps // 2
        body = FILLER_PARAGRAPH * half + NEEDLE_SENTENCE + FILLER_PARAGRAPH * (reps - half)
        return templ(body + QUESTION)

    # Converge by adjusting paragraph repetitions, then fine-tune with whitespace.
    text = make(reps)
    for _ in range(64):
        cur = n_tokens(text)
        diff = target_tokens - cur
        if abs(diff) <= tolerance:
            break
        step = max(1, abs(diff) // max(1, filler_tok))
        reps += step if diff > 0 else -step
        if reps < 1:
            reps = 1
            break
        text = make(reps)

    # Fine-tune to the target with single filler words (token-granular).
    cur = n_tokens(text)
    if cur < target_tokens - tolerance:
        # pad with short tokens inside the body region
        pad_word = " systems"
        pad_n = n_tokens(pad_word) or 1
        extra = (target_tokens - cur) // pad_n
        half = reps // 2
        body = (FILLER_PARAGRAPH * half + pad_word * extra + NEEDLE_SENTENCE
                + FILLER_PARAGRAPH * (reps - half))
        text = templ(body + QUESTION)

    return text, n_tokens(text)


if __name__ == "__main__":
    # Quick self-check.
    tok = _load_ref_tokenizer()
    for t in (4096, 8192, 16384):
        txt, n = build_niah_prompt(t, tok)
        print(f"target={t:>7}  actual={n:>7}  chars={len(txt)}")
