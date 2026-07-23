"""
query_span.py — Position-agnostic query-span extraction for DKV SRL/factual-store.

Used by both mlx_dkv_wrapper.py (Apple Silicon) and hf_dkv_wrapper.py (CUDA/HF)
so that the SRL entity-binding system always receives the correct user-question
token span regardless of where the question sits in the prompt.

Background
----------
The SRL factual-store uses `current_query_tokens` to identify which entity in the
document the user is asking about and to bias decode toward the correct fact span.
If the wrong token span is given (e.g., document filler at the tail when the actual
question is in the middle), the store produces noisy / incorrect entity bindings.

The fix here is to find the LAST user-role turn's tokens via a three-stage search:

  Stage 1 (most precise)  — encode the role-wrapped turn via apply_chat_template
                            and find it in prompt_ids by right-anchored substring.
  Stage 2 (SentencePiece) — encode just the raw content (no role markers) and
                            search again. Handles tokenizers where whitespace at
                            the start of a turn changes token boundaries.
  Stage 3 (fallback)      — return the full prompt_ids. Safe: IDF filtering in
                            the SRL layer picks out distinctive tokens anyway.

Model-agnosticism
-----------------
`apply_chat_template` is a standard HuggingFace tokenizer API (transformers>=4.34,
released 2023-09). Every instruction-tuned model ships a chat template:

  Qwen 2/2.5     <|im_start|>user\\n...\\n<|im_end|>
  Llama 3.x      <|start_header_id|>user<|end_header_id|>\\n\\n...<|eot_id|>
  Mistral v0.1+  [INST]...[/INST]
  Gemma 1/2      <start_of_turn>user\\n...<end_of_turn>
  Phi-3          <|user|>\\n...<|end|>
  Command-R      <|START_OF_TURN_TOKEN|><|USER_TOKEN|>...<|END_OF_TURN_TOKEN|>

For models without a chat template, Stage 1 raises AttributeError or Exception →
falls through to Stage 2 (raw encode) → Stage 3 if needed. Always safe.

The right-anchored search finds the LAST occurrence so multi-turn conversations
correctly identify the final user question, not an earlier repeat.
"""

from typing import List, Optional


def extract_query_token_ids(
    tokenizer,
    prompt_ids: List[int],
    messages: Optional[List[dict]] = None,
) -> List[int]:
    """Return the token-id span of the LAST user-role turn in the prompt.

    Parameters
    ----------
    tokenizer : any HuggingFace-compatible tokenizer
        Must expose `.encode(text, add_special_tokens=False)` at minimum.
        `apply_chat_template` is used when available.
    prompt_ids : list[int]
        Full tokenized prompt as a flat list of integer token IDs.
    messages : list[dict] or None
        Structured conversation list, e.g.
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}].
        If None or empty, falls back to returning `prompt_ids` unchanged.

    Returns
    -------
    list[int]
        Token IDs of the last user turn.  Always a non-empty list (falls back
        to `prompt_ids` if the span cannot be located).
    """
    if not prompt_ids:
        return []

    # ── Stage 0: extract the last user message content ────────────────────────
    last_user_content: Optional[str] = None
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            if role == "user" and content:
                last_user_content = content
                break

    if not last_user_content:
        # No messages supplied or no user role found → full-prompt fallback.
        return list(prompt_ids)

    # ── Stage 1: role-wrapped encode via apply_chat_template ──────────────────
    # This produces the exact token sequence the tokenizer uses for this turn,
    # including role markers (<|im_start|>user\n…<|im_end|> etc.).
    role_wrapped_ids: Optional[List[int]] = None
    try:
        wrapped_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": last_user_content}],
            tokenize=False,
            add_generation_prompt=False,
        )
        role_wrapped_ids = _encode_no_special(tokenizer, wrapped_text)
    except Exception:
        pass  # apply_chat_template unavailable or model has no template

    if role_wrapped_ids:
        found = _right_anchored_search(prompt_ids, role_wrapped_ids)
        if found is not None:
            return found

    # ── Stage 2: raw content encode (handles SentencePiece space-merging) ─────
    # Some tokenizers produce different token IDs for the same word depending on
    # whether it is preceded by a space. Encoding the raw content text (without
    # role markers) gives a more context-agnostic sequence that is more likely to
    # match as a subsequence of the full prompt encoding.
    raw_ids: Optional[List[int]] = None
    try:
        raw_ids = _encode_no_special(tokenizer, last_user_content)
    except Exception:
        pass

    if raw_ids:
        found = _right_anchored_search(prompt_ids, raw_ids)
        if found is not None:
            return found

    # ── Stage 3: full-prompt fallback ─────────────────────────────────────────
    # Subsequence not found (tokenizer quirk, encoding mismatch, etc.).
    # Returning the full prompt is safe: IDF filtering in finalize_srl_index
    # selects only tokens with high inverse-document-frequency, so common filler
    # words are naturally suppressed even when the whole prompt is used.
    return list(prompt_ids)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_no_special(tokenizer, text: str) -> Optional[List[int]]:
    """Encode `text` without BOS/EOS, returning a plain list[int] or None."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        # HF tokenizers return a list; MLX tokenizers may return a list too.
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return ids if ids else None
    except Exception:
        return None


def _right_anchored_search(
    haystack: List[int],
    needle: List[int],
) -> Optional[List[int]]:
    """Find the RIGHT-MOST occurrence of needle in haystack.

    Returns the matched sublist, or None if not found.
    Right-anchored so that the LAST user turn is found in multi-turn prompts
    where the same question text might appear twice (e.g., after a clarification).
    """
    n, h = len(needle), len(haystack)
    if n == 0 or n > h:
        return None
    needle_t = tuple(needle)
    for start in range(h - n, -1, -1):
        if tuple(haystack[start : start + n]) == needle_t:
            return list(haystack[start : start + n])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Instruction-pinning: lightweight pre-SRL index → pinned block indices
# ─────────────────────────────────────────────────────────────────────────────

def pinned_blocks_from_prompt(
    prompt_ids: List[int],
    query_ids: List[int],
    block_size: int,
    stop_ids: Optional[set] = None,
    idf_threshold: float = 3.0,
    max_pinned_blocks: int = 4,
) -> List[int]:
    """Return sorted block indices to pin as instruction sinks during sparse prefill.

    Background
    ----------
    During block-sparse prefill, each chunk routes back to history blocks using
    its OWN query vectors.  When the user question is embedded in the middle of
    a long document, post-question chunks are routing using document-content
    queries — they have no intrinsic reason to score the question block highly.
    The result is that tokens at positions AFTER the question may never attend
    it during prefill, leaving their KV representations question-unaware.

    This function identifies document blocks whose content is distinctively tied
    to the user question, so they can be added to the always-attended set for
    all sparse-prefill chunks that come after the question position.

    Algorithm
    ---------
    1.  Build a lightweight inverted map: query_token -> set of block indices
        containing that token.  Pure-Python CPU, ~0.05 s at 25 k tokens.
    2.  IDF = log(N_blocks / n_blocks_containing_token) + 1.
        Tokens with IDF >= idf_threshold are considered distinctive.
    3.  Score each candidate block by sum of IDF scores of its matching
        query tokens.
    4.  Return top-max_pinned_blocks block indices, sorted ascending.

    Why this finds ANSWER blocks, not just the QUESTION block
    ---------------------------------------------------------
    The distinctive tokens of the question (proper nouns, identifiers, numbers)
    appear in the document at the location of the ANSWER.  Pinning by IDF-weighted
    hit count therefore pulls in the answer-bearing blocks, which is exactly what
    the model needs to build accurate representations for tokens processed after
    the question.

    Parameters
    ----------
    prompt_ids        : full tokenized prompt (flat list[int])
    query_ids         : token ids of the user question span
    block_size        : tokens per block — must match manager.block_size
    stop_ids          : set of stop-word token ids to ignore (None → empty set)
    idf_threshold     : minimum IDF to treat a query token as distinctive (3.0)
    max_pinned_blocks : hard cap on pinned blocks — default 4 = 1024 tokens at bs=256

    Returns
    -------
    Sorted list[int] of block indices (0-based).  Empty list if no distinctive
    query tokens found → caller uses normal sparse routing with no pinning.
    """
    if not prompt_ids or not query_ids or block_size <= 0:
        return []

    _stop = stop_ids or set()
    query_set = {q for q in query_ids if q not in _stop}
    if not query_set:
        return []

    import math
    from collections import defaultdict

    n_tokens = len(prompt_ids)
    n_blocks = max(1, (n_tokens + block_size - 1) // block_size)

    # Step 1: token → block index map (only track query tokens)
    blk_hits: dict = defaultdict(set)   # token_id → set[block_idx]
    for pos, tid in enumerate(prompt_ids):
        if tid in query_set:
            blk_hits[tid].add(pos // block_size)

    if not blk_hits:
        return []

    # Step 2: IDF per query token
    idf: dict = {
        tid: math.log(n_blocks / max(1, len(blks))) + 1.0
        for tid, blks in blk_hits.items()
    }

    # Step 3: score each block by IDF-weighted hits
    block_score: dict = defaultdict(float)
    for tid, blks in blk_hits.items():
        score = idf[tid]
        if score < idf_threshold:
            continue
        for blk in blks:
            block_score[blk] += score

    if not block_score:
        return []

    # Step 4: top-K, sorted ascending (so idx_parts concatenation is ordered)
    ranked = sorted(block_score, key=lambda b: -block_score[b])
    return sorted(ranked[:max_pinned_blocks])
