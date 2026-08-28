"""Content-aware residual capture for the torch/CUDA compression path.

Pure-Python port of the three capture layers that the MLX wrapper
(serving/mlx_dkv_wrapper.py) and the native runtime (lowrank.cpp) already
apply when ranking residual candidates — closing CUDA_TRITON_AUDIT.md C10
("the torch/CUDA path lacks the boost machinery entirely"):

  layer 1 — core-segment boost + owner capture (entity names join their
            values in the exact set; binding probe 1/6 -> 6/6 on MLX),
  layer 2 — (coverage scaffold lives in the selection fn, not here),
  layer 3 — table capture (whole table-like lines, incl. header/unit and
            row-name cells, with priority; 16k straddled-table probe
            3/6 -> 6/6 == dense on MLX, 2026-07-13).

KEEP IN SYNC with the MLX helpers (_apply_owner_capture,
_detect_table_rows, _apply_table_capture, and the inline is_core/is_prose
classification) and with lowrank.cpp's mirrored block. Env dials are shared:
DKV_RESIDUAL_TOKEN_BOOST (default 8), DKV_RESIDUAL_OWNER_CAPTURE
(default on), DKV_RESIDUAL_OWNER_DIST (12), DKV_RESIDUAL_TABLE_CAPTURE
(default on), DKV_RESIDUAL_TABLE_PRIORITY (4).

No torch/mlx imports here: callable from any backend and unit-testable on CPU.
"""
import math
import os

_OWNER_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "his", "her", "our", "your", "my", "it", "he", "she", "they", "we",
    "in", "on", "at", "of", "for", "and", "but", "or", "if", "as", "by",
    "with", "from", "to", "is", "are", "was", "were", "there", "here",
}

_PROSE_PUNCT = ('.', ',', ';', '?', '!', ':', '"', "'", '(', ')', '[', ']',
                '{', '}')


def _idf_weight(tid, counts, total_tokens):
    count = counts.get(tid, 1) if counts else 1
    idf = math.log(max(total_tokens, 2) / (count + 0.1))
    return max(1.0, min(idf, 6.0))


def _compute_idf(tid, counts, total_tokens):
    """Rarity formula: IDF(t) = ln(1 + N / count(t))"""
    count = counts.get(tid, 1) if counts else 1
    N = max(total_tokens, 1)
    return math.log(1.0 + float(N) / max(float(count), 1.0))



def _detect_table_rows(tok_strs):
    """Same rules as the MLX _detect_table_rows (keep in sync):
    1. SEPARATOR rule — >= 2 standalone '|'/'&' tokens among >= 3, plus
       shape (line-initial separator, LaTeX `\\\\` terminator, or density
       >= 1/12 with >= 3 separators);
    2. COLUMNAR rule (PDF copy-paste tables have NO pipes) — >= 2
       consecutive lines of 3..48 tokens that end in a digit-bearing token,
       carry >= 2 digit tokens, and are not prose-dominated; the line above
       the run (header) joins."""
    S = len(tok_strs)
    marked = [0.0] * S
    lines = []
    line = []
    for i in range(S):
        line.append(i)
        if '\n' in tok_strs[i]:
            lines.append(line)
            line = []
    if line:
        lines.append(line)

    def _sep_rule(ln):
        if len(ln) < 3:
            return False
        seps = sum(1 for i in ln if tok_strs[i].strip() in ('|', '&'))
        if seps < 2:
            return False
        first = tok_strs[ln[0]].strip()
        return (first.startswith('|') or first.startswith('&')
                or '\\\\' in tok_strs[ln[-1]]
                or (seps >= 3 and seps * 12 >= len(ln)))

    def _columnar_candidate(ln):
        if not (3 <= len(ln) <= 48):
            return False
        n_digit = 0
        n_prose = 0
        for i in ln:
            sc = tok_strs[i].strip()
            if any(c.isdigit() for c in sc):
                n_digit += 1
            elif sc.isalpha() and sc.islower() and len(sc) >= 3:
                n_prose += 1
        if n_digit < 2 or 2 * n_digit < n_prose:
            return False
        for i in reversed(ln):
            sc = tok_strs[i].strip()
            if not sc or all(not c.isalnum() for c in sc):
                continue
            return any(c.isdigit() for c in sc)
        return False

    # Tiered marking: separator-rule lines (explicit tables) carry weight
    # 1.0; columnar-rule lines (PDF-style tables, but also equation lines
    # and short numeric records) carry 0.5 — under saturation the explicit
    # table always outranks incidental numeric lines sharing its block
    # (measured: uniform weights let reference lists steal slots from a
    # markdown table, 6/6 -> 1/6). Headers and captions inherit their run's
    # weight.
    cand = [_columnar_candidate(ln) for ln in lines]
    fired = [0.0] * len(lines)
    for li, ln in enumerate(lines):
        if _sep_rule(ln):
            fired[li] = 1.0
            for i in ln:
                marked[i] = max(marked[i], 1.0)
        elif cand[li] and ((li > 0 and cand[li - 1]) or
                           (li + 1 < len(lines) and cand[li + 1])):
            fired[li] = 0.5
            for i in ln:
                marked[i] = max(marked[i], 0.5)
            if li > 0 and not cand[li - 1] and len(lines[li - 1]) <= 48:
                for i in lines[li - 1]:
                    marked[i] = max(marked[i], 0.5)
    # CAPTION capture: a table's IDENTITY lives in its caption ("Table 4
    # reports the kernel size ablation for ...") — prose, so without this it
    # survives only as rank-r smear, and at long ctx the decoder can find
    # exact numeric rows but not tell WHICH table they belong to (measured
    # 2026-07-13, aligned style: 6/6 at 4k -> 84.2-attractor at 16k once the
    # filler contributed competing numeric tables; dense, with captions
    # exact, stayed row-correct). Walk up to 3 lines above the first fired
    # line of each run; a line mentioning 'table'/'tab.' followed by a digit
    # nearby joins the exact set.
    if os.environ.get("DKV_RESIDUAL_TABLE_CAPTION", "1") != "1":
        return marked

    def _is_caption(ln):
        text = "".join(tok_strs[i] for i in ln).lower()
        for m in ("table", "tab."):
            j = text.find(m)
            if j >= 0 and any(c.isdigit() for c in text[j:j + len(m) + 4]):
                return True
        return False

    # Columnar runs ONLY: an aligned table's rows are anonymous numbers, so
    # the caption is its only identity anchor (without it: 84.2-attractor at
    # 16k). Separator-rule tables are self-identifying (pipes + header), and
    # capturing their captions was measured NET-NEGATIVE: with BOTH captions
    # exact the decoder fused the two tables into a chimera (markdown 6/6 ->
    # 1/6; restored by scoping captions to columnar runs).
    for li in range(len(lines)):
        if fired[li] != 0.5 or (li > 0 and fired[li - 1]):
            continue
        for up in range(1, 4):
            ui = li - up
            if ui < 0:
                break
            if _is_caption(lines[ui]):
                for i in lines[ui]:
                    marked[i] = max(marked[i], fired[li])
                break
    return marked


def compute_boost_multipliers(tok_strs, tids, counts, total_tokens,
                              query_ids=None):
    """Return (boost_multipliers, n_boosted) for one block's tokens.

    tok_strs: decoded surface string per token; tids: token ids; counts:
    session token-id -> count; total_tokens: session length. Returns None
    instead of a list when boosting is disabled (DKV_RESIDUAL_TOKEN_BOOST
    <= 1)."""
    try:
        tok_boost = float(os.environ.get("DKV_RESIDUAL_TOKEN_BOOST", "8.0"))
    except ValueError:
        tok_boost = 8.0
    if tok_boost <= 1.0:
        return None, 0

    S = len(tok_strs)
    is_core, is_prose = [], []
    for s in tok_strs:
        sc = s.strip()
        has_digit = any(c.isdigit() for c in sc)
        is_upper = sc.isupper() and sc.isalpha() and len(sc) >= 2
        is_core.append(has_digit or is_upper or sc == '-' or sc == '_')
        if not sc:
            is_prose.append(True)
        elif sc in _PROSE_PUNCT:
            is_prose.append(True)
        elif sc.isalpha() and (sc.islower() or (sc.istitle() and len(sc) > 1)):
            is_prose.append(True)
        else:
            is_prose.append(False)

    segments, in_seg = [], False
    for i in range(S):
        if not is_prose[i]:
            if not in_seg:
                in_seg = True
                segments.append([i])
            else:
                segments[-1].append(i)
        else:
            in_seg = False

    try:
        boost_digits = float(os.environ.get("DKV_BOOST_DIGITS", "20.0"))
    except ValueError:
        boost_digits = 20.0
    try:
        boost_owner = float(os.environ.get("DKV_BOOST_OWNER", "14.6"))
    except ValueError:
        boost_owner = 14.6
    try:
        boost_rare = float(os.environ.get("DKV_BOOST_RARE", "7.3"))
    except ValueError:
        boost_rare = 7.3

    boost = [1.0] * S
    for seg in segments:
        if any(is_core[i] for i in seg):
            for i in seg:
                b = max(boost_digits, tok_boost * (_idf_weight(tids[i], counts, total_tokens) / 2.0))
                boost[i] = max(boost[i], b)

    # Owner capture (layer 1b)
    if os.environ.get("DKV_RESIDUAL_OWNER_CAPTURE", "1") == "1":
        try:
            owner_dist = int(os.environ.get("DKV_RESIDUAL_OWNER_DIST", "12"))
        except ValueError:
            owner_dist = 12
        for seg in segments:
            if not any(is_core[i] for i in seg):
                continue
            j, steps, run_end = seg[0] - 1, 0, -1
            while j >= 0 and steps < owner_dist:
                sc = tok_strs[j].strip()
                if sc and sc[0].isupper() and sc.lower() not in _OWNER_STOPWORDS:
                    run_end = j
                    break
                j -= 1
                steps += 1
            if run_end < 0:
                continue
            k_hi = run_end
            while k_hi + 1 < S:
                nxt = tok_strs[k_hi + 1]
                if nxt and not nxt[0].isspace() and nxt.strip().isalpha():
                    k_hi += 1
                else:
                    break
            k_lo = run_end
            while k_lo - 1 >= 0:
                prv = tok_strs[k_lo - 1]
                pc = prv.strip()
                if prv[:1].isspace() and pc and pc[0].isupper() and pc.lower() not in _OWNER_STOPWORDS:
                    k_lo -= 1
                else:
                    break
            for i in range(k_lo, k_hi + 1):
                if boost[i] > 1.0:
                    continue
                b = max(boost_owner, tok_boost * (_idf_weight(tids[i], counts, total_tokens) / 2.0))
                boost[i] = max(boost[i], b)

    # ── QUERY-PROXIMITY CAPTURE (DKV_RESIDUAL_QUERY_CAPTURE) ─────────────────
    # Every pass above ranks by SHAPE or RARITY, which are properties of the
    # document alone. Reconstruction error is too: it asks "which tokens does the
    # low-rank basis fit worst", never "which tokens is anyone going to ask for".
    #
    # That is the whole gap against attention-selected methods. Measured on this
    # repo's own natural-text sweep, DKV scores 3/12 where dense scores 12/12,
    # and the failures track how much of the answer survived capture:
    #
    #     depth 0.50   8 of the needle's 17 tokens exact   -> correct
    #     depth 0.67   7 of 11 (the run straddles a block) -> "Falcon-942"
    #     depth 0.83   3 of 17                             -> "Falcon-947"
    #
    # Partial capture of a code is worth nothing; the model returns the right
    # word and the wrong digits.
    #
    # The query is available at compress time -- the wrapper pins it before
    # prefill -- so the block can be asked the one question that matters: does
    # any of this text answer the query? Tokens whose id appears in the query get
    # a window around them boosted, because the ANSWER sits beside the query
    # terms, not on them ("the secret passcode is X" -- the query supplies
    # "secret passcode", the block must keep X).
    #
    # This does NOT raise the budget. The same rows are spent on a set chosen for
    # relevance rather than for being hard to reconstruct.
    if (query_ids and
            os.environ.get("DKV_RESIDUAL_QUERY_CAPTURE", "0") == "1"):
        try:
            q_boost = float(os.environ.get("DKV_RESIDUAL_QUERY_BOOST", "24.0"))
        except ValueError:
            q_boost = 24.0
        try:
            q_win = int(os.environ.get("DKV_RESIDUAL_QUERY_WINDOW", "24"))
        except ValueError:
            q_win = 24
        # Content words only. Matching on stopwords would light up every block
        # equally and boost nothing in particular, which is the same failure as
        # boosting everything.
        q_set = {int(t) for t in query_ids}
        hits = [i for i, t in enumerate(tids)
                if int(t) in q_set
                and tok_strs[i].strip().lower() not in _OWNER_STOPWORDS
                and len(tok_strs[i].strip()) > 2]
        for h in hits:
            lo, hi = max(0, h - q_win), min(S, h + q_win + 1)
            for i in range(lo, hi):
                # Distance-weighted, so the tokens ADJACENT to a query term --
                # where an answer to it lives -- outrank the far end of the
                # window, and a block with no query terms is untouched.
                w = q_boost * (1.0 - abs(i - h) / float(q_win + 1))
                if w > boost[i]:
                    boost[i] = w

    # Table capture (layer 3)
    if os.environ.get("DKV_RESIDUAL_TABLE_CAPTURE", "1") == "1":
        try:
            priority = float(os.environ.get("DKV_RESIDUAL_TABLE_PRIORITY", "4.0"))
        except ValueError:
            priority = 4.0
        marked = _detect_table_rows(tok_strs)
        for i, m in enumerate(marked):
            if m:
                b = tok_boost * (_idf_weight(tids[i], counts, total_tokens) / 2.0) * priority * m
                boost[i] = max(boost[i], b)

    # ── Rarity pass — RARE PROSE WORDS (Content-Aware Selection) ─────────────
    _rarity_capture_env = os.environ.get("DKV_RARITY_CAPTURE", os.environ.get("DKV_RESIDUAL_RARITY_CAPTURE", "1"))
    if _rarity_capture_env == "1" and counts:
        try:
            _rarity_w = float(os.environ.get("DKV_RARITY_WEIGHT", os.environ.get("DKV_RESIDUAL_RARITY_WEIGHT", "1.5")))
        except ValueError:
            _rarity_w = 1.5
        try:
            _rarity_min = float(os.environ.get("DKV_RARITY_MIN_IDF", os.environ.get("DKV_RESIDUAL_RARITY_MIN_IDF", "2.0")))
        except ValueError:
            _rarity_min = 2.0
        _apply_rarity_capture(boost, tok_strs, tids, counts, total_tokens,
                              rarity_weight=_rarity_w, min_idf=_rarity_min, target_mult=boost_rare)


    # Window pass (contiguous runs)
    final = list(boost)
    W = 2
    for i in range(S):
        if boost[i] > 1.0:
            for j in range(max(0, i - W), min(S, i + W + 1)):
                final[j] = max(final[j], boost[i])
    n_boosted = sum(1 for b in final if b > 1.0)
    return final, n_boosted


def _apply_rarity_capture(boost, tok_strs, tids, counts, total_tokens,
                          rarity_weight=1.5, min_idf=2.0, target_mult=7.3):
    """Apply Rarity-Aware Residual Selection (CAD Upgrade).

    Prevents syntax delimiters from dominating L2 error and starving rare words.
    Formula:
      IDF(t) = ln(1 + N / count(t))
      score(t) = error * (1.0 + weight * IDF(t))
    Exclusion guard:
      Pure punctuation / whitespace tokens are strictly filtered (1.0x).
    """
    if not counts:
        return 0
    S = len(tok_strs)
    n_boosted = 0
    for i in range(S):
        if boost[i] > 1.0:
            continue
        sc = tok_strs[i].strip()
        if not sc or not any(c.isalnum() for c in sc):
            continue  # Exclusion guard
        idf = _compute_idf(tids[i], counts, total_tokens)
        if idf >= min_idf:
            score_mult = 1.0 + rarity_weight * idf
            boost[i] = max(boost[i], max(score_mult, target_mult))
            n_boosted += 1
    return n_boosted


def atomic_runs(tok_strs, max_len=None):
    """Contiguous spans that are worth NOTHING unless captured WHOLE.

    Returns a list of half-open ``(lo, hi)`` spans over the block's ACTIVE rows,
    i.e. the same index space as ``compute_boost_multipliers``' return value.

    WHY THIS EXISTS. Residual selection ranks tokens INDIVIDUALLY and takes the
    top ``max_residual`` of them. An answer is not an individual token: Qwen
    splits ``Falcon-9427-6183`` into eleven of them
    (``' Falcon' '-' '9' '4' '2' '7' '-' '6' '1' '8' '3'``). Ranking them one at
    a time routinely keeps a PREFIX and drops the tail, and the decoder then
    reproduces exactly what survived and invents the rest.

    Measured on this repo's natural-text needle sweep (Qwen2.5-1.5B, mid,
    block_size 256, budget 40 -- the block's residual set read straight out of
    the pool after prefill, so this is which rows were chosen, not an inference
    from the answer):

        depth 0.83  layer 13  kept ' Falcon' '-' '9' '4' '2' '7' '-',
                              dropped '6' '1' '8' '3'   -> "Falcon-9427-6137"
        depth 0.50  layer 13  kept all but the final '3' -> "Falcon-9427-6185"
        depth 0.25  layer 13  kept 5 of 11               -> "Falcon" and nothing

    Half a code is not half an answer, it is a WRONG answer with the right
    shape, so the budget spent on the surviving prefix bought nothing. A run is
    therefore all-or-nothing, and the selection that consumes these spans treats
    it that way.

    A run is a maximal span of non-prose tokens containing at least one CORE
    token (digit / all-caps / '-' / '_'), extended backwards over its OWNER --
    the capitalised word that names it, the same rule the owner-capture boost
    uses -- and merged across a single non-core separator so a hyphenated code
    stays ONE run instead of three.

    Runs longer than ``max_len`` (DKV_RESIDUAL_RUN_MAX, default 32) are DROPPED
    rather than returned: a long numeric line -- a table row, a reference list --
    would consume the whole budget as one indivisible unit and evict everything
    else. Those fall back to per-token ranking, which is what they get today.
    """
    if max_len is None:
        try:
            max_len = int(os.environ.get("DKV_RESIDUAL_RUN_MAX", "32"))
        except ValueError:
            max_len = 32

    S = len(tok_strs)
    if S == 0:
        return []

    is_core, is_prose = [], []
    for s in tok_strs:
        sc = s.strip()
        has_digit = any(c.isdigit() for c in sc)
        is_upper = sc.isupper() and sc.isalpha() and len(sc) >= 2
        is_core.append(has_digit or is_upper or sc == '-' or sc == '_')
        if not sc:
            is_prose.append(True)
        elif sc in _PROSE_PUNCT:
            is_prose.append(True)
        elif sc.isalpha() and (sc.islower() or (sc.istitle() and len(sc) > 1)):
            is_prose.append(True)
        else:
            is_prose.append(False)

    # Maximal non-prose segments, keeping only those that carry a core token.
    segs, in_seg = [], False
    for i in range(S):
        if not is_prose[i]:
            if in_seg:
                segs[-1][1] = i + 1
            else:
                segs.append([i, i + 1])
                in_seg = True
        else:
            in_seg = False
    segs = [s for s in segs if any(is_core[i] for i in range(s[0], s[1]))]
    if not segs:
        return []

    # OWNER extension. 'Falcon' is prose by shape, so the segment above starts at
    # the '-' after it -- and a code without the word that names it is as useless
    # as a word without its code. Same walk-back as _apply_owner_capture.
    try:
        owner_dist = int(os.environ.get("DKV_RESIDUAL_OWNER_DIST", "12"))
    except ValueError:
        owner_dist = 12
    for seg in segs:
        j, steps = seg[0] - 1, 0
        while j >= 0 and steps < owner_dist:
            sc = tok_strs[j].strip()
            if sc and sc[0].isupper() and sc.lower() not in _OWNER_STOPWORDS:
                seg[0] = j
                break
            if not sc or sc in _PROSE_PUNCT:
                break            # a sentence break is not an owner
            j -= 1
            steps += 1

    # Merge spans separated by at most one token: ' Falcon - 9427' tokenises with
    # the separators attached differently depending on spacing, and a code split
    # into two runs can still be captured half-and-half.
    #
    # NEVER MERGE PAST max_len. Merging is an optimisation that keeps one code
    # together; it must not be able to DESTROY a run. Unconditional merging did
    # exactly that, and it cost a needle: in dense numeric prose the needle's own
    # span chained through its neighbours into a 30+-token blob, the length
    # filter below dropped the blob whole, and the block came back with NO run
    # covering the code at all -- measured at 8k depth 0.58, capture 1-4 of 11 at
    # every layer and the answer "Falcon-942.". A merge that can delete its own
    # subject is worse than no merge.
    # A SENTENCE OR LINE BREAK IS NEVER INSIDE AN ATOMIC UNIT. Merging across one
    # chains a code into whatever numbers the next sentence opens with -- measured
    # at 8k depth 0.58, 'Falcon-9427-6183' merged forward across '.\n' into the
    # following figure's axis labels and became a 31-token run, claiming 31 of the
    # 40 slots to protect 11 tokens of answer. The gap this rule is FOR is a
    # spacing artefact ('Falcon - 9427'), never a period.
    def _breaks(i):
        s = tok_strs[i]
        return ('\n' in s) or (s.strip() in ('.', '!', '?', ';', ':'))

    merged = [segs[0]]
    for lo, hi in segs[1:]:
        prev = merged[-1]
        gap = lo - prev[1]
        joinable = (gap == 0) or (gap == 1 and not _breaks(prev[1]))
        if joinable and (max(prev[1], hi) - prev[0]) <= max_len:
            prev[1] = max(prev[1], hi)
        else:
            merged.append([lo, hi])

    # What survives the cap now is only a SINGLE segment that is itself over-long
    # -- a table row, a reference list -- which is the case the cap is actually
    # for. Those fall back to per-token ranking, as they do today.
    return [(lo, hi) for lo, hi in merged if 0 < hi - lo <= max_len]


def usable_query(query_ids, total_tokens):
    """Is this pin a QUESTION, or the whole prompt wearing a question's name?

    `serving.query_span.extract_query_token_ids` returns `list(prompt_ids)`
    verbatim when it cannot find a user turn -- "No messages supplied or no user
    role found -> full-prompt fallback" -- and `hf_dkv_wrapper` pins whatever it
    returns. So the common degenerate case is NOT an absent pin, it is a pin the
    size of the context, and every downstream consumer has to decide for itself
    whether that means anything. For residual reservation it does not: a "query"
    covering the whole document points everywhere, which is the same as pointing
    nowhere.

    That distinction is worth its own function because the two failure modes sit
    on opposite sides of it, both measured on the 8k natural needle sweep:

        real question pinned, scoped reservation      12/12
        full-prompt pin treated as a real question     3/12   <- collapses
        full-prompt pin treated as no query           11/12

    A real question is tens of tokens against a context of thousands, so the
    separation is not delicate. The threshold is a FRACTION rather than a
    constant so it holds at every context length.
    """
    if not query_ids:
        return False
    try:
        frac = float(os.environ.get("DKV_RESIDUAL_QUERY_MAX_FRAC", "0.25"))
    except ValueError:
        frac = 0.25
    if not total_tokens or total_tokens <= 0:
        return True
    return len(query_ids) < max(1.0, frac * float(total_tokens))


def rank_runs_by_query(tok_strs, tids, query_ids, runs, window=None):
    """Annotate `runs` with whether each one sits beside a QUERY term.

    Returns the same spans as ``(lo, hi, priority)``, priority 1 or 0. The
    consumer fills the residual budget in priority order first, so this decides
    WHICH WHOLE RUN gets the slots -- it never changes how many.

    WHY THIS IS NOT THE QUERY BOOST THAT DID NOT WORK. `DKV_RESIDUAL_QUERY_CAPTURE`
    multiplies a per-token score, and it could not lift a code at all: a core
    token already sits at `tok_boost * idf/2` = 24 and the query pass writes
    `if w > boost[i]`, whose maximum is also 24. It measurably moved capture and
    left recall at 3/12. Ordering whole RUNS is a different lever -- selection is
    now all-or-nothing, so the only question left is which run wins, and that is
    exactly the question a query can answer.

    The failure it targets, measured at 8k depth 0.0 with run-atomic selection on
    (the needle's own rows, read out of the pool after prefill):

        24 of 28 layers   captured 11 of 11
        L16 L18 L26 L27   captured  0 of 11

    Block 0 holds the chat template plus the paper's title, authors and a URL --
    14 runs, 63 tokens, for 40 slots. Those competitors are genuinely
    hard to reconstruct and completely irrelevant to "what is the secret
    passcode", which is the one thing reconstruction error cannot express.

    SELF-LIMITING. If more than half the block's runs look query-relevant the
    signal carries no ranking information, so it is dropped entirely. That is the
    degenerate case in production, where `_pending_query` falls back to the whole
    prompt when no question span can be extracted -- a fallback that would
    otherwise mark everything relevant and quietly randomise the order.
    """
    if not runs or not query_ids or not tids:
        return [(lo, hi) for lo, hi in runs]
    if window is None:
        try:
            window = int(os.environ.get("DKV_RESIDUAL_RUN_QUERY_WINDOW", "24"))
        except ValueError:
            window = 24

    # Content words only. Matching stopwords would light up every run equally,
    # which is the same as lighting up none.
    q = {int(t) for t in query_ids}
    S = min(len(tok_strs), len(tids))
    hits = [i for i in range(S)
            if int(tids[i]) in q
            and len(tok_strs[i].strip()) > 2
            and tok_strs[i].strip().lower() not in _OWNER_STOPWORDS]
    # No query term in THIS block is a real answer -- "the query does not point
    # here" -- so annotate with all-zero priority rather than returning bare
    # spans. The bare form means "no usable query was consulted at all", which
    # the selector treats completely differently; see the note there.
    if not hits:
        return [(lo, hi, 0) for lo, hi in runs]

    out, n_prio = [], 0
    for lo, hi in runs:
        near = any(lo - window <= h < hi + window for h in hits)
        out.append((lo, hi, 1 if near else 0))
        n_prio += int(near)
    if n_prio * 2 > len(runs):
        # The query matched more than half the block's runs, so it carries no
        # RANKING information -- but it was still a real query, and "matched too
        # much to discriminate" is not the same as "there was no query". Annotate
        # with all-zero priority so the selector reserves nothing here instead of
        # falling back to reserving by error.
        #
        # This is the case document synthesis lives in: a question about a whole
        # paper shares content words with most of it. Returning bare spans here
        # cost 13.3 -> 6.7 by re-engaging the reservation on every such block.
        return [(lo, hi, 0) for lo, hi, _p in out]
    return out

