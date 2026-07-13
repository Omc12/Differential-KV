"""Content-aware residual capture for the torch/CUDA compression path.

Pure-Python port of the three capture layers that the MLX wrapper
(serving/mlx_diffkv_wrapper.py) and the native runtime (lowrank.cpp) already
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
DIFFKV_RESIDUAL_TOKEN_BOOST (default 8), DIFFKV_RESIDUAL_OWNER_CAPTURE
(default on), DIFFKV_RESIDUAL_OWNER_DIST (12), DIFFKV_RESIDUAL_TABLE_CAPTURE
(default on), DIFFKV_RESIDUAL_TABLE_PRIORITY (4).

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


def _detect_table_rows(tok_strs):
    """Same rule as the MLX _detect_table_rows: a line is table-like when it
    has >= 2 STANDALONE separator tokens (strip == '|' or '&') among >= 3
    tokens AND table shape (starts with a separator, or >= 3 separators at
    density >= 1/12 — the density guard rejects prose with inline math |x|)."""
    S = len(tok_strs)
    marked = [False] * S
    line = []

    def _flush():
        if len(line) < 3:
            return
        seps = sum(1 for i in line if tok_strs[i].strip() in ('|', '&'))
        if seps < 2:
            return
        first = tok_strs[line[0]].strip()
        if not (first.startswith('|') or first.startswith('&')
                or '\\\\' in tok_strs[line[-1]]          # LaTeX row terminator
                or (seps >= 3 and seps * 12 >= len(line))):
            return
        for i in line:
            marked[i] = True

    for i in range(S):
        line.append(i)
        if '\n' in tok_strs[i]:
            _flush()
            line = []
    _flush()
    return marked


def compute_boost_multipliers(tok_strs, tids, counts, total_tokens):
    """Return (boost_multipliers, n_boosted) for one block's tokens.

    tok_strs: decoded surface string per token; tids: token ids; counts:
    session token-id -> count; total_tokens: session length. Returns None
    instead of a list when boosting is disabled (DIFFKV_RESIDUAL_TOKEN_BOOST
    <= 1)."""
    try:
        tok_boost = float(os.environ.get("DIFFKV_RESIDUAL_TOKEN_BOOST", "8.0"))
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

    boost = [1.0] * S
    for seg in segments:
        if any(is_core[i] for i in seg):
            for i in seg:
                boost[i] = tok_boost * (_idf_weight(tids[i], counts, total_tokens) / 2.0)

    # Owner capture (layer 1b)
    if os.environ.get("DIFFKV_RESIDUAL_OWNER_CAPTURE", "1") == "1":
        try:
            owner_dist = int(os.environ.get("DIFFKV_RESIDUAL_OWNER_DIST", "12"))
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
                boost[i] = tok_boost * (_idf_weight(tids[i], counts, total_tokens) / 2.0)

    # Table capture (layer 3)
    if os.environ.get("DIFFKV_RESIDUAL_TABLE_CAPTURE", "1") == "1":
        try:
            priority = float(os.environ.get("DIFFKV_RESIDUAL_TABLE_PRIORITY", "4.0"))
        except ValueError:
            priority = 4.0
        marked = _detect_table_rows(tok_strs)
        for i, m in enumerate(marked):
            if m:
                b = tok_boost * (_idf_weight(tids[i], counts, total_tokens) / 2.0) * priority
                boost[i] = max(boost[i], b)

    # Window pass (contiguous runs)
    final = list(boost)
    W = 2
    for i in range(S):
        if boost[i] > 1.0:
            for j in range(max(0, i - W), min(S, i + W + 1)):
                final[j] = max(final[j], boost[i])
    n_boosted = sum(1 for b in final if b > 1.0)
    return final, n_boosted
