import os
import sys
import time
import math
import re

# DIFFKV_SPARSE_BIAS — additive bias (in nats) on the compressed/sparse half's LSE in the
# sparse⊕dense flash merge. Default 0.0 (exact flash attention; required for the parity
# oracle). The low-rank reconstruction systematically UNDER-scores the compressed pool, so
# when the answer lives in OLD (compressed) context that is out-competed by the recent exact
# dense window, the model reads the wrong region. Concretely: MLX compressed multi-fact
# synthesis@8k summarized the recent filler (Pride & Prejudice) instead of the buried paper —
# the sparse half attended the correct paper tokens but LOST the merge (proven 2026-07-04 via
# forced-half + DIFFKV_DBG_LSE_SHARE). A +2.0 bias tips the blend to the paper (filler→paper
# summary) while NIAH stays 4/4 (bench 4k–32k) + 3/3 (16k depths 0.1/0.5/0.9) and relational
# stays 4/4 — NIAH's exact needle residual has a large margin the bias doesn't erode. +4.0
# DOES break NIAH (needle corruption), so a fixed value is narrow → use "auto".
# It does NOT fix 16k synthesis (there the router selects filler blocks, so the paper isn't in
# the sparse half at all — a separate routing problem). See HANDOFF_MLX_SYNTHESIS.md.
#
# MODES:
#   "0.0"        (default) → exact flash merge, parity-safe no-op.
#   "<float>"    → fixed additive bias (e.g. "2.0"); tuned per model/ctx, narrow safe window.
#   "auto[,base]"→ ADAPTIVE (recommended). Applies `base` (default 2.0) when the sparse pool is
#                  competitive and DECAYS it to 0 as the dense half pulls ahead, so it boosts
#                  synthesis (sparse-competitive) yet leaves NIAH untouched (the exact needle
#                  residual makes lse_dense dominate → bias→0). Verified 2026-07-04: NIAH
#                  forced-sparse 3/3 (4k/16k/32k) + synthesis@8k reads the PAPER. Formula:
#                  bias = max(0, base − 0.5·max(0, (lse_dense−lse_sparse) − 4)).
_SPARSE_BIAS_ENV = os.environ.get("DIFFKV_SPARSE_BIAS", "0.0").strip().lower()
if _SPARSE_BIAS_ENV.startswith("auto"):
    _SPARSE_BIAS_MODE = "auto"
    _parts = _SPARSE_BIAS_ENV.split(",")
    try:
        _SPARSE_BIAS_BASE = float(_parts[1]) if len(_parts) > 1 and _parts[1] else 2.0
    except ValueError:
        _SPARSE_BIAS_BASE = 2.0
    _SPARSE_BIAS = 0.0
else:
    _SPARSE_BIAS_MODE = "fixed"
    _SPARSE_BIAS_BASE = 0.0
    try:
        _SPARSE_BIAS = float(_SPARSE_BIAS_ENV)
    except ValueError:
        _SPARSE_BIAS = 0.0

from collections import Counter
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load as mlx_load
import torch

def _normalize_references(text: str) -> str:
    """Normalise citation-list formatting inconsistencies produced by the model."""
    lines = text.split('\n')
    
    # 1. Search for a reference header line
    header_re = re.compile(r'\b(references?|bibliography|works\s+cited|reference\s+list|sources|citations)\b', re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if len(line) <= 100 and header_re.search(line):
            header_idx = i
    
    # 2. Find matching reference entries
    ref_entry_re = re.compile(r'^(?:[iI]n\s+)?(?:[*\-•]\s*)?\[\d+\]')
    unambiguous_re = re.compile(r'^(?:[*\-•]\s*)?\[\d+\]')
    
    matching_indices = []
    unambiguous_indices = []
    for i, line in enumerate(lines):
        if header_idx is not None and i <= header_idx:
            continue
        stripped = line.strip()
        if ref_entry_re.match(stripped):
            matching_indices.append(i)
            if unambiguous_re.match(stripped):
                unambiguous_indices.append(i)
                
    if header_idx is not None and not matching_indices:
        matching_indices = []
        unambiguous_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ref_entry_re.match(stripped):
                matching_indices.append(i)
                if unambiguous_re.match(stripped):
                    unambiguous_indices.append(i)
        header_idx = None

    if not matching_indices:
        return text
        
    if header_idx is not None:
        ref_start_idx = header_idx + 1
    elif unambiguous_indices:
        ref_start_idx = unambiguous_indices[0]
    else:
        return text
                
    body = '\n'.join(lines[:ref_start_idx])
    ref_block = '\n'.join(lines[ref_start_idx:])
    
    pattern = re.compile(
        r'^\s*'
        r'(?:[iI]n\s+)?'
        r'(?:[*\-•]\s*)?'
        r'(\[\d+\])'
        r'(?:,\s*|\.\s*|\s+)?',
        re.MULTILINE
    )
    normalized_ref_block = pattern.sub(r'\1 ', ref_block)
    
    if body:
        return body + '\n' + normalized_ref_block
    return normalized_ref_block

class MLXCompressedBlock:
    def __init__(self, anchor_idx: int, token_indices: List[int], U: mx.array, V_K: mx.array, V_V: mx.array, anchor_k: mx.array, anchor_v: mx.array, scale: float, seq_len: int):
        self.anchor_idx = anchor_idx
        self.token_indices = token_indices
        self.U = U                  # mx.array [S_comp, R]
        self.V_K = V_K              # mx.array [H_kv, R, D]
        self.V_V = V_V              # mx.array [H_kv, R, D]
        self.anchor_k = anchor_k    # mx.array [H_kv, D]
        self.anchor_v = anchor_v    # mx.array [H_kv, D]
        self.scale = scale          # float
        self.seq_len = seq_len      # int
        
    def clone(self):
        return MLXCompressedBlock(
            self.anchor_idx,
            self.token_indices.copy(),
            self.U,
            self.V_K,
            self.V_V,
            self.anchor_k,
            self.anchor_v,
            self.scale,
            self.seq_len
        )

def _capture_policy_env() -> Tuple[bool, float]:
    """Experimental residual-capture policies (env-gated, both default OFF).

    DIFFKV_RES_V_ONLY=1 — keep the JOINT error ranking, but store
    SVD-reconstructed K (not exact K) for the captured rows; V stays exact.
    K reconstructs at ~1% error while V is 25-70% (2026-07-02 session §2.4),
    so exact residual K should be dispensable; if recall holds, res_k storage
    can be dropped entirely (recomputed at decode), doubling the V-residual
    budget at equal memory.
    NOTE (measured 2026-07-03): ranking by V-error ALONE was also tried and
    REJECTED — easy-NIAH@4k fell 3/3 → 1/3 ("OMG"/"OCTOPUS" confabulations).
    V error is ubiquitous across rows, so the discriminative capture signal
    lives in the K half of the joint error; do not resurrect V-only ranking
    without new evidence.

    DIFFKV_RESIDUAL_COVERAGE_FRAC=f — reserve round(f*max_residual) slots for
    stride-stratified coverage of the block, so ranked capture is never fully
    zero-sum (the boost-displacement failure mode: boosting needle digits
    evicted the adjacent 'TA' row at 16k/0.9).
    """
    res_v_only = os.environ.get("DIFFKV_RES_V_ONLY", "0") == "1"
    try:
        cov_frac = float(os.environ.get("DIFFKV_RESIDUAL_COVERAGE_FRAC", "0"))
    except ValueError:
        cov_frac = 0.0
    return res_v_only, cov_frac


_COV_BONUS_CACHE: Dict[Tuple[int, int], mx.array] = {}

def _coverage_bonus(S_comp: int, max_res: int, cov_frac: float) -> Optional[mx.array]:
    """(S_comp,) float32 vector with +1e12 at the stratified coverage columns,
    or None when coverage is off. Adding it before the top-k argsort forces
    those columns into the residual set while ranking the rest normally."""
    if cov_frac <= 0.0 or max_res <= 0:
        return None
    n_cov = min(max_res, max(1, int(round(cov_frac * max_res))))
    key = (S_comp, n_cov)
    got = _COV_BONUS_CACHE.get(key)
    if got is None:
        cols = np.unique(np.round(np.linspace(0, S_comp - 1, n_cov)).astype(int))
        bonus = np.zeros((S_comp,), dtype=np.float32)
        bonus[cols] = 1e12
        got = _COV_BONUS_CACHE[key] = mx.array(bonus)
    return got


# Sentence-initial capitalized function words that must never be mistaken for an
# entity name by the owner-capture walk.
_OWNER_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "his", "her", "our", "your", "my", "it", "he", "she", "they", "we",
    "in", "on", "at", "of", "for", "and", "but", "or", "if", "as", "by",
    "with", "from", "to", "is", "are", "was", "were", "there", "here",
}


def _apply_owner_capture(boost_multipliers, segment_indices, is_core, tok_strs,
                         tids, counts, total_tokens, tok_boost):
    """Relational-locality capture (DIFFKV_RESIDUAL_OWNER_CAPTURE, default ON).

    A fact's exact residuals preserve its VALUE — digits/identifiers are
    is_core and win boosted slots — but not its OWNER: entity names are
    title-case, classified is_prose, and never boosted, so they survive only
    as rank-r reconstruction. The binding probe (2026-07-12, 8k, 6 planted
    entity→value pairs) showed the consequence: dense list-all 5/6 correct vs
    compressed 1/6, with real planted VALUES bound to CORRUPTED names
    ("Okazaki"→"Okinawa"/"Okapi", "Brancusi"→"Bruckner", value→entity 1/4 and
    0/4) while entity→value stayed 4/4 — the values are exact, the owners are
    smeared, and the association dies with the owner's surface form.

    Fix: for every core segment, walk LEFT up to DIFFKV_RESIDUAL_OWNER_DIST
    (default 12) tokens to the nearest capitalized word that isn't a
    sentence-initial function word, then expand it to the full surface run —
    its subword continuations to the right (Qwen BPE: continuations lack the
    leading space) and any preceding capitalized words of a multi-word name —
    and give those rows the same idf-weighted boost the fact rows get. The
    owner then competes for exact-residual slots alongside its value.
    Returns the number of newly boosted rows."""
    import math
    if os.environ.get("DIFFKV_RESIDUAL_OWNER_CAPTURE", "1") != "1":
        return 0
    try:
        owner_dist = int(os.environ.get("DIFFKV_RESIDUAL_OWNER_DIST", "12"))
    except ValueError:
        owner_dist = 12
    S = len(tok_strs)
    n_boosted = 0
    for seg in segment_indices:
        if not any(is_core[i] for i in seg):
            continue
        j = seg[0] - 1
        steps = 0
        run_end = -1
        while j >= 0 and steps < owner_dist:
            sc = tok_strs[j].strip()
            if sc and sc[0].isupper() and sc.lower() not in _OWNER_STOPWORDS:
                run_end = j
                break
            j -= 1
            steps += 1
        if run_end < 0:
            continue
        # Right: subword continuations complete the surface form (an exact
        # " Ok" next to a lossy "az"+"aki" still decodes as garbage).
        k_hi = run_end
        while k_hi + 1 < S:
            nxt = tok_strs[k_hi + 1]
            if nxt and not nxt[0].isspace() and nxt.strip().isalpha():
                k_hi += 1
            else:
                break
        # Left: preceding capitalized words of a multi-word name.
        k_lo = run_end
        while k_lo - 1 >= 0:
            prv = tok_strs[k_lo - 1]
            pc = prv.strip()
            if prv[:1].isspace() and pc and pc[0].isupper() and pc.lower() not in _OWNER_STOPWORDS:
                k_lo -= 1
            else:
                break
        for i in range(k_lo, k_hi + 1):
            if boost_multipliers[i] > 1.0:
                continue
            count = counts.get(tids[i], 1)
            idf = math.log(max(total_tokens, 2) / (count + 0.1))
            rarity_weight = max(1.0, min(idf, 6.0))
            boost_multipliers[i] = tok_boost * (rarity_weight / 2.0)
            n_boosted += 1
    return n_boosted


# ── Relational EDGE vocabulary ────────────────────────────────────────────────
# The closed-class connectives that carry an EDGE between two concepts and whose
# meaning FLIPS under low-rank smear. Owner/content/table capture pin the NODES
# (entities, values, cells); these tokens are the EDGES the SVD pool otherwise
# reconstructs into a nearby-but-wrong relation ("reduces"→"increases",
# "without"→"with", "approaches … as … grows"→"larger receptive field").
#
# STRONG tier = negation + limiting/asymptotic + equivalence/identity. These are
# the hardest meaning-flippers and are RARE in filler prose, so they are captured
# even in a pure-prose block with no other captured node (tightly capped). WEAK
# tier = causal/conditional/logical connectives, which are common everywhere, so
# they are captured ONLY when they sit next to already-captured content (else a
# paragraph of filler would spend its whole residual budget on "as"/"so"/"if").
_EDGE_STRONG = frozenset({
    # negation / exclusion
    "not", "no", "without", "never", "cannot", "nor", "neither", "none", "non",
    "unlike", "except", "unless", "instead", "rather",
    # limiting / comparative / asymptotic
    "approaches", "approach", "approaching", "converges", "converge",
    "converging", "tends", "tend", "tending", "asymptotically", "asymptotic",
    "bounded", "unbounded", "grows", "grow", "growing", "increases", "increase",
    "increasing", "decreases", "decrease", "decreasing", "larger", "smaller",
    "greater", "less", "fewer", "exceeds", "exceed", "approximately",
    "proportional", "inversely", "monotonically", "monotonic", "vanishes",
    "vanish", "diverges", "diverge", "saturates", "saturate", "plateaus",
    "linearly", "exponentially", "sublinear", "superlinear", "quadratic",
    # equivalence / identity / reduction
    "equals", "equal", "equivalent", "equivalently", "identical", "becomes",
    "become", "becoming", "reduces", "reduce", "reducing", "recovers",
    "recover", "corresponds", "correspond", "coincides", "coincide", "matches",
    "iff", "namely", "precisely", "exactly", "generalizes", "generalize",
    "specializes", "collapses", "collapse",
})
_EDGE_WEAK = frozenset({
    "because", "causes", "cause", "causing", "caused", "leads", "lead",
    "leading", "results", "result", "resulting", "due", "therefore", "thus",
    "hence", "enables", "enable", "enabling", "prevents", "prevent",
    "preventing", "requires", "require", "requiring", "implies", "imply",
    "implying", "yields", "yield", "yielding", "since", "if", "when",
    "whenever", "while", "whereas", "as", "only", "then", "so", "although",
    "though", "despite", "however", "otherwise",
})


def _edge_tier(s):
    """2 = strong (meaning-flipper), 1 = weak (common causal/conditional),
    0 = not a relational connective. Matches the stripped, lower-cased token."""
    w = s.strip().lower().strip(".,;:!?()[]{}\"'")
    if not w:
        return 0
    if w in _EDGE_STRONG:
        return 2
    if w in _EDGE_WEAK:
        return 1
    return 0


def _apply_relational_capture(boost_multipliers, tok_strs, tids, counts,
                              total_tokens, tok_boost):
    """Relational EDGE capture (DIFFKV_RESIDUAL_EDGE_CAPTURE, default ON).

    **The edge-fidelity problem.** Residual capture pins the NODES of a document
    (digits/identifiers via is_core, entity names via owner-capture, cells via
    table-capture) as EXACT K/V, but the connectives that BIND those nodes —
    relational verbs, negations, limit/equivalence markers — are lowercase prose,
    never captured, and survive only as rank-r reconstruction. SVD discards the
    low-energy directions that separate "reduces"↔"increases" and "with"↔
    "without", so the reconstructed key/value is a smear and the small decoder
    fills the edge from its prior: a limiting condition ("X approaches Y AS N
    grows") flattens into a generic claim ("larger receptive field"). Because the
    router scores blocks by their EXACT residual keys, an uncaptured connective is
    also weak in the routing signature — its block is less likely to even be
    selected when the query is about that relation. Nodes survive; edges rebound.

    **Fix.** Give the relational connectives the same exact-residual treatment,
    but scaled to sit BELOW the content they connect so they never displace a
    value or an owner (edge_scale < 1). STRONG connectives (negation / limit /
    equivalence) are the meaning-flippers and are rare in filler, so they are
    captured even with no other node in the block; WEAK connectives (common
    causal/conditional words) are captured only within DIFFKV_EDGE_RADIUS of an
    already-captured node. At most DIFFKV_EDGE_MAX tokens per block are added,
    prioritised (tier, has-node, proximity), so the extra residuals — and the RAM
    they cost — stay bounded well inside the max_residual cap. Runs AFTER the
    window pass, on the committed boost map, mirroring the other capture helpers.
    Returns the number of newly boosted rows."""
    if os.environ.get("DIFFKV_RESIDUAL_EDGE_CAPTURE", "1") != "1":
        return 0
    try:
        radius = int(os.environ.get("DIFFKV_EDGE_RADIUS", "6"))
    except ValueError:
        radius = 6
    try:
        edge_scale = float(os.environ.get("DIFFKV_EDGE_SCALE", "0.7"))
    except ValueError:
        edge_scale = 0.7
    try:
        edge_max = int(os.environ.get("DIFFKV_EDGE_MAX", "10"))
    except ValueError:
        edge_max = 10
    try:
        edge_floor = float(os.environ.get("DIFFKV_EDGE_FLOOR", "0.5"))
    except ValueError:
        edge_floor = 0.5
    try:
        rare_frac = float(os.environ.get("DIFFKV_EDGE_RARE_FRAC", "0.002"))
    except ValueError:
        rare_frac = 0.002
    if radius <= 0 or edge_scale <= 0.0 or edge_max <= 0:
        return 0
    # A STRONG connective may enter the exact set WITHOUT a captured node next to
    # it only if it is DISTINCTIVE in this document (in-doc count below rare_max).
    # This is what keeps decode cost flat: common words ("not", "more", "less",
    # "as") recur in every prose block, and capturing them everywhere would bloat
    # the per-block residual count the decoder must reconstruct — so they fire
    # ONLY next to content they actually negate/bound. Rare relational verbs
    # ("approaches", "converges", "asymptotically", "equals") are informative
    # wherever they appear, so they may fire alone (few per document).
    rare_max = max(8, int(total_tokens * rare_frac))

    S = len(tok_strs)
    node = [boost_multipliers[i] > 1.0 for i in range(S)]

    # Nearest captured-node distance in two O(S) sweeps (INF = out of radius).
    INF = radius + 1
    dist = [INF] * S
    node_boost = [0.0] * S          # boost of the nearest node (either side)
    last_i = -1
    for i in range(S):
        if node[i]:
            last_i = i
        if last_i >= 0 and (i - last_i) < dist[i]:
            dist[i] = i - last_i
            node_boost[i] = boost_multipliers[last_i]
    last_i = -1
    for i in range(S - 1, -1, -1):
        if node[i]:
            last_i = i
        if last_i >= 0 and (last_i - i) < dist[i]:
            dist[i] = last_i - i
            node_boost[i] = boost_multipliers[last_i]

    # Candidates: (tier, has_node, -dist, i). WEAK requires a node in radius;
    # STRONG may fire alone (its default anchor is the flat edge_floor boost).
    cands = []
    for i in range(S):
        if node[i]:
            continue
        tier = _edge_tier(tok_strs[i])
        if tier == 0:
            continue
        near = dist[i] <= radius
        if not near:
            # Weak connectives never fire off-node; strong ones only if rare.
            if tier != 2:
                continue
            cnt = counts.get(tids[i], 1) if tids is not None else 1
            if cnt > rare_max:
                continue
        cands.append((tier, 1 if near else 0, -dist[i], i))
    if not cands:
        return 0
    cands.sort(reverse=True)

    n_edge = 0
    tier_w = {2: 1.0, 1: 0.85}
    for tier, has_node, _negd, i in cands[:edge_max]:
        anchor = node_boost[i] if has_node else (tok_boost * edge_floor)
        val = anchor * edge_scale * tier_w[tier]
        # Never let an edge outrank the content it binds.
        if has_node:
            val = min(val, node_boost[i] * 0.95)
        if val > boost_multipliers[i]:
            boost_multipliers[i] = val
            n_edge += 1
    return n_edge


def _detect_table_rows(tok_strs):
    """Mark tokens that sit on TABLE-LIKE lines. A line = tokens up to and
    including a newline-bearing token. Two detection rules, both required
    because real documents arrive in both shapes:

    1. SEPARATOR rule (markdown/LaTeX sources): >= 2 STANDALONE separator
       tokens (strip == '|' / '&') among >= 3 tokens, plus table shape —
       line-initial separator, LaTeX `\\\\` terminator, or separator density
       >= 1/12 with >= 3 separators (the density guard rejects prose with
       inline |x−y| math, which otherwise marked 19 false-positive blocks).

    2. COLUMNAR rule (PDF copy-paste — the NAT-paper report 2026-07-13: PDF
       tables flatten to whitespace-aligned rows with NO pipes, the separator
       rule never fires, rows over-compress and generation collapses onto
       one high-salience fragment reused for every row). A CANDIDATE line
       has 3..48 tokens, ends in a digit-bearing token (skipping trailing
       punctuation), carries >= 2 digit-bearing tokens, and is not
       prose-dominated (2*digit_tokens >= lowercase words len >= 3). Two or
       more CONSECUTIVE candidates fire, marking their tokens plus one line
       above (the header). A lone numeric prose sentence has no numeric
       neighbor and stays unmarked; reference lists do fire — acceptable,
       exact citations are benign.

    Returns a per-token bool list."""
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
                or '\\\\' in tok_strs[ln[-1]]            # LaTeX row terminator
                or (seps >= 3 and seps * 12 >= len(ln)))

    def _columnar_candidate(ln):
        # Real table rows are SHORT. References ("[3] D. Achlioptas, ...,
        # 2001.") are 25-45 tokens, prose-heavy, and also end in digits —
        # marking them was measured to STEAL residual slots from a real
        # table sharing the block (markdown probe 6/6 -> 1/6 before this
        # tightening). <= 20 tokens qualifies outright; 21..48 only when
        # digit tokens outnumber prose words.
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
        if n_digit < 2:
            return False
        if len(ln) > 20 and n_digit < n_prose:
            return False
        for i in reversed(ln):                 # ends in a number
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
    if os.environ.get("DIFFKV_RESIDUAL_TABLE_CAPTION", "1") != "1":
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


def _apply_table_capture(boost_multipliers, tok_strs, tids, counts,
                         total_tokens, tok_boost):
    """Structured-data capture (DIFFKV_RESIDUAL_TABLE_CAPTURE, default ON).

    Tables break both existing capture rules at once. (1) A table row's cells
    are digits (is_core) so the whole body is boosted — but a block holding a
    table plus real technical filler carries MORE boosted rows than
    max_residual slots (measured 2026-07-13, NAT-style 6x4 table straddling a
    block boundary in paper text: 181 boosted for 128 slots), and which rows
    go lossy is decided by SVD error — structure-blind, so fragments of
    different rows survive and the decoder reassembles a plausible-but-wrong
    table (83.2 migrating from 7x7 to 3x3, fabricated 4x4 rows). (2) Header
    and unit cells ('Kernel', 'imgs', '/sec', 'Swin') are prose/core-less →
    never boosted → survive only as rank-r smear — which is how real imgs/sec
    throughputs come back as invented 'G/s' numbers.

    Fix: every token on a table-like line (pipes/ampersands — including the
    alpha header, unit, and row-name cells) gets the core boost multiplied by
    DIFFKV_RESIDUAL_TABLE_PRIORITY (default 4). Under saturation the err×boost
    ranking then keeps table rows ahead of ordinary boosted segments, and the
    lowest-information table rows (separator dashes, pipes — highly regular,
    near-zero SVD error) degrade first. The existing boosted-row budget floor
    picks these rows up automatically. Returns the number of marked rows."""
    # Default ON (measured 2026-07-13, 16k straddled-table probe, MLX:
    # list-all 3/6 → 6/6 == dense; NIAH 16k d0.5+d0.9 exact, MN 3/3,
    # synthesis 8k 6.7 == same-day capture-off control 6.7).
    if os.environ.get("DIFFKV_RESIDUAL_TABLE_CAPTURE", "1") != "1":
        return 0
    import math
    try:
        priority = float(os.environ.get("DIFFKV_RESIDUAL_TABLE_PRIORITY", "4.0"))
    except ValueError:
        priority = 4.0
    marked = _detect_table_rows(tok_strs)
    n_marked = 0
    for i, m in enumerate(marked):
        if not m:
            continue
        count = counts.get(tids[i], 1)
        idf = math.log(max(total_tokens, 2) / (count + 0.1))
        rarity_weight = max(1.0, min(idf, 6.0))
        boost_multipliers[i] = max(boost_multipliers[i],
                                   tok_boost * (rarity_weight / 2.0) * priority * m)
        n_marked += 1
    return n_marked


def compress_mlx_block_batched(deltas: mx.array, rank: int, n_oversamples: int = 5, n_iter: int = 2) -> Tuple[mx.array, mx.array, mx.array]:
    """Compress a batch of KV delta vectors using randomised truncated SVD on GPU in parallel.
    
    deltas shape: (B_batch, n, d)
    Returns:
      U_k: shape (B_batch, n, rank)
      Vh_k: shape (B_batch, rank, d)
      scale: shape (B_batch,)
    """
    B_batch, n, d = deltas.shape
    r_proj = min(rank + n_oversamples, n, d)
    
    # Cast to float32 to prevent float16 norm overflow and type errors in QR/SVD
    deltas_f32 = deltas.astype(mx.float32)
    
    # 1. Scale each matrix in the batch by its max absolute value
    scales = mx.max(mx.abs(deltas_f32), axis=(1, 2), keepdims=True)
    scales = mx.maximum(scales, 1e-9)
    x = deltas_f32 / scales
    
    # 2. Random projection matrix Omega (shared deterministic projection)
    seed = int(os.environ.get("DIFFKV_SVD_SEED", "1234"))
    key = mx.random.key(seed)
    Omega_single = mx.random.normal(shape=(d, r_proj), key=key, dtype=mx.float32)
    Omega = mx.broadcast_to(mx.expand_dims(Omega_single, 0), (B_batch, d, r_proj))
    
    # 3. Power iteration
    Y = x @ Omega  # (B_batch, n, r_proj)
    for _ in range(n_iter):
        Y = x @ (mx.transpose(x, (0, 2, 1)) @ Y)
        
    # 4. Batched QR decomposition
    Q, _ = mx.linalg.qr(Y, stream=mx.cpu)
    
    # 5. Project onto low-rank subspace
    B = mx.transpose(Q, (0, 2, 1)) @ x
    
    # 6. Batched covariance SVD to avoid N x N (2048 x 2048) full matrix memory allocation (20.6 GB)
    B_cov = B @ mx.transpose(B, (0, 2, 1))  # (B_batch, r_proj, r_proj)
    U_b, S_sq, _ = mx.linalg.svd(B_cov, stream=mx.cpu)
    S = mx.sqrt(mx.maximum(S_sq, 1e-9))
    
    # Reconstruct Vh: Vh = (U_b.T @ B) / S
    Vh = mx.transpose(U_b, (0, 2, 1)) @ B  # (B_batch, r_proj, d)
    Vh = Vh / mx.maximum(mx.expand_dims(S, 2), 1e-9)
    
    # 7. Truncate to rank and reconstruct U
    U_k = (Q @ U_b)[:, :, :rank]
    S_k = S[:, :rank]
    Vh_k = Vh[:, :rank, :]
    
    # Scale U_k by S_k
    U_k = U_k * mx.expand_dims(S_k, 1)
    
    return U_k.astype(deltas.dtype), Vh_k.astype(deltas.dtype), scales.squeeze(-1).squeeze(-1).astype(deltas.dtype)

def compress_mlx_block(deltas: mx.array, rank: int, n_oversamples: int = 5, n_iter: int = 2) -> Tuple[mx.array, mx.array, float, int]:
    """Compress a block of KV delta vectors using randomised truncated SVD.

    Delegates to compress_mlx_block_batched with batch=1, keeping all computation
    in the MLX/LAPACK unified path. This eliminates the prior mx.eval + np.array +
    np.linalg.qr/svd + np.array→mx.array round-trip on every decode eviction.

    mx.linalg.qr / svd use stream=mx.cpu internally (Apple LAPACK), so they do
    not require an explicit np.array transfer — the data stays on unified memory.
    The only CPU sync in the normal case is a single .item() for the scalar scale.

    NaN guard replaces np.linalg.LinAlgError: MLX does not raise on SVD failure,
    but the batched path pre-normalises by max-abs, so NaN only occurs on truly
    degenerate (effectively zero) blocks. The check is skipped on normal blocks.
    """
    n, d = deltas.shape
    rank = min(rank, n, d)
    if rank < 1:
        return mx.zeros((n, 1), dtype=deltas.dtype), mx.zeros((1, d), dtype=deltas.dtype), 1.0, 1

    # Expand to batch-of-1 and call the GPU/LAPACK batched path.
    U_batch, Vh_batch, scales_batch = compress_mlx_block_batched(
        mx.expand_dims(deltas, 0), rank, n_oversamples, n_iter
    )
    U_k   = U_batch[0]    # [n, rank]
    Vh_k  = Vh_batch[0]   # [rank, d]
    # One sync for the scalar scale value — unavoidable to return a Python float.
    scale = float(scales_batch[0].item())

    # NaN guard: only fires on degenerate blocks (scale clamped to ≥1e-9 by the
    # batched path, so NaN requires truly zero-energy data). Skipped on normal blocks
    # to keep the normal-case sync count at exactly 1 (the scale .item() above).
    if scale < 1e-6:
        if bool(mx.any(mx.isnan(U_k)).item()):
            k = min(rank, n, d)
            return (
                mx.zeros((n, k), dtype=deltas.dtype),
                mx.zeros((k, d), dtype=deltas.dtype),
                1.0, k,
            )

    k = min(rank, n, d)
    return U_k.astype(deltas.dtype), Vh_k.astype(deltas.dtype), scale, k

@mx.compile
def compute_decode_attention_static(
    q: mx.array,              # [H_q, D]
    comp_U: mx.array,         # [nb, S_comp, rank]       — pre-sliced to active blocks
    comp_VK: mx.array,        # [nb, kv_heads, rank, D]
    comp_VV: mx.array,        # [nb, kv_heads, rank, D]
    comp_anc_k: mx.array,     # [nb, kv_heads, D]
    comp_anc_v: mx.array,     # [nb, kv_heads, D]
    comp_scale: mx.array,     # [nb]
    comp_seq_len: mx.array,   # [nb]
    res_mask: mx.array,       # [nb, S_comp] bool — True where the position is an exact residual
    dense_k: mx.array,        # [kv_heads, max_dense_len, D]  (fixed-size padded buffer)
    dense_v: mx.array,        # [kv_heads, max_dense_len, D]
    dense_mask: mx.array,     # [max_dense_len] bool mask
    nb_actual: mx.array,      # tensor — how many blocks are actually valid
    scale: float,
    gpk: int,
    kv_heads: int,
    block_size: int,
    rank: int,
    max_dense_len: int,
):
    H_q, D = q.shape
    nb      = comp_U.shape[0]   # real block count (padded to power of 2)
    S_comp  = block_size - 1

    # ── 1. Sparse / Compressed Attention ─────────────────────────────────────
    if gpk > 1:
        H_kv = comp_anc_k.shape[1]
        
        # 1. AncK score (fp16 product, fp32 SUM — the validated pre-W1 form; see the
        # decode-precision note above the LSE merge for why W1's fp32 casts here were
        # reverted).
        AncK_exp = mx.expand_dims(comp_anc_k.transpose(1, 0, 2), 1) # [H_kv, 1, nb, D]
        q_exp = mx.expand_dims(q.reshape(H_kv, gpk, D), 2)           # [H_kv, gpk, 1, D]
        s_anc = mx.sum((q_exp * AncK_exp).astype(mx.float32), axis=-1) * scale     # [H_kv, gpk, nb]
        s_anc = s_anc.reshape(H_q, nb).astype(q.dtype)

        # 2. VK projection
        VK_exp = mx.expand_dims(comp_VK.transpose(1, 0, 2, 3), 1)   # [H_kv, 1, nb, rank, D]
        q_exp2 = mx.expand_dims(mx.expand_dims(q.reshape(H_kv, gpk, D), 2), 3) # [H_kv, gpk, 1, 1, D]
        q_proj_n = mx.sum((q_exp2 * VK_exp).astype(mx.float32), axis=-1) * scale   # [H_kv, gpk, nb, rank]
        q_proj_n = q_proj_n.reshape(H_q, nb, rank).astype(q.dtype)
    else:
        AncK_e_perm = comp_anc_k.transpose(1, 0, 2)   # [H_q, nb, D]
        s_anc = mx.sum((mx.expand_dims(q, 1) * AncK_e_perm).astype(mx.float32), axis=-1) * scale  # [H_q, nb]
        s_anc = s_anc.astype(q.dtype)

        VK_e_perm = comp_VK.transpose(1, 0, 2, 3)                            # [H_q, nb, rank, D]
        q_expanded = mx.expand_dims(mx.expand_dims(q, 1), 2)              # [H_q, 1, 1, D]
        q_proj_n   = mx.sum((q_expanded * VK_e_perm).astype(mx.float32), axis=-1) * scale      # [H_q, nb, rank]
        q_proj_n   = q_proj_n.astype(q.dtype)

    # Mask out padded blocks (index >= nb_actual)
    block_mask = mx.arange(nb) < nb_actual
    s_anc = mx.where(mx.expand_dims(block_mask, 0), s_anc, -float('inf'))

    # Delta scores: (q @ VK) @ U^T  →  [H_q, nb, S_comp]
    q_proj_n_perm       = q_proj_n.transpose(1, 0, 2)                 # [nb, H_q, rank]
    comp_U_transposed   = comp_U.transpose(0, 2, 1)                   # [nb, rank, S_comp]
    comp_U_t_exp        = mx.expand_dims(comp_U_transposed, 1)        # [nb, 1, rank, S_comp]
    q_proj_n_exp        = mx.expand_dims(q_proj_n_perm, 2)            # [nb, H_q, 1, rank]
    delta_s = mx.matmul(q_proj_n_exp, comp_U_t_exp).squeeze(2)        # [nb, H_q, S_comp]
    delta_s = delta_s.transpose(1, 0, 2)                               # [H_q, nb, S_comp]

    delta_s = delta_s * comp_scale.reshape(1, -1, 1)                  # apply svd_scale
    delta_s = delta_s + mx.expand_dims(s_anc, -1)                     # add anchor score

    # Drop exact-residual positions from the SVD pool: their lossy low-rank twin
    # is set to -inf so it gets zero softmax weight here, leaving the exact copy in
    # the dense pool as the token's sole representation. All-False mask = identity.
    delta_s = mx.where(mx.expand_dims(res_mask, 0), -float('inf'), delta_s)

    # Mask padding positions within partially-filled blocks
    s_range   = mx.arange(S_comp).reshape(1, 1, -1)
    valid_msk = s_range < comp_seq_len.reshape(1, -1, 1)
    delta_s   = mx.where(valid_msk, delta_s, -float('inf'))

    # Concatenate anchor + delta scores → [H_q, nb * block_size]
    scores_blocks = mx.concatenate([mx.expand_dims(s_anc, -1), delta_s], axis=-1)
    scores_sparse = scores_blocks.reshape(H_q, -1)  # [H_q, nb*block_size]

    # Accumulate logsumexp in float32 (overflow-safe), then cast back to the
    # activation dtype — the pre-W1 form. See the decode-precision note at the merge.
    lse_sparse = mx.logsumexp(scores_sparse.astype(mx.float32), axis=-1).astype(q.dtype)   # [H_q]
    w          = mx.softmax(scores_sparse, axis=-1)      # [H_q, nb*block_size]

    W_comp    = w.reshape(H_q, nb, block_size)           # [H_q, nb, block_size]
    w_anc     = W_comp[:, :, 0]                          # [H_q, nb]
    w_d       = W_comp[:, :, 1:]                         # [H_q, nb, S_comp]

    # Anchor output contribution
    w_block_sum = w_anc + mx.sum(w_d, axis=-1)           # [H_q, nb]
    if gpk > 1:
        AncV_exp = mx.expand_dims(comp_anc_v.transpose(1, 0, 2), 1)             # [H_kv, 1, nb, D]
        w_block_sum_exp = mx.expand_dims(w_block_sum.reshape(H_kv, gpk, nb), 3) # [H_kv, gpk, nb, 1]
        O_anc = mx.sum(w_block_sum_exp * AncV_exp, axis=2)                 # [H_kv, gpk, D]
        O_anc = O_anc.reshape(H_q, D)
    else:
        AncV_e_perm = comp_anc_v.transpose(1, 0, 2)
        O_anc = mx.sum(mx.expand_dims(w_block_sum, -1) * AncV_e_perm, axis=1)  # [H_q, D]

    # Delta (SVD) output contribution
    w_d_perm  = w_d.transpose(1, 0, 2)                          # [nb, H_q, S_comp]
    comp_U_exp = mx.expand_dims(comp_U, 1)                       # [nb, 1, S_comp, rank]
    w_proj     = mx.matmul(mx.expand_dims(w_d_perm, 2), comp_U_exp).squeeze(2)  # [nb, H_q, rank]
    w_proj     = w_proj * comp_scale.reshape(-1, 1, 1)           # apply svd_scale

    if gpk > 1:
        w_proj_exp = mx.expand_dims(w_proj.reshape(nb, H_kv, gpk, rank), 4)     # [nb, H_kv, gpk, rank, 1]
        VV_exp = mx.expand_dims(comp_VV, 2)                                     # [nb, H_kv, 1, rank, D]
        O_delta_block = mx.sum(w_proj_exp * VV_exp, axis=3)                 # [nb, H_kv, gpk, D]
        O_delta_block = O_delta_block.reshape(nb, H_q, D)
    else:
        O_delta_block = mx.matmul(mx.expand_dims(w_proj, 2), comp_VV).squeeze(2)  # [nb, H_q, D]
    O_delta       = mx.sum(O_delta_block, axis=0)                           # [H_q, D]

    out_sparse = O_anc + O_delta
    out_sparse = mx.where(mx.isnan(out_sparse), 0.0, out_sparse)

    # ── 2. Dense (recency window) Attention ──────────────────────────────────
    dense_mask_expanded = mx.expand_dims(dense_mask, 0)  # [1, max_dense_len]

    if gpk > 1:
        q_exp = mx.expand_dims(q.reshape(H_kv, gpk, D), 2)                       # [H_kv, gpk, 1, D]
        dk_exp = mx.expand_dims(dense_k, 1)                                      # [H_kv, 1, max_dense_len, D]
        scores_dense = mx.sum((q_exp * dk_exp).astype(mx.float32), axis=-1) * scale             # [H_kv, gpk, max_dense_len]
        scores_dense = scores_dense.reshape(H_q, -1).astype(q.dtype)
    else:
        scores_dense  = mx.sum((mx.expand_dims(q, 1) * dense_k).astype(mx.float32), axis=-1) * scale
        scores_dense  = scores_dense.astype(q.dtype)

    scores_dense  = mx.where(dense_mask_expanded, scores_dense, -float('inf'))
    lse_dense     = mx.logsumexp(scores_dense.astype(mx.float32), axis=-1).astype(q.dtype)
    weights_dense = mx.softmax(scores_dense, axis=-1)

    if gpk > 1:
        w_exp = mx.expand_dims(weights_dense.reshape(H_kv, gpk, -1), 3)          # [H_kv, gpk, max_dense_len, 1]
        dv_exp = mx.expand_dims(dense_v, 1)                                      # [H_kv, 1, max_dense_len, D]
        out_dense = mx.sum(w_exp * dv_exp, axis=2)                          # [H_kv, gpk, D]
        out_dense = out_dense.reshape(H_q, D)
    else:
        out_dense     = mx.sum(mx.expand_dims(weights_dense, -1) * dense_v, axis=1)
    out_dense     = mx.where(mx.isnan(out_dense), 0.0, out_dense)

    # ── 3. Flash-style LSE merge ──────────────────────────────────────────────
    NEG        = -1e9
    lse_sparse = mx.where(mx.isnan(lse_sparse) | mx.isinf(lse_sparse), NEG, lse_sparse)
    lse_dense  = mx.where(mx.isnan(lse_dense)  | mx.isinf(lse_dense),  NEG, lse_dense)
    out_sparse = mx.where(mx.isnan(out_sparse), 0.0, out_sparse)
    out_dense  = mx.where(mx.isnan(out_dense),  0.0, out_dense)

    # Correct the compressed pool's systematic LSE under-scoring (see DIFFKV_SPARSE_BIAS
    # note at module top). Default 0.0 → exact flash merge (parity-safe, no-op).
    if _SPARSE_BIAS_MODE == "auto":
        # Adaptive: full boost when sparse is competitive, decays to 0 as the dense half
        # (e.g. an exact needle residual) pulls ahead — synthesis-helpful AND NIAH-safe.
        adaptive_bias = mx.maximum(0.0, _SPARSE_BIAS_BASE - 0.5 * mx.maximum(0.0, (lse_dense - lse_sparse) - 4.0))
        lse_sparse = mx.where(lse_sparse <= NEG, lse_sparse, lse_sparse + adaptive_bias)
    elif _SPARSE_BIAS != 0.0:
        lse_sparse = mx.where(lse_sparse <= NEG, lse_sparse, lse_sparse + _SPARSE_BIAS)

    lse_max  = mx.maximum(lse_sparse, lse_dense)
    w_sparse = mx.exp(lse_sparse - lse_max)
    w_dense  = mx.exp(lse_dense  - lse_max)
    denom    = w_sparse + w_dense + 1e-9

    # DECODE-PRECISION NOTE (2026-07-04). This combine, and the score computations
    # above, keep the pre-W1 fp16 arithmetic (fp32 only where it always was: the
    # logsumexp accumulation). The eighth pass (W1) recast operands/accumulators to
    # fp32 for "overflow safety"; that is numerically cleaner but shifted the combine
    # by ~fp16-epsilon, which — because the >=16k compressed-decode retrieval sits on
    # a knife-edge (single-row/epsilon perturbations flip a cell; see AUDIT) —
    # REGRESSED NIAH from PASS to a repetition-loop FAIL at 16k/32k. A/B proven
    # 2026-07-04: reverting these casts (this combine was the load-bearing one)
    # restores exact recall at 16k/32k with parity/relational unchanged. W1 justified
    # the fp32 merge as a synthesis fix, but that was DISPROVEN: MLX-compressed
    # synthesis@8k scored 3.3/100 both with and without the fp32 merge, so the graded
    # blend is not the synthesis lever and the flag was dropped. Do not re-introduce
    # fp32 casts on this decode path without re-checking niah_recall.py --bench 16k/32k.
    out_combined = (
        out_sparse * mx.expand_dims(w_sparse, -1)
        + out_dense * mx.expand_dims(w_dense, -1)
    ) / mx.expand_dims(denom, -1)
    return out_combined, lse_sparse, lse_dense, scores_sparse


@mx.compile
def _dense_only_attention_static(
    q: mx.array,       # [H_q, D]
    dense_k: mx.array, # [kv_heads, max_dense_len, D]
    dense_v: mx.array,
    dense_len: mx.array,
    scale: float,
    gpk: int,
    max_dense_len: int,
):
    """Pure-dense decode attention — used when no compressed blocks exist yet."""
    dense_idx           = mx.arange(max_dense_len)
    dense_mask          = dense_idx < dense_len
    dense_mask_expanded = mx.expand_dims(dense_mask, 0)

    if gpk > 1:
        H_q, D = q.shape
        H_kv = dense_k.shape[0]
        q_exp = mx.expand_dims(q.reshape(H_kv, gpk, D), 2)        # [H_kv, gpk, 1, D]
        dk_exp = mx.expand_dims(dense_k, 1)                       # [H_kv, 1, max_dense_len, D]
        scores = mx.sum(q_exp.astype(mx.float32) * dk_exp.astype(mx.float32), axis=-1) * scale     # [H_kv, gpk, max_dense_len]
        scores = scores.reshape(H_q, -1).astype(q.dtype)
    else:
        scores  = mx.sum(mx.expand_dims(q, 1).astype(mx.float32) * dense_k.astype(mx.float32), axis=-1) * scale
        scores  = scores.astype(q.dtype)

    scores  = mx.where(dense_mask_expanded, scores, -float('inf'))
    weights = mx.softmax(scores, axis=-1)

    if gpk > 1:
        w_exp = mx.expand_dims(weights.reshape(H_kv, gpk, -1), 3)  # [H_kv, gpk, max_dense_len, 1]
        dv_exp = mx.expand_dims(dense_v, 1)                        # [H_kv, 1, max_dense_len, D]
        out = mx.sum(w_exp * dv_exp, axis=2)                 # [H_kv, gpk, D]
        out = out.reshape(H_q, D)
    else:
        out     = mx.sum(mx.expand_dims(weights, -1) * dense_v, axis=1)
    return mx.where(mx.isnan(out), 0.0, out)

@mx.compile
def _execute_decode_attention_compiled(
    q: mx.array,
    dense_k: mx.array,
    dense_v: mx.array,
    dense_len: mx.array,
    comp_U: mx.array,
    comp_VK: mx.array,
    comp_VV: mx.array,
    comp_anc_k: mx.array,
    comp_anc_v: mx.array,
    comp_min_k: mx.array,
    comp_max_k: mx.array,
    comp_scale: mx.array,
    comp_seq_len: mx.array,
    comp_res_k: mx.array,
    comp_res_v: mx.array,
    comp_res_n: mx.array,
    res_mask: mx.array,
    cached_sel: mx.array,
    nb_actual: mx.array,
    # Static parameters
    scale: float,
    gpk: int,
    kv_heads: int,
    block_size: int,
    rank: int,
    max_dense_len: int,
    max_residual: int,
    route_residuals: int,
    k_eff: int,
    router: str,
    use_topk: bool,
    use_cached_sel: bool,
):
    nb = comp_U.shape[0]

    if use_topk:
        if use_cached_sel:
            sel = cached_sel
        else:
            if router == "residual" and max_residual > 0:
                R = min(route_residuals, max_residual)
                res_n_slice = comp_res_n
                res_valid = mx.expand_dims(mx.arange(R), 0) < mx.expand_dims(mx.minimum(res_n_slice, R), 1)
                relevance = _block_relevance_residual(
                    q, comp_anc_k, comp_res_k[:, :R], res_valid, scale, gpk
                )
            else:
                relevance = _block_relevance_minmax(
                    q, comp_min_k, comp_max_k, scale, gpk
                )
            # Mask out relevance of padded blocks (index >= nb_actual) to ensure
            # argsort selects only valid blocks.
            block_mask = mx.arange(nb) < nb_actual
            relevance = relevance + (1.0 - block_mask.astype(relevance.dtype)) * -1e9
            sel = mx.argsort(relevance)[-k_eff:]

        topk_sel = sel
        comp_U_s       = mx.take(comp_U,       sel, axis=0)
        comp_VK_s      = mx.take(comp_VK,      sel, axis=0)
        comp_VV_s      = mx.take(comp_VV,      sel, axis=0)
        comp_anc_k_s   = mx.take(comp_anc_k,   sel, axis=0)
        comp_anc_v_s   = mx.take(comp_anc_v,   sel, axis=0)
        comp_scale_s   = mx.take(comp_scale,   sel, axis=0)
        comp_seq_len_s = mx.take(comp_seq_len, sel, axis=0)
        res_mask_s     = mx.take(res_mask,     sel, axis=0)

        # Vectorized top-K residual gather
        rk = mx.take(comp_res_k, sel, axis=0)
        rv = mx.take(comp_res_v, sel, axis=0)
        Ksel, Rw = rk.shape[0], rk.shape[1]
        res_k_all = rk.transpose(2, 0, 1, 3).reshape(kv_heads, Ksel * Rw, -1)
        res_v_all = rv.transpose(2, 0, 1, 3).reshape(kv_heads, Ksel * Rw, -1)
        total_res = Ksel * Rw
        nb_actual_for_attn = mx.array(k_eff, dtype=mx.int32)
    else:
        comp_U_s       = comp_U
        comp_VK_s      = comp_VK
        comp_VV_s      = comp_VV
        comp_anc_k_s   = comp_anc_k
        comp_anc_v_s   = comp_anc_v
        comp_scale_s   = comp_scale
        comp_seq_len_s = comp_seq_len
        res_mask_s     = res_mask

        # Uniform gather all blocks
        rk = comp_res_k
        rv = comp_res_v
        nb_blocks = rk.shape[0]
        R_width = rk.shape[1]
        res_k_all = rk.transpose(2, 0, 1, 3).reshape(kv_heads, nb_blocks * R_width, -1)
        res_v_all = rv.transpose(2, 0, 1, 3).reshape(kv_heads, nb_blocks * R_width, -1)
        total_res = nb_blocks * R_width
        nb_actual_for_attn = nb_actual

    # Static layout concatenation
    dense_k_for_attn = mx.concatenate([res_k_all, dense_k], axis=1)
    dense_v_for_attn = mx.concatenate([res_v_all, dense_v], axis=1)
    current_max_dense_len = total_res + max_dense_len

    # Mask out padded residuals and padded dense elements
    res_mask_attn = mx.arange(total_res) < (nb_actual_for_attn * max_residual)
    dense_mask_attn = mx.arange(max_dense_len) < dense_len
    dense_mask_combined = mx.concatenate([res_mask_attn, dense_mask_attn], axis=0)

    out_combined, lse_sparse, lse_dense, scores_sparse = compute_decode_attention_static(
        q, comp_U_s, comp_VK_s, comp_VV_s, comp_anc_k_s, comp_anc_v_s,
        comp_scale_s, comp_seq_len_s, res_mask_s,
        dense_k_for_attn, dense_v_for_attn, dense_mask_combined,
        nb_actual_for_attn,
        scale, gpk, kv_heads, block_size, rank,
        current_max_dense_len
    )

    return out_combined, (sel if use_topk else cached_sel), lse_sparse, lse_dense, scores_sparse


@mx.compile
def _block_relevance_minmax(
    q: mx.array,              # [H_q, D]
    comp_min_k: mx.array,     # [nb, kv_heads, D]  element-wise key min over block
    comp_max_k: mx.array,     # [nb, kv_heads, D]  element-wise key max over block
    scale: float,
    gpk: int,
):
    """Quest-style per-block relevance router for top-K block selection.

    For each block, returns an upper bound on its max attention score q·k:

        bound(block) = sum_d max(q_d * min_d, q_d * max_d)   (then * scale, max over heads)

    computed from the element-wise key min/max over the block's real keys. Cheap
    (O(nb·D)). The bound is loose, so the caller keeps a generous K (a fraction of
    the block count at very large contexts). The expensive value reconstruction +
    exact-residual attention then run only for the top-K blocks, so decode cost
    scales with K, not total context.
    """
    if gpk > 1:
        H_q, D = q.shape
        H_kv = comp_min_k.shape[1]
        nb = comp_min_k.shape[0]
        MIN_exp = mx.expand_dims(comp_min_k.transpose(1, 0, 2), 1) # [H_kv, 1, nb, D]
        MAX_exp = mx.expand_dims(comp_max_k.transpose(1, 0, 2), 1) # [H_kv, 1, nb, D]
        q_exp = mx.expand_dims(q.reshape(H_kv, gpk, D), 2)          # [H_kv, gpk, 1, D]
        # fp16 product / fp32 sum — pre-W1 router arithmetic (kept in lockstep with
        # the decode path; see the decode-precision note in compute_decode_attention_static).
        bound = mx.sum(mx.maximum(q_exp * MIN_exp, q_exp * MAX_exp), axis=-1) * scale # [H_kv, gpk, nb]
        bound = bound.reshape(H_q, nb)
    else:
        MIN_p = comp_min_k.transpose(1, 0, 2)                  # [H, nb, D]
        MAX_p = comp_max_k.transpose(1, 0, 2)
        q_e   = mx.expand_dims(q, 1)                    # [H, 1, D]
        bound = mx.sum(mx.maximum(q_e * MIN_p, q_e * MAX_p), axis=-1) * scale  # [H, nb]
    return mx.max(bound, axis=0)                    # [nb]


@mx.compile
def _block_relevance_residual(
    q: mx.array,              # [H_q, D]
    comp_anc_k: mx.array,     # [nb, kv_heads, D]            exact anchor key
    comp_res_k: mx.array,     # [nb, R, kv_heads, D]         top-R exact residual keys
    res_valid: mx.array,      # [nb, R] bool
    scale: float,
    gpk: int,
):
    """Exact-key relevance router: rank blocks by the largest TRUE q·k over each
    block's anchor + its top-R most-distinctive (highest-error) residual tokens.

    Unlike a min/max box or an SVD low-rank score — both cheap *summaries* that by
    construction miss low-energy outliers — the residuals ARE the block's outlier
    tokens (a buried passcode is exactly such a token). Scoring them directly gives
    a TIGHT, exact relevance signal that reliably keeps the needle's block in the
    top-K even at very large block counts. Cost is O(nb·R·D); R≪block_size keeps it
    cheap enough for 1M-token contexts. Model-agnostic: no tuning to head count,
    RoPE, or content — it scores whatever each block's distinctive keys are.
    """
    if gpk > 1:
        H_q, D = q.shape
        H_kv = comp_anc_k.shape[1]
        nb = comp_anc_k.shape[0]
        
        # fp16 product / fp32 sum — pre-W1 router arithmetic (see the decode-precision
        # note in compute_decode_attention_static; kept in lockstep with the decode path).
        ANC_exp = mx.expand_dims(comp_anc_k.transpose(1, 0, 2), 1) # [H_kv, 1, nb, D]
        q_exp = mx.expand_dims(q.reshape(H_kv, gpk, D), 2)          # [H_kv, gpk, 1, D]
        s_anc = mx.sum((q_exp * ANC_exp).astype(mx.float32), axis=-1) * scale     # [H_kv, gpk, nb]
        s_anc = s_anc.reshape(H_q, nb).astype(q.dtype)

        RK_exp = mx.expand_dims(comp_res_k.transpose(2, 0, 1, 3), 1) # [H_kv, 1, nb, R, D]
        q_exp2 = mx.expand_dims(mx.expand_dims(q.reshape(H_kv, gpk, D), 2), 3) # [H_kv, gpk, 1, 1, D]
        s_res = mx.sum((q_exp2 * RK_exp).astype(mx.float32), axis=-1) * scale       # [H_kv, gpk, nb, R]
        s_res = s_res.astype(q.dtype)
        res_valid_exp = mx.expand_dims(mx.expand_dims(res_valid, 0), 1)    # [1, 1, nb, R]
        s_res = mx.where(res_valid_exp, s_res, -float('inf'))
        res_max = mx.max(s_res, axis=-1)                       # [H_kv, gpk, nb]
        res_max = res_max.reshape(H_q, nb)
    else:
        ANC_p = comp_anc_k.transpose(1, 0, 2)                          # [H, nb, D]
        s_anc = mx.sum((mx.expand_dims(q, 1) * ANC_p).astype(mx.float32), axis=-1) * scale  # [H, nb]
        s_anc = s_anc.astype(q.dtype)

        RK_p  = comp_res_k.transpose(2, 0, 1, 3)                        # [H, nb, R, D]
        q_e2  = mx.expand_dims(mx.expand_dims(q, 1), 1)         # [H, 1, 1, D]
        s_res = mx.sum((q_e2 * RK_p).astype(mx.float32), axis=-1) * scale            # [H, nb, R]
        s_res = s_res.astype(q.dtype)
        s_res = mx.where(mx.expand_dims(res_valid, 0), s_res, -float('inf'))
        res_max = mx.max(s_res, axis=-1)                        # [H, nb]

    return mx.max(mx.maximum(s_anc, res_max), axis=0)       # [nb]



def _cache_fetch(cache, keys, values):
    """Call cache.update_and_fetch and dequantize if the cache is quantized.

    KVCache returns plain mx.array tensors.
    QuantizedKVCache returns a tuple-of-tuples (w, scales, biases) per K/V.
    We transparently dequantize so callers always get mx.array back.
    """
    raw_k, raw_v = cache.update_and_fetch(keys, values)
    if isinstance(raw_k, (list, tuple)):
        # QuantizedKVCache: raw_k = (w, scales, biases)
        gs  = getattr(cache, "group_size", 64)
        bits = getattr(cache, "bits", 8)
        raw_k = mx.dequantize(raw_k[0], raw_k[1], raw_k[2], gs, bits)
        raw_v = mx.dequantize(raw_v[0], raw_v[1], raw_v[2], gs, bits)
    return raw_k, raw_v


def _sparse_prefill_attend(
    q_rot: mx.array,       # [1, H_q, L, D]   rotated queries for the current chunk
    all_k: mx.array,       # [1, H_kv, T, D]  rotated keys, ALL tokens so far (T = cur_start + L)
    all_v: mx.array,       # [1, H_kv, T, D]
    cur_start: int,        # absolute position of the first query in this chunk
    scale: float,
    gpk: int,              # GQA group size = H_q // H_kv
    block_size: int,
    window: int,           # exact recency window (tokens), always attended
    sink_blocks: int,      # leading blocks kept as always-attended attention sinks
    kmin: int,
    frac: float,
    dbg: bool = False,
):
    """DSA/NSA-style block-sparse PREFILL attention for one chunk (HANDOFF §DSA).

    Instead of dense attention over all T keys, build a SPARSE key/value set —
        [ leading sink blocks | top-K routed history blocks | recency window | current chunk ]
    — and run ONE masked SDPA over it. History keys (absolute pos < cur_start) are fully
    visible; the current chunk is causal. Blocks are not yet compressed during prefill, so
    routing is a Quest-style min/max bound (`_block_relevance_minmax`) computed on-the-fly
    from the RAW block keys. Compute drops O(L*T) -> O(L*Ksel). Returns [1, H_q, L, D].
    """
    _, H_q, L, D = q_rot.shape
    H_kv = all_k.shape[1]
    T = all_k.shape[2]
    neg_inf = mx.array(-float("inf"), dtype=q_rot.dtype)
    zero = mx.array(0.0, dtype=q_rot.dtype)

    # Aligned routable region: whole blocks fully inside [sink_end, cur_start - window).
    sink_end = sink_blocks * block_size
    first_blk = (sink_end + block_size - 1) // block_size    # first block fully >= sink_end
    last_blk = (cur_start - window) // block_size            # blocks fully below cur_start-window
    nb = last_blk - first_blk

    if nb <= 0:
        # Not enough prunable history — dense causal over everything.
        ii = mx.arange(L).reshape(L, 1) + cur_start
        jj = mx.arange(T).reshape(1, T)
        mask = mx.where(jj <= ii, zero, neg_inf)
        return mx.fast.scaled_dot_product_attention(q_rot, all_k, all_v, scale=scale, mask=mask)

    aligned_lo = first_blk * block_size
    aligned_hi = last_blk * block_size

    # Per-block key min/max over the raw keys → [nb, H_kv, D].
    mid_k = all_k[:, :, aligned_lo:aligned_hi, :].reshape(H_kv, nb, block_size, D)
    comp_min_k = mx.min(mid_k, axis=2).transpose(1, 0, 2)
    comp_max_k = mx.max(mid_k, axis=2).transpose(1, 0, 2)

    # Coarse block selection uses a single pooled query per head (mean over the chunk).
    q_rep = mx.mean(q_rot[0], axis=1)                        # [H_q, D]
    rel = _block_relevance_minmax(q_rep, comp_min_k, comp_max_k, scale, gpk)  # [nb]

    K = min(nb, max(kmin, int(math.ceil(frac * nb))))
    # Build the gather index list ENTIRELY in MLX (no host sync). The mask makes ALL history
    # keys fully visible regardless of order, so the selected blocks need NOT be sorted — which
    # lets us skip the per-layer `mx.eval(sel)` + `.tolist()` GPU→CPU sync that otherwise
    # serialized ~L/CH×n_layers times per prefill.
    i32 = mx.int32
    if K >= nb:
        sel_tok = mx.arange(aligned_lo, aligned_hi, dtype=i32)          # all routable tokens
    elif K <= 0:
        sel_tok = mx.arange(0, 0, dtype=i32)                            # pure StreamingLLM: no routed blocks
    else:
        sel_blk = mx.argpartition(-rel, K)[:K].astype(i32)             # [K] routable block idxs
        sel_abs = (first_blk + sel_blk) * block_size                   # [K] absolute block starts
        offs = mx.arange(block_size, dtype=i32)                        # [block_size]
        sel_tok = (mx.expand_dims(sel_abs, 1) + mx.expand_dims(offs, 0)).reshape(-1)  # [K*bs]

    # Global key index list:
    #   [0, aligned_lo)          leading sink blocks + pre-alignment slack (fully attended)
    #   selected blocks          K*block_size rows (unordered — mask treats them uniformly)
    #   [aligned_hi, cur_start)  post-alignment slack + recency window (fully attended)
    #   [cur_start, T)           current chunk (causal)
    idx_parts = []
    if aligned_lo > 0:
        idx_parts.append(mx.arange(0, aligned_lo, dtype=i32))
    idx_parts.append(sel_tok)
    if cur_start > aligned_hi:
        idx_parts.append(mx.arange(aligned_hi, cur_start, dtype=i32))
    idx_parts.append(mx.arange(cur_start, T, dtype=i32))
    key_idx = mx.concatenate(idx_parts)                     # [Ksel]  (shape is static → no sync)
    Ksel = int(key_idx.shape[0])

    k_sel = mx.take(all_k, key_idx, axis=2)                 # [1, H_kv, Ksel, D]
    v_sel = mx.take(all_v, key_idx, axis=2)

    # Additive mask [L, Ksel]: history keys visible (0); current chunk causal.
    n_hist = Ksel - L
    hist_mask = mx.broadcast_to(zero, (L, n_hist))
    ii = mx.arange(L).reshape(L, 1)
    jj = mx.arange(L).reshape(1, L)
    cur_mask = mx.where(jj <= ii, zero, neg_inf)            # [L, L]
    mask = mx.concatenate([hist_mask, cur_mask], axis=1)    # [L, Ksel]

    out = mx.fast.scaled_dot_product_attention(q_rot, k_sel, v_sel, scale=scale, mask=mask)
    if dbg:
        print(f"[SP] cur_start={cur_start} L={L} nb={nb} K={K} Ksel={Ksel} dense={T} "
              f"({100.0*Ksel/max(1,T):.0f}% of dense)", flush=True)
    return out


_LEGO_MASK_CACHE: Dict[Tuple[int, int, Any], mx.array] = {}


def _lego_uniform_mask(L: int, S_hist: int, dt) -> mx.array:
    """[zeros(L,S_hist) | causal(L,L)] additive mask, cached.

    In the uniform-studs case every attended history row is valid (anchors,
    full residual sets, sinks, ring), so the mask depends only on (L, S_hist).
    Building it per layer materialised ~12 MB of transients per layer per chunk
    at 16k — cached, it is built once per (chunk shape) and shared by all 28
    layers and by later chunks with the same shape."""
    key = (L, S_hist, dt)
    m = _LEGO_MASK_CACHE.get(key)
    if m is None:
        if len(_LEGO_MASK_CACHE) > 64:
            _LEGO_MASK_CACHE.clear()
        neg_inf = mx.array(-float("inf"), dtype=dt)
        zero = mx.array(0.0, dtype=dt)
        hist = mx.broadcast_to(zero, (L, S_hist))
        ii = mx.arange(L).reshape(L, 1)
        jj = mx.arange(L).reshape(1, L)
        cur = mx.where(jj <= ii, zero, neg_inf)
        m = mx.concatenate([hist, cur], axis=1).reshape(1, 1, L, S_hist + L)
        mx.eval(m)
        _LEGO_MASK_CACHE[key] = m
    return m


def _lego_prefill_attend(
    manager,
    session: Dict,
    layer_idx: int,
    q_rot: mx.array,       # [1, H_q, L, D]  rotated queries for the current chunk
    k_rot: mx.array,       # [1, H_kv, L, D] rotated keys of the current chunk
    v_cur: mx.array,       # [1, H_kv, L, D]
    cur_start: int,        # absolute position of the first query in this chunk
    scale: float,
    gpk: int,
    dbg: bool = False,
):
    """LEGO streaming-prefill attention (DIFFKV_LEGO_PREFILL — see the flag note
    in MLXKVBlockManager.__init__).

    Attends the current chunk against
        [ raw sinks | top-K routed COMPRESSED far blocks (materialised) | raw recency RING | self(causal) ]
    built entirely from the DiffKV session state — the raw prompt KV cache is
    never consulted, so prefill peak raw KV is O(sinks + ring + chunk), not O(T).
    The ring (last ~DIFFKV_LEGO_RING tokens, raw and exact) is the chunk's local
    neighborhood; lego pieces cover only whole blocks below the ring.

    Block routing uses the stored per-block summaries (anchor + exact residual
    keys — the same residual router decode uses), and materialisation is the same
    math as the decode route step: recon[t>0] = anchor + scale·(U[t]·V), position 0
    = anchor, exact residual rows appended and their lossy low-rank twins masked.
    Unlike decode, no sparse bias is applied (prefill hidden states should stay as
    close to the exact computation as possible) and invalid rows are masked with
    -inf rather than compacted (a per-layer host sync per chunk isn't worth ~30%
    fewer key rows at chunk granularity). Returns [1, H_q, L, D].
    """
    _, H_q, L, D = q_rot.shape
    H_kv = manager.kv_heads
    bs = manager.block_size
    S_comp = bs - 1
    nb = session["num_blocks"][layer_idx]
    sb = min(manager._sp_sink_blocks, nb)
    R = manager.max_residual
    # DIFFKV_LEGO_FP32 — SDPA dtype dial. Default now FOLLOWS THE ROW SOURCE:
    # in studs mode (the default) every attended row is an EXACT fp16 tensor
    # (sinks/ring/residuals/anchors are stored fp16), so fp16 SDPA is the same
    # arithmetic the validated sparse prefill runs — and skipping the fp32 casts
    # removes ~8 MB/layer/chunk of transient copies (the ring recast alone was
    # ~224 MB per 512-token chunk across 28 layers — the "lego RAM spikes"
    # report, 2026-07-12). Recon mode keeps fp32 (reconstruction noise sits on
    # a knife-edge; mirrors DIFFKV_DECODE_FUSED_FP32). Explicit env overrides both.
    _fp32_env = os.environ.get("DIFFKV_LEGO_FP32")
    _use_recon = os.environ.get("DIFFKV_LEGO_RECON", "0") == "1"
    if _fp32_env is not None:
        _fdt = mx.float32 if _fp32_env != "0" else q_rot.dtype
    else:
        _fdt = mx.float32 if _use_recon else q_rot.dtype
    neg_inf = mx.array(-float("inf"), dtype=_fdt)
    zero = mx.array(0.0, dtype=_fdt)

    # ── 1. Route the compressed FAR blocks (whole blocks below the ring) ────
    ring_start = session["lego_ring_start"][layer_idx]      # block-aligned, absolute
    far_nb = min(nb, ring_start // bs)
    nb_routable = far_nb - sb
    q_rep = mx.mean(q_rot[0], axis=1)                       # [H_q, D]
    ak_all = session["comp_anc_k"][layer_idx][sb:far_nb]    # [nb_r, H_kv, D]
    K_route = min(nb_routable, max(manager._lego_kmin,
                                   int(math.ceil(manager._lego_frac * nb_routable))))
    if K_route >= nb_routable:
        sel_abs = mx.arange(sb, far_nb, dtype=mx.int32)
        K_route = nb_routable
    else:
        if R > 0 and manager._lego_router != "minmax":
            R_route = min(manager.route_residuals, R)
            rk_all = session["comp_res_k"][layer_idx][sb:far_nb, :R_route]  # [nb_r, R_route, H_kv, D]
            res_n_np = np.asarray(session["comp_res_n"][layer_idx][sb:far_nb], dtype=np.int32)
            rvld = mx.expand_dims(mx.arange(R_route), 0) < mx.expand_dims(
                mx.minimum(mx.array(res_n_np), R_route), 1)
            rel = _block_relevance_residual(q_rep, ak_all, rk_all, rvld, scale, gpk)
        else:
            rel = _block_relevance_minmax(
                q_rep,
                session["comp_min_k"][layer_idx][sb:far_nb],
                session["comp_max_k"][layer_idx][sb:far_nb],
                scale, gpk)
        sel_abs = (mx.argpartition(-rel, K_route)[:K_route] + sb).astype(mx.int32)

    # ── 2. Materialise the selected blocks (same math as the decode route) ──
    ak  = mx.take(session["comp_anc_k"][layer_idx], sel_abs, 0)   # [K, H_kv, D]
    av  = mx.take(session["comp_anc_v"][layer_idx], sel_abs, 0)

    # DIFFKV_LEGO_RECON (default 0 — studs-only): whether routed blocks contribute
    # their RECONSTRUCTED low-rank rows to the chunk attention, or only their
    # exact rows (anchor + residual "studs"). Studs-only is pure omission — zero
    # reconstruction noise; the residuals are each block's highest-error (most
    # distinctive) half, so what is omitted is exactly the well-approximated
    # low-information half. A/B on real-paper synthesis (2026-07-11, Qwen-1.5B):
    #   recon+bias0 3.3@8k/6.7@16k · recon+bias2 10.0@8k/0.0@16k (ctx-dependent,
    #   not shippable) · STUDS 10.0@8k/6.7@16k — Pareto-best, and beats the
    #   no-lego baseline at 16k (3.3). It is also ~3x fewer block rows per chunk.
    # =1 re-enables recon rows (pair with DIFFKV_LEGO_BIAS to counter their
    # systematic under-scoring). (_use_recon is resolved above with the dtype dial.)
    ak_e = mx.expand_dims(ak, 2)
    av_e = mx.expand_dims(av, 2)
    if _use_recon:
        U   = mx.take(session["comp_U"][layer_idx],     sel_abs, 0)   # [K, S_comp, rank]
        VK  = mx.take(session["comp_VK"][layer_idx],    sel_abs, 0)
        VV  = mx.take(session["comp_VV"][layer_idx],    sel_abs, 0)
        sc  = mx.take(session["comp_scale"][layer_idx], sel_abs, 0)   # [K] fp32
        csl = mx.take(session["comp_seq_len"][layer_idx], sel_abs, 0) # [K]
        rmask = mx.take(session["comp_res_mask"][layer_idx], sel_abs, 0)  # [K, S_comp]
        delta_k = mx.einsum("bsr,bhrd->bhsd", U, VK) * sc.reshape(K_route, 1, 1, 1)
        delta_v = mx.einsum("bsr,bhrd->bhsd", U, VV) * sc.reshape(K_route, 1, 1, 1)
        blk_k = mx.concatenate([ak_e, ak_e + delta_k], axis=2) \
                  .transpose(1, 0, 2, 3).reshape(H_kv, K_route * bs, D).astype(_fdt)
        blk_v = mx.concatenate([av_e, av_e + delta_v], axis=2) \
                  .transpose(1, 0, 2, 3).reshape(H_kv, K_route * bs, D).astype(_fdt)

        # Validity: padded positions of partial blocks + the low-rank twins of exact
        # residual rows are masked out (their exact copies are the residual rows below).
        pos = mx.arange(S_comp).reshape(1, S_comp)
        recon_valid = (pos < csl.reshape(K_route, 1)) & (~rmask)
        blk_valid = mx.concatenate(
            [mx.ones((K_route, 1), dtype=mx.bool_), recon_valid], axis=1).reshape(K_route * bs)
        # DIFFKV_LEGO_BIAS — additive score bias (nats) on the RECON block rows only
        # (not residuals, not sinks/ring), the same correction the decode paths apply
        # (see DIFFKV_SPARSE_BIAS at module top): low-rank reconstruction
        # systematically UNDER-scores, so without it chunks under-attend the far
        # field relative to the exact ring. Default 0.0 (no correction).
        try:
            _lego_bias = float(os.environ.get("DIFFKV_LEGO_BIAS", "0"))
        except ValueError:
            _lego_bias = 0.0
        blk_add = mx.where(blk_valid, mx.array(_lego_bias, dtype=_fdt), neg_inf)  # [K*bs]
    else:
        # Anchors only; the blocks' exact residual rows are appended below.
        blk_k = ak_e.transpose(1, 0, 2, 3).reshape(H_kv, K_route, D).astype(_fdt)
        blk_v = av_e.transpose(1, 0, 2, 3).reshape(H_kv, K_route, D).astype(_fdt)
        blk_add = mx.broadcast_to(zero, (K_route,))

    parts_k = [session["lego_sink_k"][layer_idx].astype(_fdt), blk_k]
    parts_v = [session["lego_sink_v"][layer_idx].astype(_fdt), blk_v]
    S0 = int(parts_k[0].shape[1])
    mask_parts = [mx.broadcast_to(zero, (S0,)), blk_add]

    if R > 0:
        rk = mx.take(session["comp_res_k"][layer_idx], sel_abs, 0)   # [K, R, H_kv, D]
        rv = mx.take(session["comp_res_v"][layer_idx], sel_abs, 0)
        res_n_all = mx.array(np.asarray(session["comp_res_n"][layer_idx], dtype=np.int32))
        res_n_sel = mx.take(res_n_all, sel_abs)
        res_valid = (mx.arange(R).reshape(1, R) < res_n_sel.reshape(K_route, 1)).reshape(K_route * R)
        parts_k.append(rk.transpose(2, 0, 1, 3).reshape(H_kv, K_route * R, D).astype(_fdt))
        parts_v.append(rv.transpose(2, 0, 1, 3).reshape(H_kv, K_route * R, D).astype(_fdt))
        mask_parts.append(mx.where(res_valid, zero, neg_inf))

    # ── 3. Raw recency ring (exact local neighborhood) + current chunk ──────
    ring_k = session["lego_ring_k"][layer_idx]              # covers [ring_start, cur_start)
    n_ring = int(ring_k.shape[1])
    if n_ring > 0:
        parts_k.append(ring_k.astype(_fdt))
        parts_v.append(session["lego_ring_v"][layer_idx].astype(_fdt))
        mask_parts.append(mx.broadcast_to(zero, (n_ring,)))
    parts_k.append(k_rot[0].astype(_fdt))
    parts_v.append(v_cur[0].astype(_fdt))

    k_all = mx.concatenate(parts_k, axis=1)                      # [H_kv, S, D]
    v_all = mx.concatenate(parts_v, axis=1)
    S_hist = int(k_all.shape[1]) - L

    # Uniform-studs fast path: every history row valid → the mask is a pure
    # function of (L, S_hist) and comes from the shared cache (built once per
    # chunk shape instead of once per layer). Uniformity check is host-only:
    # prefill blocks always carry full residual sets.
    _res_n_list = session["comp_res_n"][layer_idx]
    _studs_uniform = (not _use_recon) and (
        R == 0 or min(_res_n_list[sb:far_nb]) == R)
    if _studs_uniform:
        mask = _lego_uniform_mask(L, S_hist, _fdt)
    else:
        hist_add = mx.concatenate(mask_parts)                    # [S_hist]
        hist_mask = mx.broadcast_to(hist_add.reshape(1, S_hist), (L, S_hist))
        ii = mx.arange(L).reshape(L, 1)
        jj = mx.arange(L).reshape(1, L)
        cur_mask = mx.where(jj <= ii, zero, neg_inf)             # [L, L]
        mask = mx.concatenate([hist_mask, cur_mask], axis=1).reshape(1, 1, L, S_hist + L)

    out = mx.fast.scaled_dot_product_attention(
        q_rot.astype(_fdt),
        mx.expand_dims(k_all, 0),
        mx.expand_dims(v_all, 0),
        scale=scale,
        mask=mask).astype(q_rot.dtype)
    if dbg:
        T = cur_start + L
        print(f"[LEGO] cur_start={cur_start} L={L} far_nb={far_nb}/{nb} K={K_route} "
              f"ring={n_ring} rows={S_hist + L} (raw would be {T}; "
              f"{100.0 * (S_hist + L) / max(1, T):.0f}%)", flush=True)
    return out


class DummyMLXPool:
    def __init__(self, manager):
        self.manager = manager

    @property
    def _free_indices(self):
        allocated = 0
        for session in self.manager.sessions.values():
            comp_len = session["num_blocks"][0] * self.manager.block_size
            dense_len = session["dense_lens"][0]
            total_len = comp_len + dense_len
            num_logical_blocks = (total_len + self.manager.block_size - 1) // self.manager.block_size
            allocated += num_logical_blocks
        free_count = max(0, self.manager.max_blocks - allocated)
        return [0] * free_count

    @property
    def current_blocks(self):
        return self.manager.max_blocks

    @property
    def W_proj(self):
        return getattr(self.manager, "W_proj", None)

class MLXKVBlockManager:
    @property
    def native_pool(self):
        return DummyMLXPool(self)

    def __init__(self, num_layers: int, heads: int, kv_heads: int, head_dim: int, rank: int, block_size: int, recency_window: int = 512):
        self.num_layers = num_layers
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.block_size = block_size
        
        # ── Dense recency window ───────────────────────────────────────────────
        # recency_window = number of most-recent EXACT (uncompressed) tokens kept
        # per layer. Scales with model capacity via a factor rather than a flat
        # constant: round_to_next_512(num_layers * head_dim * dense_window_factor).
        # Qwen2.5-1.5B (28 layers, 128 head_dim): factor=0.25 → 896 → 1024.
        # DIFFKV_ENGAGE_THRESHOLD hard-overrides; DIFFKV_DENSE_WINDOW_FACTOR tunes
        # the factor only (lower it, e.g. 0.125, to trade recency for RAM).
        env_engage = os.environ.get("DIFFKV_ENGAGE_THRESHOLD")
        if env_engage is not None:
            try:
                recency_window = int(env_engage)
            except ValueError:
                pass
        else:
            _factor = float(os.environ.get("DIFFKV_DENSE_WINDOW_FACTOR", "0.25"))
            _raw = num_layers * head_dim * _factor
            # Round up to nearest 512 for stable compiled-kernel shapes.
            recency_window = max(512, (int(_raw) + 511) // 512 * 512)
        self.recency_window = recency_window
        
        # max_blocks: how many SVD-compressed blocks we can hold simultaneously.
        # 256 blocks × 256 tokens = 65536 compressed tokens — enough for 64k context.
        # Configurable via DIFFKV_MAX_BLOCKS env var (lower = less VRAM).
        self.max_blocks = int(os.environ.get("DIFFKV_MAX_BLOCKS", "256"))
        # Exact residual tokens kept per block (top-by-reconstruction-error).
        # DEFAULT 128 (raised from 64 on 2026-07-04) — DO NOT lower this for a decode-speed
        # win without re-checking PROSE fidelity: raising 64→128 measurably fixed multi-fact
        # PROSE recall from compressed context (names/places/roles), which the low-rank
        # reconstruction otherwise confabulates. A/B on 3 buried-fact prompts (rover log,
        # corporate memo): at 64 the model returned "Dr. Sarah Thompson", "Dr. Toshiko Yamada",
        # "Boston, MA" (all WRONG); at 128 it returned the correct "Dr. Sara Voss",
        # "Dr. Yuki Tanaka", "Fairhaven Square". NIAH (distinctive-token recall) is 4/4 at both
        # 64 and 128 — 64 was already enough for a single needle, which is why the earlier
        # "MAX_RESIDUAL increases don't help" note (true for NIAH) did NOT generalize to prose.
        # COST is ~25-30% slower decode (e.g. 16k: ~14→~10 tps) + 2× residual pool memory;
        # accepted deliberately (accuracy > short-context throughput). To trade back for speed
        # on retrieval-only workloads, set DIFFKV_MAX_RESIDUAL=64.
        # Restore default to 128 for full correctness and to prevent repetitive loops.
        # Users can override this to 32 or 64 via DIFFKV_MAX_RESIDUAL if needed.
        self.max_residual = int(os.environ.get("DIFFKV_MAX_RESIDUAL", "128"))
        # Top-K block routing: when >0 and the live block count exceeds it, decode
        # scores all blocks cheaply (Quest-style key min/max upper bound) but runs
        # the expensive value reconstruction + exact-residual attention only for the
        # K most relevant blocks — so decode cost scales with K, not total context.
        # K = max(topk_blocks, topk_frac*nb). Default is fixed K=16 (topk_frac=0):
        # validated EXACT recall + ~14 tps flat to 32k. topk_frac>0 grows K with the
        # block count for >32k experiments — note the loose min/max bound means even
        # a fraction does not reliably retain the needle block at 64k+ (256+ blocks),
        # so dense/all-blocks is preferred there; see COMPRESSED_DECODE_OPTIMIZATION.md.
        # topk_blocks=0 disables routing entirely (attend every block).
        self.topk_blocks = int(os.environ.get("DIFFKV_TOPK_BLOCKS", "16"))
        self.topk_frac   = float(os.environ.get("DIFFKV_TOPK_FRAC", "0.0"))
        # High-Quality Mode (cross-runtime toggle, mirrors native src/main.cpp):
        #   DIFFKV_HIGH_QUALITY_ROUTING = 1 -> attend ALL compressed blocks at decode
        #     (max synthesis fidelity, cost scales with full context).
        #   unset/0/auto -> fast bounded-K routing (default; MLX already routes top-K,
        #     which is the whole reason its decode is flat-tps). MLX has no per-query
        #     detector, so "auto" resolves to fast here.
        _hq = os.environ.get("DIFFKV_HIGH_QUALITY_ROUTING", "0").strip().lower()
        self._high_quality_routing = _hq not in ("0", "", "false", "off", "auto")
        # Router for top-K selection:
        #   "residual" (default) — rank blocks by exact q·k over each block's anchor
        #     + its R most-distinctive residual keys. Tight and content/model-agnostic;
        #     reliably retains a buried-needle block even at 256+ blocks (64k+) where
        #     summary routers fail. DIFFKV_ROUTE_RESIDUALS = R; 0 (default) means use
        #     ALL residuals (= max_residual), which is what robustly recalls at 64k
        #     (a needle token can be a mid-rank residual). Lower R is faster but may
        #     drop deep needles; R=16 is enough through ~32k.
        #   "minmax" — cheaper Quest key min/max bound; faster but loose, drops the
        #     needle block past ~32k. Good when context stays small.
        self.router = os.environ.get("DIFFKV_ROUTER", "residual").lower()
        # DIFFKV_ROUTE_RESIDUALS = how many residuals per block the ROUTER scores when ranking
        # blocks (this is O(nb·R·D) PER TOKEN and is the dominant DECODE cost at long context —
        # confirmed 2026-07-04: route_once, which skips it, ~doubled 32k tps). It is SEPARATE
        # from max_residual (how many residuals are ATTENDED for the selected blocks). Default
        # was `max_residual`, so the prose bump (64→128) accidentally doubled router cost too.
        # Sweep (Qwen-1.5B, forced sparse): R=16/32 → NIAH ok but SYNTHESIS breaks (router can't
        # rank the diffuse paper blocks) at ~13/12 tps@32k; R=64 → NIAH AND synthesis both pass
        # at 9.5 tps@32k (+38% vs R=128's 6.9) — the sweet spot; R=128 → 6.9 tps. So DECOUPLE
        # and default to min(64, max_residual): keeps prose fidelity (128 attended) with a fast
        # router. Raise DIFFKV_ROUTE_RESIDUALS toward max_residual for 64k+ (deep mid-rank
        # needles) if recall drops there; 0 = use max_residual (the old, slower behavior).
        _rr = int(os.environ.get("DIFFKV_ROUTE_RESIDUALS", "0"))
        self.route_residuals = _rr if _rr > 0 else min(64, self.max_residual)
        # When on, exclude exact-residual token positions from the SVD reconstruction
        # pool so a captured token's ONLY representation at decode is its exact copy in
        # the dense pool — its lossy low-rank twin no longer dilutes it. Zero memory
        # cost; fixes precise-token (e.g. digit) corruption that the residual capture
        # was meant to prevent but couldn't because both copies were attended.
        self._res_exclude_svd = os.environ.get("DIFFKV_RESIDUAL_EXCLUDE_SVD", "1").strip().lower() not in ("0", "off", "false", "no")
        self.max_dense_len = self.recency_window + self.block_size
        self._comp_res_n_const = mx.full((self.max_blocks,), self.max_residual, dtype=mx.int32)

        # Revert default to 0 (disabled) to guarantee full routing correctness at every
        # token step. Leaving it enabled by default causes stale block routing over the
        # cache interval (16 tokens), which degrades factual table/structure extraction
        # accuracy and triggers repetitive loops.
        # To restore the decode cache speedup: DIFFKV_DECODE_CACHE=1
        self._decode_cache = os.environ.get("DIFFKV_DECODE_CACHE", "0") == "1"
        # DIFFKV_DECODE_FUSED=1 (default): single-SDPA-launch per layer per token over a
        # persistent fused buffer (blocks+residuals+dense window written once per route
        # interval, one-row in-place append per token). Removes the 2× per-token concat
        # + per-token mask build of the legacy decode-cache path. See
        # _execute_decode_cache for the buffer-dtype dial (DIFFKV_DECODE_FUSED_FP32).
        # =0 restores the old concat-per-token path bit-for-bit.
        self._decode_fused = os.environ.get("DIFFKV_DECODE_FUSED", "1") != "0"
        # Re-route + re-materialise every N tokens. Higher N = faster (less materialisation) but
        # staler block selection. Measured @32k: N=8→18, 16→20, 32→23 tps; NIAH exact + synthesis
        # reads paper at all three. 16 balances speed vs staleness for varied (chat) generation;
        # raise toward 32 for retrieval-heavy/long-answer workloads.
        self._decode_cache_interval = max(1, int(os.environ.get("DIFFKV_DECODE_CACHE_INTERVAL", "16")))

        # ── DIFFKV_SPARSE_PREFILL=1 — DSA/NSA-style block-sparse PREFILL (HANDOFF §DSA). ──
        # Prefill is otherwise dense O(L^2): every chunk attends over ALL preceding tokens.
        # With this on, a chunk instead attends to a SPARSE key set:
        #   [block 0 (attention sink) | top-K routed history blocks | recency window | self(causal)]
        # selected by a Quest-style min/max router over the raw (not-yet-compressed) KV. This is
        # training-free sparse attention (StreamingLLM sinks + MInference block retrieval) on a
        # frozen model, so it carries real accuracy risk — default OFF, gated by ctx, verified via
        # niah_recall before any default flip. Compute drops O(L^2)->O(L*K); memory is unchanged in
        # this stage (raw KV still held) — the memory win (drop compressed blocks' raw KV) is a
        # separate follow-up (see diffkv_on_the_fly_bugs.md).
        # DEFAULT ON (16th pass): flipped after the full guardrail suite passed sparse-ON — NIAH
        # sweep 4/4 (4k-32k), multi-depth 3/3, multi-needle 1/1, synthesis reads the paper,
        # relational 4/4. Reversible: DIFFKV_SPARSE_PREFILL=0. Compute-only (no memory change);
        # only engages above _sp_min_ctx tokens, no-op below.
        self._sparse_prefill = os.environ.get("DIFFKV_SPARSE_PREFILL", "1") != "0"
        # Only engage once the current chunk's start position is this far in — small prompts stay
        # dense (sparse only helps when there is enough history beyond the window to prune, and the
        # gather/route overhead only amortizes at long ctx).
        self._sp_min_ctx = int(os.environ.get("DIFFKV_SPARSE_PREFILL_MIN", "2048"))
        # Exact recency window (tokens, always attended, in addition to the current chunk).
        self._sp_window = int(os.environ.get("DIFFKV_SPARSE_PREFILL_WINDOW", "1024"))
        # Number of leading blocks kept as always-attended attention sinks (StreamingLLM).
        self._sp_sink_blocks = int(os.environ.get("DIFFKV_SPARSE_PREFILL_SINK_BLOCKS", "1"))
        # Top-K routed history blocks = max(KMIN, ceil(FRAC * nb)). These ON-defaults (8 / 0.05)
        # were verified on Qwen2.5-1.5B: NIAH bench 2/2 @16k+32k, multi-depth 3/3 @16k, multi-needle
        # 1/1 @16k, with 16k prefill -14% and 32k -31% vs dense. They are model/needle-tuned — a
        # diffuse-query or larger model may want a larger K (raise FRAC). The win grows with ctx
        # (attention's share of the forward grows with L).
        self._sp_kmin = int(os.environ.get("DIFFKV_SPARSE_PREFILL_KMIN", "8"))
        self._sp_frac = float(os.environ.get("DIFFKV_SPARSE_PREFILL_FRAC", "0.05"))
        self._sp_dbg = os.environ.get("DIFFKV_SPARSE_PREFILL_DBG", "0") == "1"

        # ── DIFFKV_LEGO_PREFILL=1 — streaming "lego-block" prefill (memory follow-up to
        # sparse prefill; see the note above about dropping compressed blocks' raw KV). ──
        # Sparse prefill prunes COMPUTE but still retains the full raw prompt KV cache
        # (routing reads raw keys), so prefill peak memory stays O(T) — and DiffKV holds
        # the compressed pool ON TOP of it, which is why prefill peak exceeds dense.
        # Lego mode instead builds the context like lego bricks: each finished block is
        # compressed on the fly (the existing per-chunk flush) into a self-contained
        # piece [anchor + low-rank deltas + exact residual "studs" + key min/max summary],
        # its raw KV is dropped, and every NEW chunk connects only to the top-K pieces
        # whose summaries score highest for its pooled query (the same residual router
        # decode uses). Chunk attention runs over
        #   [raw sink tokens | materialised top-K compressed blocks | raw recency RING | self(causal)]
        # so raw KV is bounded by O(sinks + ring + chunk) instead of O(T).
        # Engages only when (a) the chunk start is past LEGO_MIN_CTX, (b) compressed
        # blocks exist beyond the sinks, and (c) the TOTAL prompt will use compressed
        # decode (otherwise dense decode would need the full cache we stopped keeping).
        # Once engaged it is sticky for the session and the raw prompt cache is dropped.
        # DEFAULT OFF (flipped back 2026-07-12, reversing the earlier "synthesis
        # identical to lego-OFF" default-ON call above — that measurement predates
        # owner-capture/coverage residual selection and no longer holds). Re-measured
        # with CURRENT defaults (owner-capture + coverage residual selection): NIAH
        # 6/6 and multi-needle both identical lego ON/OFF (recall genuinely
        # unaffected — the claim above WAS right about that), but synthesis is NOT
        # identical: lego=1 scores 0.0/100 @8k vs lego=0's 6.7/100 (real-paper
        # linkage task; native shows the same pattern — margins/synthesis both cost
        # ~1 unit while memory drops 14-17%). This is a genuine memory-for-fidelity
        # trade, not a strict improvement, so it stays opt-in like native
        # (docs/NATIVE_LEGO_PORT_PLAN.md): DIFFKV_LEGO_PREFILL=1 for memory-
        # constrained long-context runs where the prefill-peak win (still real —
        # see the mechanism notes above) outweighs the synthesis cost.
        self._lego_prefill = os.environ.get("DIFFKV_LEGO_PREFILL", "0") == "1"
        self._lego_min_ctx = int(os.environ.get("DIFFKV_LEGO_MIN_CTX", str(self._sp_min_ctx)))
        self._lego_kmin = int(os.environ.get("DIFFKV_LEGO_KMIN", str(self._sp_kmin)))
        self._lego_frac = float(os.environ.get("DIFFKV_LEGO_FRAC", str(self._sp_frac)))
        # Router for the far-block selection: "minmax" (DEFAULT) is the Quest key
        # min/max bound over the stored block summaries — O(nb·D) per layer per
        # chunk, the same signal the validated sparse prefill routed with;
        # "residual" scores anchor + top-R exact residual keys per block
        # (decode-grade, O(nb·R·D) — ~64x the routing cost). A/B 2026-07-12: for
        # the POOLED chunk query the cheap bound is quality-identical (synthesis
        # 10.0/6.7 both, NIAH bench + multi-needle + depths same profile).
        self._lego_router = os.environ.get("DIFFKV_LEGO_ROUTER", "minmax").strip().lower()
        # Raw recency RING (tokens): the last RING tokens are kept raw (rolling,
        # block-aligned) and attended EXACTLY; lego pieces cover only the far field
        # beyond it. The ring is the prefill-memory dial: peak raw KV is O(RING),
        # not O(T). Fidelity note (2026-07-11): shadow parity shows lego attention
        # is CLOSER to exact dense than the validated sparse prefill at every
        # layer/chunk (16k, layer 27 final chunk: cos 0.9953 vs 0.9904), with or
        # without the ring dial — one knife-edge NIAH cell (16k/0.1) case-flips
        # ('DELTA'→'Delta', content correct) under any variant of this path; it is
        # margin-limited, not a content bug.
        _ring = int(os.environ.get("DIFFKV_LEGO_RING", "4096"))
        # Ring must cover at least the flush tail (recency_window + block) so the
        # exact region is contiguous up to the current chunk.
        self._lego_ring = max(_ring, self.recency_window + 2 * self.block_size)

        self.sessions = {}
        self.active_session_ids = ["default"]
        self.position_ids = None
        self._session_token_ids = {}
        self._session_checkpoints = {}
        self._session_srl = {}

        # ── Optional factual-store / SRL subsystem (WS2) ──────────────────────
        # The whole build→query→consume pipeline already exists in this file; it
        # was dead only because get_srl_state returned None and nothing built the
        # store. Gated behind DIFFKV_FACTUAL_STORE (default off) so the validated
        # sparse path is untouched until this is proven out.
        self._factual_enabled = os.environ.get("DIFFKV_FACTUAL_STORE", "0").strip().lower() in ("1", "on", "true", "yes")
        self._factual_stores: dict = {}
        self._prefill_kv_capture: dict = {}   # sid -> {layer_idx: [K_cpu, V_cpu]}
        self._pending_query: dict = {}        # sid -> query token ids (entity-binding hint)
        self._stop_token_ids: set = set()     # set by the wrapper after load
        self.tokenizer = None                 # set by the wrapper after load
        # Random projection W_proj [DESC_DIM, head_dim] for factual descriptors —
        # fixed at construction, normalized rows (mirrors KVRuntimeManager). torch
        # CPU so the factual store (torch-based) can run alongside the MLX kernels.
        self.W_proj = None
        if self._factual_enabled:
            import torch as _torch
            _desc_dim = 64
            _W = _torch.randn(_desc_dim, self.head_dim, dtype=_torch.float32)
            self.W_proj = _W / (_W.norm(dim=1, keepdim=True) + 1e-8)

    def get_srl_state(self, session_id: str):
        return self._session_srl.get(session_id)

    def capture_factual_prefill_kv(self, session_id: str, layer_idx: int, K_unrot: mx.array, V: mx.array):
        """Stash UNROTATED prefill K/V (layers 0 and middle only) as torch CPU
        tensors for FactualExactStore.build. Layer 0 supplies span descriptors +
        key norms; the middle layer supplies the Eagle look-back self-similarity.
        The store's descriptors must come from the SAME unrotated layer-0 K that the
        decode-time query uses as proxy Q, so the spaces are comparable."""
        if not self._factual_enabled:
            return
        mid = self.num_layers // 2
        if layer_idx not in (0, mid):
            return
        import numpy as _np, torch as _torch
        # K_unrot / V: mx.array [1, kv_heads, L, head_dim]
        k_t = _torch.from_numpy(_np.array(K_unrot.astype(mx.float32)))
        v_t = _torch.from_numpy(_np.array(V.astype(mx.float32)))
        cap = self._prefill_kv_capture.setdefault(session_id, {})
        if layer_idx not in cap:
            cap[layer_idx] = [k_t, v_t]
        else:
            cap[layer_idx][0] = _torch.cat([cap[layer_idx][0], k_t], dim=2)
            cap[layer_idx][1] = _torch.cat([cap[layer_idx][1], v_t], dim=2)

    def finalize_srl_index(self, session_id: str, cached_len: int = 0):
        """Build the SessionSRLState + FactualExactStore once, at the prefill→decode
        boundary, from the captured unrotated prefill K/V.

        WS2-full: also builds an InvertedTokenIndex (important_vocab + IDF) and passes
        it to the store so ENTITY ASSIGNMENT (RC4 distinguishing-token / IDF binding)
        and decode-time entity binding work — without it the store biases every
        matched fact equally and multi-entity generation loops. The query tokens that
        drive entity binding come from the caller-named question (`_pending_query`,
        set via generate(query_text=...)); otherwise the uncached tail of the prompt."""
        if not self._factual_enabled or session_id in self._session_srl:
            return
        cap = self._prefill_kv_capture.get(session_id)
        sess = self.sessions.get(session_id)
        token_ids = sess.get("token_ids") if sess else None
        if not cap or not token_ids or self.W_proj is None:
            return
        try:
            import torch as _torch
            from native_core.srl.session_srl_state import SessionSRLState
            from native_core.srl.factual_store import FactualExactStore
            from native_core.srl.inverted_index import build_inverted_index
            tok_ids = token_ids if isinstance(token_ids, _torch.Tensor) else _torch.tensor(list(token_ids), dtype=_torch.long)
            bs = self.block_size
            n_slots = max(1, (int(tok_ids.numel()) + bs - 1) // bs)
            slot_ids = list(range(n_slots))
            try:
                inv_index = build_inverted_index(tok_ids, slot_ids, bs, set(self._stop_token_ids))
                # Let the store decode tokens (sentence/line-boundary span splitting +
                # helper-word filtering in prime detection).
                if inv_index is not None and self.tokenizer is not None:
                    inv_index._tokenizer_ref = self.tokenizer
            except Exception:
                inv_index = None
            srl_state = SessionSRLState(
                semantic_index=None, chunk_graph=None, inverted_index=inv_index,
                ordered_slot_ids=slot_ids, sink_blocks=[],
            )
            # Entity-binding query tokens: the caller-named question if given, else
            # the uncached tail (whole prompt for a single-turn request).
            pq = self._pending_query.pop(session_id, None)
            srl_state.current_query_tokens = list(pq) if pq else list(token_ids[cached_len:])
            store = FactualExactStore(session_id)
            store.build(
                prefill_kv=cap,
                token_ids=tok_ids,
                W_proj=self.W_proj,
                stop_token_ids=self._stop_token_ids,
                slot_ids=slot_ids,
                block_size=bs,
                inv_index=inv_index,
            )
            srl_state.prompt_eagle_scores = getattr(store, "eagle_scores", None)
            try:
                srl_state.setup_sas_and_eqa(tok_ids, self._stop_token_ids, self.tokenizer)
            except Exception:
                pass
            self._session_srl[session_id] = srl_state
            self._factual_stores[session_id] = store
            if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
                _vocab = len(inv_index.important_vocab) if inv_index is not None else 0
                print(f"[FACTUAL] built session={session_id} entries={len(store.entries)} "
                      f"primes={sum(1 for e in store.entries if getattr(e,'is_prime',False))} "
                      f"vocab={_vocab} qtoks={len(srl_state.current_query_tokens)}", flush=True)
                if os.environ.get("DIFFKV_FACTUAL_DBG") == "1" and self.tokenizer is not None:
                    for e in store.entries:
                        _dec = self.tokenizer.decode(e.tokens)
                        if any(c.isdigit() for c in _dec):
                            print(f"[FENTRY] digit-span prime={getattr(e,'is_prime',False)} "
                                  f"eid={getattr(e,'entity_id',-1)} dist={getattr(e,'distinguishing_token',None)} "
                                  f"toks={_dec!r}", flush=True)
        except Exception as fe:
            import traceback; traceback.print_exc()
            print(f"[FACTUAL] WARNING: build failed for {session_id}: {fe}")
        finally:
            self._prefill_kv_capture.pop(session_id, None)

    def _create_empty_session(self, max_blocks: int = None) -> Dict[str, Any]:
        if max_blocks is None:
            max_blocks = self.max_blocks
        # Use float16 explicitly to halve the RAM vs float32 defaults
        dtype = mx.float16
        return {
            "max_blocks": max_blocks,
            "dense_keys":   [mx.zeros((1, self.kv_heads, self.max_dense_len, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "dense_values": [mx.zeros((1, self.kv_heads, self.max_dense_len, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "dense_lens":   [0 for _ in range(self.num_layers)],
            "dense_lens_mx": [mx.array(0, dtype=mx.int32) for _ in range(self.num_layers)],
            
            "prefill_K_chunks": [[] for _ in range(self.num_layers)],
            "prefill_V_chunks": [[] for _ in range(self.num_layers)],
            
            "num_blocks": [0 for _ in range(self.num_layers)],
            "comp_U":     [mx.zeros((max_blocks, self.block_size - 1, self.rank), dtype=dtype) for _ in range(self.num_layers)],
            "comp_VK":    [mx.zeros((max_blocks, self.kv_heads, self.rank, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_VV":    [mx.zeros((max_blocks, self.kv_heads, self.rank, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_anc_k": [mx.zeros((max_blocks, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_anc_v": [mx.zeros((max_blocks, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            # Per-block element-wise key min/max — one of two signals the top-K
            # router uses: a Quest-style upper bound on the block's max q·k. It is
            # cheap but LOOSE (over-estimates at large block counts), so the router
            # also scores the exact residual keys to reliably rank needle blocks.
            "comp_min_k": [mx.zeros((max_blocks, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_max_k": [mx.zeros((max_blocks, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_scale":    [mx.zeros((max_blocks,)) for _ in range(self.num_layers)],
            "comp_seq_len": [mx.zeros((max_blocks,), dtype=mx.int32) for _ in range(self.num_layers)],
            
            "comp_res_k": [mx.zeros((max_blocks, self.max_residual, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_res_v": [mx.zeros((max_blocks, self.max_residual, self.kv_heads, self.head_dim), dtype=dtype) for _ in range(self.num_layers)],
            "comp_res_n": [[0] * max_blocks for _ in range(self.num_layers)],
            # Per-block boolean mask of which delta positions (0..block_size-2) are kept
            # as EXACT residuals — used to exclude them from the SVD pool at decode.
            "comp_res_mask": [mx.zeros((max_blocks, self.block_size - 1), dtype=mx.bool_) for _ in range(self.num_layers)],
            
            "token_ids": [],
            "token_counts": Counter()
        }

    def init_session(self, session_id: str, prefill_len: int = 0, max_tokens_hint: int = None):
        if session_id not in self.sessions:
            max_gen = max_tokens_hint if max_tokens_hint is not None else 2048
            expected_total_len = prefill_len + max_gen
            needed_blocks = math.ceil(expected_total_len / self.block_size)
            sess_max_blocks = min(self.max_blocks, max(1, needed_blocks))
            self.sessions[session_id] = self._create_empty_session(sess_max_blocks)

    def _ensure_block_capacity(self, session: Dict, need_blocks: int) -> int:
        """Grow the per-layer comp_* pools so the session can hold `need_blocks`
        compressed blocks. Sessions are created right-sized from the prefill-length
        hint; growth covers hint misses (generations longer than the hint, reused
        sessions) that previously either clamped the flush — silently dropping
        compression — or evicted the oldest block. Doubles capacity (amortized O(1)
        copies), capped at the global self.max_blocks. Returns the new capacity."""
        cur = session.get("max_blocks", self.max_blocks)
        if need_blocks <= cur or cur >= self.max_blocks:
            return cur
        new_cap = min(self.max_blocks, max(need_blocks, cur * 2))
        grow = new_cap - cur
        f16 = mx.float16
        grown_tails = {
            "comp_U":       ((self.block_size - 1, self.rank), f16),
            "comp_VK":      ((self.kv_heads, self.rank, self.head_dim), f16),
            "comp_VV":      ((self.kv_heads, self.rank, self.head_dim), f16),
            "comp_anc_k":   ((self.kv_heads, self.head_dim), f16),
            "comp_anc_v":   ((self.kv_heads, self.head_dim), f16),
            "comp_min_k":   ((self.kv_heads, self.head_dim), f16),
            "comp_max_k":   ((self.kv_heads, self.head_dim), f16),
            "comp_scale":   ((), mx.float32),
            "comp_seq_len": ((), mx.int32),
            "comp_res_k":   ((self.max_residual, self.kv_heads, self.head_dim), f16),
            "comp_res_v":   ((self.max_residual, self.kv_heads, self.head_dim), f16),
            "comp_res_mask": ((self.block_size - 1,), mx.bool_),
        }
        for l in range(self.num_layers):
            for key, (tail, dt) in grown_tails.items():
                if key not in session:
                    continue
                pad = mx.zeros((grow,) + tail, dtype=dt)
                session[key][l] = mx.concatenate([session[key][l], pad], axis=0)
            session["comp_res_n"][l] = session["comp_res_n"][l] + [0] * grow
        session["max_blocks"] = new_cap
        return new_cap

    def clear_session(self, session_id: str):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.clear_session(session_id)
        self.sessions.pop(session_id, None)
        if hasattr(self, "patched_model") and self.patched_model is not None:
            cache_key = (session_id,)
            if cache_key in self.patched_model._prefill_caches:
                del self.patched_model._prefill_caches[cache_key]
            if cache_key in self.patched_model._prev_was_prefill:
                del self.patched_model._prev_was_prefill[cache_key]
            self.patched_model._decode_compressed.pop(cache_key, None)

    def snapshot_session(self, session_id: str, checkpoint_id: str):
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found to snapshot.")
        src = self.sessions[session_id]
        max_b = src.get("max_blocks", self.max_blocks)
        self._session_checkpoints[checkpoint_id] = {
            "max_blocks": max_b,
            "dense_keys": [mx.array(k) for k in src["dense_keys"]],
            "dense_values": [mx.array(v) for v in src["dense_values"]],
            "dense_lens": src["dense_lens"].copy(),
            "dense_lens_mx": [mx.array(dl) for dl in src["dense_lens_mx"]] if "dense_lens_mx" in src else [mx.array(dl, dtype=mx.int32) for dl in src["dense_lens"]],
            "num_blocks": src["num_blocks"].copy(),
            "comp_U": [mx.array(u) for u in src["comp_U"]],
            "comp_VK": [mx.array(vk) for vk in src["comp_VK"]],
            "comp_VV": [mx.array(vv) for vv in src["comp_VV"]],
            "comp_anc_k": [mx.array(ak) for ak in src["comp_anc_k"]],
            "comp_anc_v": [mx.array(av) for av in src["comp_anc_v"]],
            "comp_min_k": [mx.array(a) for a in src["comp_min_k"]],
            "comp_max_k": [mx.array(a) for a in src["comp_max_k"]],
            "comp_scale": [mx.array(s) for s in src["comp_scale"]],
            "comp_seq_len": [mx.array(sl) for sl in src["comp_seq_len"]],
            "comp_res_k": [mx.array(rk) for rk in src["comp_res_k"]],
            "comp_res_v": [mx.array(rv) for rv in src["comp_res_v"]],
            "comp_res_n": [list(rn) for rn in src["comp_res_n"]],
            "comp_res_mask": [mx.array(rm) for rm in src["comp_res_mask"]] if "comp_res_mask" in src else [mx.zeros((max_b, self.block_size - 1), dtype=mx.bool_) for _ in range(self.num_layers)],
            "token_ids": src["token_ids"].copy() if "token_ids" in src else [],
            "token_counts": Counter(src["token_ids"]) if "token_ids" in src else Counter()
        }
        if hasattr(self, "patched_model") and self.patched_model is not None:
            if not hasattr(self, "_session_checkpoints_prompt_cache"):
                self._session_checkpoints_prompt_cache = {}
            src_key = (session_id,)
            if src_key in self.patched_model._prefill_caches:
                from mlx_lm.models.cache import KVCache
                src_cache = self.patched_model._prefill_caches[src_key]
                dst_cache = []
                for layer_cache in src_cache:
                    new_layer = KVCache()
                    if layer_cache.keys is not None:
                        new_layer.keys = mx.array(layer_cache.keys)
                    if layer_cache.values is not None:
                        new_layer.values = mx.array(layer_cache.values)
                    if hasattr(layer_cache, "offset"):
                        new_layer.offset = layer_cache.offset
                    if hasattr(layer_cache, "step"):
                        new_layer.step = layer_cache.step
                    dst_cache.append(new_layer)
                self._session_checkpoints_prompt_cache[checkpoint_id] = dst_cache
 
    def restore_session(self, session_id: str, checkpoint_id: str):
        if checkpoint_id not in self._session_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
        ckpt = self._session_checkpoints[checkpoint_id]
        max_b = ckpt.get("max_blocks", self.max_blocks)
        self.sessions[session_id] = {
            "max_blocks": max_b,
            "dense_keys": [mx.array(k) for k in ckpt["dense_keys"]],
            "dense_values": [mx.array(v) for v in ckpt["dense_values"]],
            "dense_lens": ckpt["dense_lens"].copy(),
            "dense_lens_mx": [mx.array(dl) for dl in ckpt["dense_lens_mx"]] if "dense_lens_mx" in ckpt else [mx.array(dl, dtype=mx.int32) for dl in ckpt["dense_lens"]],
            "num_blocks": ckpt["num_blocks"].copy(),
            "comp_U": [mx.array(u) for u in ckpt["comp_U"]],
            "comp_VK": [mx.array(vk) for vk in ckpt["comp_VK"]],
            "comp_VV": [mx.array(vv) for vv in ckpt["comp_VV"]],
            "comp_anc_k": [mx.array(ak) for ak in ckpt["comp_anc_k"]],
            "comp_anc_v": [mx.array(av) for av in ckpt["comp_anc_v"]],
            "comp_min_k": [mx.array(a) for a in ckpt["comp_min_k"]],
            "comp_max_k": [mx.array(a) for a in ckpt["comp_max_k"]],
            "comp_scale": [mx.array(s) for s in ckpt["comp_scale"]],
            "comp_seq_len": [mx.array(sl) for sl in ckpt["comp_seq_len"]],
            "comp_res_k": [mx.array(rk) for rk in ckpt["comp_res_k"]],
            "comp_res_v": [mx.array(rv) for rv in ckpt["comp_res_v"]],
            "comp_res_n": [list(rn) for rn in ckpt["comp_res_n"]],
            "comp_res_mask": [mx.array(rm) for rm in ckpt["comp_res_mask"]] if "comp_res_mask" in ckpt else [mx.zeros((max_b, self.block_size - 1), dtype=mx.bool_) for _ in range(self.num_layers)],
            "token_ids": ckpt["token_ids"].copy() if "token_ids" in ckpt else [],
            "token_counts": Counter(ckpt["token_ids"]) if "token_ids" in ckpt else Counter()
        }
        if hasattr(self, "patched_model") and self.patched_model is not None:
            if hasattr(self, "_session_checkpoints_prompt_cache") and checkpoint_id in self._session_checkpoints_prompt_cache:
                from mlx_lm.models.cache import KVCache
                ckpt_cache = self._session_checkpoints_prompt_cache[checkpoint_id]
                dst_cache = []
                for layer_cache in ckpt_cache:
                    new_layer = KVCache()
                    if layer_cache.keys is not None:
                        new_layer.keys = mx.array(layer_cache.keys)
                    if layer_cache.values is not None:
                        new_layer.values = mx.array(layer_cache.values)
                    if hasattr(layer_cache, "offset"):
                        new_layer.offset = layer_cache.offset
                    if hasattr(layer_cache, "step"):
                        new_layer.step = layer_cache.step
                    dst_cache.append(new_layer)
                self.patched_model._prefill_caches[(session_id,)] = dst_cache

    def delete_checkpoint(self, checkpoint_id: str):
        self._session_checkpoints.pop(checkpoint_id, None)
        if hasattr(self, "_session_checkpoints_prompt_cache"):
            self._session_checkpoints_prompt_cache.pop(checkpoint_id, None)

    def get_streaming_summary(self, session_id: str = None) -> dict:
        return {"streaming_ingest": False}

    def get_streaming_blocks(self, session_id: str, layer_idx: int) -> list:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        comp_len = session["num_blocks"][layer_idx] * self.block_size
        dense_len = session["dense_lens"][layer_idx]
        total_len = comp_len + dense_len
        num_logical_blocks = (total_len + self.block_size - 1) // self.block_size
        return [object() for _ in range(num_logical_blocks)]

    def get_raw_blocks(self, session_id: str, layer_idx: int) -> list:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        comp_len = session["num_blocks"][layer_idx] * self.block_size
        dense_len = session["dense_lens"][layer_idx]
        total_len = comp_len + dense_len
        num_logical_blocks = (total_len + self.block_size - 1) // self.block_size
        return [object() for _ in range(num_logical_blocks)]

    def rollback_session(self, session_id: str, target_len: int, clear_srl: bool = False):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.rollback_session(session_id, target_len, clear_srl=clear_srl)
        session = self.sessions.get(session_id)
        if session is None:
            return

        # Block set is about to change — drop the cached residual gather AND the
        # decode-cache (materialised fused buffers). Without this, a rollback that
        # lands on the SAME block count would keep serving the stale pre-rollback
        # cache for up to a full route interval.
        if "_res_cache" in session:
            session["_res_cache"].clear()
        if "_cache_kv" in session:
            session["_cache_kv"].clear()

        for layer_idx in range(self.num_layers):
            num_blocks = session["num_blocks"][layer_idx]
            comp_len = num_blocks * self.block_size
            
            if target_len <= comp_len:
                # Discard compressed blocks from the end
                keep_blocks = target_len // self.block_size
                session["num_blocks"][layer_idx] = keep_blocks
                # Remaining tokens in dense window
                dense_len = target_len - (keep_blocks * self.block_size)
                session["dense_lens"][layer_idx] = dense_len
                
                # Zero out discarded dense tokens and blocks
                session["dense_keys"][layer_idx][0, :, dense_len:] = 0.0
                session["dense_values"][layer_idx][0, :, dense_len:] = 0.0
                session["comp_U"][layer_idx][keep_blocks:] = 0.0
                session["comp_VK"][layer_idx][keep_blocks:] = 0.0
                session["comp_VV"][layer_idx][keep_blocks:] = 0.0
                session["comp_anc_k"][layer_idx][keep_blocks:] = 0.0
                session["comp_anc_v"][layer_idx][keep_blocks:] = 0.0
                session["comp_min_k"][layer_idx][keep_blocks:] = 0.0
                session["comp_max_k"][layer_idx][keep_blocks:] = 0.0
                session["comp_scale"][layer_idx][keep_blocks:] = 0.0
                session["comp_seq_len"][layer_idx][keep_blocks:] = 0
                session["comp_res_k"][layer_idx][keep_blocks:] = 0.0
                session["comp_res_v"][layer_idx][keep_blocks:] = 0.0
                if "comp_res_mask" in session:
                    session["comp_res_mask"][layer_idx][keep_blocks:] = False
                for b_i in range(keep_blocks, session.get("max_blocks", self.max_blocks)):
                    session["comp_res_n"][layer_idx][b_i] = 0
            else:
                # Only slice dense window
                dense_len = target_len - comp_len
                session["dense_lens"][layer_idx] = dense_len
                session["dense_keys"][layer_idx][0, :, dense_len:] = 0.0
                session["dense_values"][layer_idx][0, :, dense_len:] = 0.0
            
            mx.eval(
                session["dense_keys"][layer_idx],
                session["dense_values"][layer_idx],
                session["comp_U"][layer_idx],
                session["comp_VK"][layer_idx],
                session["comp_VV"][layer_idx],
                session["comp_anc_k"][layer_idx],
                session["comp_anc_v"][layer_idx],
                session["comp_min_k"][layer_idx],
                session["comp_max_k"][layer_idx],
                session["comp_res_k"][layer_idx],
                session["comp_res_v"][layer_idx]
            )
                
        if "token_ids" in session and session["token_ids"]:
            session["token_ids"] = session["token_ids"][:target_len]
            session["token_counts"] = Counter(session["token_ids"])

        if hasattr(self, "patched_model") and self.patched_model is not None:
            cache_key = (session_id,)
            if cache_key in self.patched_model._prefill_caches:
                for layer_cache in self.patched_model._prefill_caches[cache_key]:
                    if hasattr(layer_cache, "trim"):
                        layer_cache.trim(target_len)

    def clone_session(self, src_sid: str, dst_sid: str):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.clone_session(src_sid, dst_sid)
        if src_sid not in self.sessions:
            return
        src = self.sessions[src_sid]
        self.sessions[dst_sid] = {
            "dense_keys": [mx.array(k) for k in src["dense_keys"]],
            "dense_values": [mx.array(v) for v in src["dense_values"]],
            "dense_lens": src["dense_lens"].copy(),
            "dense_lens_mx": [mx.array(dl) for dl in src["dense_lens_mx"]] if "dense_lens_mx" in src else [mx.array(dl, dtype=mx.int32) for dl in src["dense_lens"]],
            "num_blocks": src["num_blocks"].copy(),
            "comp_U": [mx.array(u) for u in src["comp_U"]],
            "comp_VK": [mx.array(vk) for vk in src["comp_VK"]],
            "comp_VV": [mx.array(vv) for vv in src["comp_VV"]],
            "comp_anc_k": [mx.array(ak) for ak in src["comp_anc_k"]],
            "comp_anc_v": [mx.array(av) for av in src["comp_anc_v"]],
            "comp_min_k": [mx.array(a) for a in src["comp_min_k"]],
            "comp_max_k": [mx.array(a) for a in src["comp_max_k"]],
            "comp_scale": [mx.array(s) for s in src["comp_scale"]],
            "comp_seq_len": [mx.array(sl) for sl in src["comp_seq_len"]],
            "comp_res_k": [mx.array(rk) for rk in src["comp_res_k"]],
            "comp_res_v": [mx.array(rv) for rv in src["comp_res_v"]],
            "comp_res_n": [list(rn) for rn in src["comp_res_n"]],
            "comp_res_mask": [mx.array(rm) for rm in src["comp_res_mask"]] if "comp_res_mask" in src else [mx.zeros((self.max_blocks, self.block_size - 1), dtype=mx.bool_) for _ in range(self.num_layers)],
            "token_ids": src["token_ids"].copy() if "token_ids" in src else [],
            "token_counts": Counter(src["token_ids"]) if "token_ids" in src else Counter()
        }
        if hasattr(self, "patched_model") and self.patched_model is not None:
            src_key = (src_sid,)
            dst_key = (dst_sid,)
            if src_key in self.patched_model._prefill_caches:
                from mlx_lm.models.cache import KVCache
                src_cache = self.patched_model._prefill_caches[src_key]
                dst_cache = []
                for layer_cache in src_cache:
                    new_layer = KVCache()
                    if layer_cache.keys is not None:
                        new_layer.keys = mx.array(layer_cache.keys)
                    if layer_cache.values is not None:
                        new_layer.values = mx.array(layer_cache.values)
                    if hasattr(layer_cache, "offset"):
                        new_layer.offset = layer_cache.offset
                    if hasattr(layer_cache, "step"):
                        new_layer.step = layer_cache.step
                    dst_cache.append(new_layer)
                self.patched_model._prefill_caches[dst_key] = dst_cache
                if src_key in self.patched_model._prev_was_prefill:
                    self.patched_model._prev_was_prefill[dst_key] = self.patched_model._prev_was_prefill[src_key]

    def get_session_sequence_length(self, session_id: str) -> int:
        session = self.sessions.get(session_id)
        if session is None:
            return 0
        comp_len = session["num_blocks"][0] * self.block_size
        dense_len = session["dense_lens"][0]
        return comp_len + dense_len

    def register_prefill_tokens(self, session_id: str, token_ids: torch.Tensor):
        session = self._get_or_create_session(session_id)
        tids_list = token_ids.cpu().tolist()
        session["token_ids"].extend(tids_list)
        if "token_counts" not in session or session["token_counts"] is None:
            session["token_counts"] = Counter()
        session["token_counts"].update(tids_list)

    def finalize_compressed_blocks(self):
        pass

    def compress_deferred_prefill_blocks(self, session_id: str):
        """Flush stashed prefill K/V through the batched SVD compressor.

        STREAMING semantics: safe (and intended) to call after EVERY prefill
        chunk. Each call virtually concatenates [current dense tail | newly
        stashed chunks], compresses every full block that clears the recency
        window (one batched SVD across layers x blocks), and leaves the
        remainder in the dense buffers. This keeps peak uncompressed KV
        bounded by ~(recency_window + chunk) tokens instead of the whole
        prompt — the difference between fitting and not fitting 32k+ prefill
        in 8GB. Calling it once at end-of-prefill is numerically equivalent.

        Invariant relied on here and in _compress_block: compressed blocks
        tile the prompt contiguously from token 0, so global block index b
        covers tokens [b*block_size, (b+1)*block_size).
        """
        session = self.sessions.get(session_id)
        if session is None:
            return

        # 1. Virtually concatenate [dense tail | stashed chunks] per layer.
        #    The tail is whatever survived the previous flush uncompressed;
        #    without it, per-chunk calls would clobber prior chunks (the
        #    2026-07-02 regression: only the last 512 tokens survived prefill).
        K_all_layers = []
        V_all_layers = []
        for l in range(self.num_layers):
            if not session["prefill_K_chunks"][l]:
                return
            parts_k = list(session["prefill_K_chunks"][l])
            parts_v = list(session["prefill_V_chunks"][l])
            tail = session["dense_lens"][l]
            if tail > 0:
                parts_k.insert(0, mx.expand_dims(session["dense_keys"][l][0, :, :tail], 0))
                parts_v.insert(0, mx.expand_dims(session["dense_values"][l][0, :, :tail], 0))
            K_all_layers.append(mx.concatenate(parts_k, axis=2) if len(parts_k) > 1 else parts_k[0])
            V_all_layers.append(mx.concatenate(parts_v, axis=2) if len(parts_v) > 1 else parts_v[0])
            # Clear stashed chunks
            session["prefill_K_chunks"][l] = []
            session["prefill_V_chunks"][l] = []

        L = K_all_layers[0].shape[2]

        # 2. How many full blocks clear the recency window?
        num_blocks = (L - self.recency_window) // self.block_size

        # Session pool capacity guard (reused sessions can outgrow their
        # allocation): grow the pools first; clamp only at the global cap.
        max_b = session.get("max_blocks", self.max_blocks)
        start_blocks = session["num_blocks"][0]
        if num_blocks > 0 and start_blocks + num_blocks > max_b:
            max_b = self._ensure_block_capacity(session, start_blocks + num_blocks)
        if num_blocks > 0 and start_blocks + num_blocks > max_b:
            print(f"[DiffKV MLX] WARNING: session '{session_id}' block pool full "
                  f"({start_blocks}+{num_blocks} > {max_b}); clamping flush.")
            num_blocks = max(0, max_b - start_blocks)

        if num_blocks <= 0:
            # Nothing clears the window yet: everything (tail + new) stays
            # dense. num_blocks<=0 implies L < recency_window + block_size,
            # so this always fits max_dense_len.
            for l in range(self.num_layers):
                L_dense = L
                session["dense_keys"][l][0, :, :L_dense]   = K_all_layers[l].squeeze(0)
                session["dense_values"][l][0, :, :L_dense] = V_all_layers[l].squeeze(0)
                session["dense_lens"][l] = L_dense
                session["dense_lens_mx"][l] = mx.array(L_dense, dtype=mx.int32)
            return

        N_comp = num_blocks * self.block_size
        S_comp = self.block_size - 1
        B_batch = self.num_layers * num_blocks
        
        # 3. Build deltas for all blocks across all layers
        accum_deltas_k = []
        accum_deltas_v = []
        accum_anchors_k = []
        accum_anchors_v = []
        accum_blocks_k = []
        accum_blocks_v = []
        
        for l in range(self.num_layers):
            K_all = K_all_layers[l]
            V_all = V_all_layers[l]
            
            K_comp = K_all[:, :, :N_comp, :]
            V_comp = V_all[:, :, :N_comp, :]
            
            # Shape: (H_kv, num_blocks, block_size, D)
            K_comp_blocks = K_comp.squeeze(0).reshape(self.kv_heads, num_blocks, self.block_size, self.head_dim).transpose(1, 0, 2, 3)
            V_comp_blocks = V_comp.squeeze(0).reshape(self.kv_heads, num_blocks, self.block_size, self.head_dim).transpose(1, 0, 2, 3)
            
            anchor_k = K_comp_blocks[:, :, 0, :]  # (num_blocks, H_kv, D)
            anchor_v = V_comp_blocks[:, :, 0, :]  # (num_blocks, H_kv, D)
            
            deltas_k = K_comp_blocks[:, :, 1:, :] - mx.expand_dims(anchor_k, 2)  # (num_blocks, H_kv, S_comp, D)
            deltas_v = V_comp_blocks[:, :, 1:, :] - mx.expand_dims(anchor_v, 2)  # (num_blocks, H_kv, S_comp, D)
            
            # (num_blocks, S_comp, H_kv * D)
            deltas_k_2d = deltas_k.transpose(0, 2, 1, 3).reshape(num_blocks, S_comp, -1)
            deltas_v_2d = deltas_v.transpose(0, 2, 1, 3).reshape(num_blocks, S_comp, -1)
            
            accum_deltas_k.append(deltas_k_2d)
            accum_deltas_v.append(deltas_v_2d)
            accum_anchors_k.append(anchor_k)
            accum_anchors_v.append(anchor_v)
            accum_blocks_k.append(K_comp_blocks)
            accum_blocks_v.append(V_comp_blocks)
            
        # Concatenate across layers
        batch_deltas_k = mx.concatenate(accum_deltas_k, axis=0)  # (B_batch, S_comp, H_kv * D)
        batch_deltas_v = mx.concatenate(accum_deltas_v, axis=0)  # (B_batch, S_comp, H_kv * D)
        batch_anchors_k = mx.concatenate(accum_anchors_k, axis=0)  # (B_batch, H_kv, D)
        batch_anchors_v = mx.concatenate(accum_anchors_v, axis=0)  # (B_batch, H_kv, D)
        batch_blocks_k = mx.concatenate(accum_blocks_k, axis=0)  # (B_batch, H_kv, block_size, D)
        batch_blocks_v = mx.concatenate(accum_blocks_v, axis=0)  # (B_batch, H_kv, block_size, D)
        
        # 4. V-side rebalancing for the joint K|V SVD
        v_scale_on = os.environ.get("DIFFKV_V_SCALE", "1") != "0"
        v_gain = 1.0
        if v_scale_on:
            eK = mx.sum(batch_deltas_k.astype(mx.float32)**2, axis=(1, 2))
            eV = mx.sum(batch_deltas_v.astype(mx.float32)**2, axis=(1, 2))
            
            v_gain = mx.sqrt(eK / mx.maximum(eV, 1e-12))
            v_gain = mx.minimum(mx.maximum(v_gain, 1.0), 10000.0)
            
            v_gain_broadcast = mx.expand_dims(mx.expand_dims(v_gain, 1), 2)  # (B_batch, 1, 1)
            batch_deltas_v_scaled = batch_deltas_v * v_gain_broadcast
            batch_deltas = mx.concatenate([batch_deltas_k, batch_deltas_v_scaled], axis=2)
        else:
            batch_deltas = mx.concatenate([batch_deltas_k, batch_deltas_v], axis=2)
            
        token_norms = mx.linalg.norm(batch_deltas, axis=-1, keepdims=True)
        token_norms = mx.maximum(token_norms, 1e-5)
        batch_deltas_normalized = batch_deltas / token_norms
        
        # 5. Batched GPU SVD
        U_batch, Vh_batch, scales_batch = compress_mlx_block_batched(batch_deltas_normalized, self.rank)
        
        U_batch = U_batch * token_norms  # U_batch shape: (B_batch, S_comp, rank)
        
        # Split Vh_batch back into VK and VV
        VK_flat = Vh_batch[:, :, :self.kv_heads * self.head_dim]
        VV_flat = Vh_batch[:, :, self.kv_heads * self.head_dim:]
        
        if v_scale_on:
            # Unscale V components
            v_gain_div = mx.expand_dims(mx.expand_dims(v_gain, 1), 2)
            VV_flat = VV_flat / v_gain_div
            
        # Reshape to (B_batch, H_kv, rank, D)
        VK_batch = VK_flat.reshape(B_batch, self.rank, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        VV_batch = VV_flat.reshape(B_batch, self.rank, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 6. SVD Residual Correction (batched)
        recon_delta = mx.matmul(U_batch, Vh_batch) * mx.expand_dims(mx.expand_dims(scales_batch, 1), 2)
        recon_delta_k = recon_delta[:, :, :self.kv_heads * self.head_dim]
        recon_delta_v = recon_delta[:, :, self.kv_heads * self.head_dim:]
        if v_scale_on:
            recon_delta_v = recon_delta_v / v_gain_broadcast
            
        errors_k = mx.linalg.norm(batch_deltas_k - recon_delta_k, axis=-1)
        errors_v = mx.linalg.norm(batch_deltas_v - recon_delta_v, axis=-1)
        
        if v_scale_on:
            errors_v_balanced = errors_v * mx.expand_dims(v_gain, 1)
        else:
            errors_v_balanced = errors_v
        joint_errors = mx.sqrt(errors_k**2 + errors_v_balanced**2)
        
        # Content-aware residual capture / token boosting
        tok_boost_env = os.environ.get("DIFFKV_RESIDUAL_TOKEN_BOOST")
        tok_boost = 8.0
        if tok_boost_env is not None:
            try:
                tok_boost = float(tok_boost_env)
            except ValueError:
                pass
                
        boost_rows = []   # per-block final boost multipliers (also feeds the budget floor)
        if os.environ.get("DIFFKV_DBG_TABLE") == "1":
            print(f"[DBG_TABLE/gate-batched] tok_boost={tok_boost} "
                  f"has_token_ids={'token_ids' in session} "
                  f"len={len(session.get('token_ids', []))} "
                  f"num_blocks={num_blocks} start_blocks={start_blocks} S_comp={S_comp}",
                  flush=True)
        if tok_boost > 1.0 and "token_ids" in session and len(session["token_ids"]) > 0:
            # Boosts depend only on token ids, so compute once per BLOCK and
            # tile across layers (28x less work), with a persistent per-token
            # decode cache (tokenizer.decode dominated 32k prefill otherwise).
            counts = session.get("token_counts", {})
            total_tokens = len(session["token_ids"])
            decode_cache = getattr(self, "_tok_decode_cache", None)
            if decode_cache is None:
                decode_cache = self._tok_decode_cache = {}
            # No size cap: this is keyed by TOKEN ID, not by position or session
            # length, so it is naturally bounded by vocabulary size (~152k for
            # Qwen2.5, worst case ~20MB of short strings) regardless of how long
            # a session runs. A periodic clear() here would evict good entries
            # and force repeat tokenizer.decode() calls on a long/diverse
            # document — undoing the exact cost this cache exists to avoid.
            boost_rows = []
            for block_idx in range(num_blocks):
                # Global block index: blocks tile the prompt from token 0 and
                # start_blocks blocks were compressed by earlier flushes.
                abs_start = (start_blocks + block_idx) * self.block_size
                tids = session["token_ids"][abs_start + 1 : abs_start + self.block_size]

                boost_multipliers = [1.0] * S_comp
                if len(tids) == S_comp:
                    tok_strs = []
                    for tid in tids:
                        s = decode_cache.get(tid)
                        if s is None:
                            s = decode_cache[tid] = self.tokenizer.decode([tid])
                        tok_strs.append(s)

                    is_core = []
                    for s in tok_strs:
                        s_clean = s.strip()
                        has_digit = any(c.isdigit() for c in s_clean)
                        is_upper = s_clean.isupper() and s_clean.isalpha() and len(s_clean) >= 2
                        is_core.append(has_digit or is_upper or s_clean == '-' or s_clean == '_')

                    is_prose = []
                    for s in tok_strs:
                        s_clean = s.strip()
                        if not s_clean:
                            is_prose.append(True)
                            continue
                        if s_clean in ('.', ',', ';', '?', '!', ':', '"', "'", '(', ')', '[', ']', '{', '}'):
                            is_prose.append(True)
                            continue
                        if s_clean.isalpha():
                            if s_clean.islower() or (s_clean.istitle() and len(s_clean) > 1):
                                is_prose.append(True)
                                continue
                        is_prose.append(False)

                    in_segment = False
                    segment_indices = []
                    for i in range(S_comp):
                        if not is_prose[i]:
                            if not in_segment:
                                in_segment = True
                                segment_indices.append([i])
                            else:
                                segment_indices[-1].append(i)
                        else:
                            in_segment = False

                    for seg in segment_indices:
                        contains_core = any(is_core[i] for i in seg)
                        if contains_core:
                            for i in seg:
                                tid = tids[i]
                                count = counts.get(tid, 1)
                                idf = math.log(max(total_tokens, 2) / (count + 0.1))
                                rarity_weight = max(1.0, min(idf, 6.0))
                                boost_multipliers[i] = tok_boost * (rarity_weight / 2.0)

                    # Owner capture: the fact's entity name gets the same
                    # exact-residual treatment as its value (see helper doc).
                    _apply_owner_capture(boost_multipliers, segment_indices, is_core,
                                         tok_strs, tids, counts, total_tokens, tok_boost)

                    # Table capture: whole table-like lines (incl. header/unit
                    # cells) enter the exact set with priority (see helper doc).
                    n_tab = _apply_table_capture(boost_multipliers, tok_strs, tids,
                                                 counts, total_tokens, tok_boost)
                    if os.environ.get("DIFFKV_DBG_TABLE") == "1":
                        if n_tab:
                            if not hasattr(self, "_dbg_table_blocks"):
                                self._dbg_table_blocks = {}
                            self._dbg_table_blocks[start_blocks + block_idx] = \
                                _detect_table_rows(tok_strs)
                        if any('|' in s for s in tok_strs):
                            _joined = "".join(tok_strs)
                            print(f"[DBG_TABLE/content] block {start_blocks + block_idx} "
                                  f"n_tab={n_tab} text={_joined[:180]!r}", flush=True)

                    # Apply window boost (Phase 2: contiguous runs)
                    final_boosts = list(boost_multipliers)
                    W = 2
                    for idx in range(S_comp):
                        if boost_multipliers[idx] > 1.0:
                            for j in range(max(0, idx - W), min(S_comp, idx + W + 1)):
                                final_boosts[j] = max(final_boosts[j], boost_multipliers[idx])
                    boost_multipliers = final_boosts

                    # Relational edge capture: pin the connectives that BIND the
                    # captured nodes (negation/limit/equivalence/causal) into the
                    # exact set so the SVD pool can't rebound them (see helper doc).
                    _apply_relational_capture(boost_multipliers, tok_strs, tids,
                                              counts, total_tokens, tok_boost)
                boost_rows.append(boost_multipliers)
            # b = layer*num_blocks + block  →  tile the per-block rows per layer
            boost_np = np.tile(np.asarray(boost_rows, dtype=np.float32), (self.num_layers, 1))
            boost_mx = mx.array(boost_np).astype(joint_errors.dtype)
            joint_errors = joint_errors * boost_mx
            
        # OPT-A: Adaptive residual budget — MLX prefill port.
        batch_norms_k = mx.linalg.norm(batch_deltas_k, axis=-1)
        batch_norms_v = mx.linalg.norm(batch_deltas_v, axis=-1)
        rel_error_k = errors_k / mx.maximum(batch_norms_k, 1e-8)
        rel_error_v = errors_v / mx.maximum(batch_norms_v, 1e-8)

        # MLX lacks batched mx.median; sort and retrieve the middle element per block.
        sorted_k = mx.sort(rel_error_k, axis=-1)
        sorted_v = mx.sort(rel_error_v, axis=-1)
        med_k = sorted_k[:, S_comp // 2]
        med_v = sorted_v[:, S_comp // 2]
        max_med_err = mx.maximum(med_k, med_v).tolist()

        n_res_batch = []
        for val in max_med_err:
            val = float(val)
            b_res = self.max_residual
            if val < 0.05:
                b_res = min(8, b_res)
            elif val < 0.15:
                b_res = min(16, b_res)
            n_res_batch.append(b_res)

        # Budget floor for boosted rows: a fact block's boosted set (value +
        # owner + window glue, ~11-15 rows) exceeds the easy-block cap of 8 —
        # without this floor the adaptive budget silently evicts the owner the
        # capture just fought for. Coverage-aware: MLX's coverage quota is
        # n_cov = cov_frac*max_residual columns with a +1e12 bonus (NOTE: the
        # native port derives its quota from the BLOCK budget instead — smaller;
        # align before enabling coverage by default here), and those columns
        # outrank every boosted row, so the ranked rows must fit in what's left.
        # Floor = boosted + n_cov + margin (DIFFKV_RESIDUAL_FLOOR_MARGIN, def 4).
        res_v_only, cov_frac = _capture_policy_env()
        if boost_rows:
            try:
                _floor_margin = int(os.environ.get("DIFFKV_RESIDUAL_FLOOR_MARGIN", "4"))
            except ValueError:
                _floor_margin = 4
            _n_cov = 0
            if cov_frac > 0.0 and self.max_residual > 0:
                _n_cov = min(self.max_residual, max(1, int(round(cov_frac * self.max_residual))))
            floors = [
                min(self.max_residual, sum(1 for m in row if m > 1.0) + _n_cov + _floor_margin)
                if any(m > 1.0 for m in row) else 0
                for row in boost_rows
            ]
            if any(floors):
                nb_ = len(boost_rows)
                n_res_batch = [max(n_res_batch[i], floors[i % nb_])
                               for i in range(len(n_res_batch))]
        if self.max_residual > 0:
            capture_scores = joint_errors
            cov_bonus = _coverage_bonus(S_comp, self.max_residual, cov_frac)
            if cov_bonus is not None:
                capture_scores = capture_scores.astype(mx.float32) + cov_bonus
            top_k = mx.argsort(capture_scores, axis=-1)[:, -self.max_residual:][:, ::-1]
            if cov_bonus is not None:
                # Ranked (distinctive) rows FIRST, coverage scaffold APPENDED
                # within a budget-based quota. The residual arrays double as the
                # block's ROUTING signature (decode relevance reads the first
                # route_residuals rows) — coverage-first ordering blinded the
                # router (native multi-needle 3/3→0/3). Mirrors lowrank.cpp.
                tk = np.asarray(top_k)
                cov_set = np.zeros(S_comp, dtype=bool)
                cov_set[np.asarray(cov_bonus) > 0] = True
                out_rows = np.empty_like(tk)
                for r in range(tk.shape[0]):
                    row = tk[r]
                    m = cov_set[row]
                    ranked, covs = row[~m], row[m]
                    n_res_r = int(n_res_batch[r])
                    n_cov_r = 0
                    if covs.size and n_res_r > 0 and cov_frac > 0.0:
                        n_cov_r = int(min(covs.size, max(1, round(cov_frac * n_res_r))))
                    n_rank_r = max(0, n_res_r - n_cov_r)
                    out_rows[r] = np.concatenate([
                        ranked[:n_rank_r], covs[:n_cov_r],
                        ranked[n_rank_r:], covs[n_cov_r:],
                    ])[:tk.shape[1]]
                top_k = mx.array(out_rows)
            if os.environ.get("DIFFKV_DBG_TABLE") == "1" and getattr(self, "_dbg_table_blocks", None):
                tk_dbg = np.asarray(top_k)
                for _bi in range(num_blocks):
                    _g = start_blocks + _bi
                    _marked = self._dbg_table_blocks.get(_g)
                    if _marked is None:
                        continue
                    _midx = {i for i, m in enumerate(_marked) if m}
                    _hits = []
                    for _ly in range(self.num_layers):
                        _b = _ly * num_blocks + _bi
                        _nr = int(n_res_batch[_b])
                        _hits.append(len(_midx & set(tk_dbg[_b, :_nr].tolist())))
                    _nb = sum(1 for m in boost_rows[_bi] if m > 1.0) if boost_rows else -1
                    _hs = sorted(_hits)
                    print(f"[DBG_TABLE] block {_g}: marked={len(_midx)} boosted={_nb} "
                          f"n_res_l0={int(n_res_batch[_bi])} kept_marked "
                          f"min/med/max={_hs[0]}/{_hs[len(_hs)//2]}/{_hs[-1]}",
                          flush=True)
            indices = mx.expand_dims(mx.expand_dims(top_k + 1, -1), -1)
            batch_blocks_k_t = batch_blocks_k.transpose(0, 2, 1, 3)
            batch_blocks_v_t = batch_blocks_v.transpose(0, 2, 1, 3)
            res_k_padded = mx.take_along_axis(batch_blocks_k_t, indices, axis=1)
            res_v_padded = mx.take_along_axis(batch_blocks_v_t, indices, axis=1)
            if res_v_only:
                # Store SVD-reconstructed K for the captured rows (V stays
                # exact): tests that the router/attention tolerate ~1%-error
                # residual keys — the precondition for dropping res_k storage.
                recon_k_rows = (recon_delta_k.reshape(B_batch, S_comp, self.kv_heads, self.head_dim)
                                + mx.expand_dims(batch_anchors_k, 1))
                idx_recon = mx.expand_dims(mx.expand_dims(top_k, -1), -1)
                res_k_padded = mx.take_along_axis(recon_k_rows, idx_recon, axis=1).astype(res_v_padded.dtype)

            # Build active mask to zero out inactive slots and keep static self.max_residual shape intact
            active_mask_np = np.zeros((B_batch, self.max_residual), dtype=np.bool_)
            for b in range(B_batch):
                active_mask_np[b, :n_res_batch[b]] = True
            active_mask = mx.array(active_mask_np)

            # Zero out capped slots
            res_k_padded = res_k_padded * mx.expand_dims(mx.expand_dims(active_mask, -1), -1)
            res_v_padded = res_v_padded * mx.expand_dims(mx.expand_dims(active_mask, -1), -1)

            # Construct res mask using only active indices
            match = (top_k[:, :, None] == mx.arange(S_comp)[None, None, :])
            match = match & mx.expand_dims(active_mask, -1)
            res_mask = mx.any(match, axis=1)
        else:
            top_k = None
            res_k_padded = mx.zeros((B_batch, self.max_residual, self.kv_heads, self.head_dim), dtype=batch_blocks_k.dtype)
            res_v_padded = mx.zeros((B_batch, self.max_residual, self.kv_heads, self.head_dim), dtype=batch_blocks_v.dtype)
            res_mask = mx.zeros((B_batch, S_comp), dtype=mx.bool_)
            
        # 7. Scatter back to session layers
        for l in range(self.num_layers):
            start_idx = session["num_blocks"][l]
            l_slice = slice(l * num_blocks, (l + 1) * num_blocks)
            
            session["comp_U"][l][start_idx:start_idx+num_blocks] = U_batch[l_slice]
            session["comp_VK"][l][start_idx:start_idx+num_blocks] = VK_batch[l_slice]
            session["comp_VV"][l][start_idx:start_idx+num_blocks] = VV_batch[l_slice]
            session["comp_anc_k"][l][start_idx:start_idx+num_blocks] = batch_anchors_k[l_slice]
            session["comp_anc_v"][l][start_idx:start_idx+num_blocks] = batch_anchors_v[l_slice]
            session["comp_min_k"][l][start_idx:start_idx+num_blocks] = mx.min(batch_blocks_k[l_slice], axis=2)
            session["comp_max_k"][l][start_idx:start_idx+num_blocks] = mx.max(batch_blocks_k[l_slice], axis=2)
            session["comp_scale"][l][start_idx:start_idx+num_blocks] = scales_batch[l_slice]
            session["comp_seq_len"][l][start_idx:start_idx+num_blocks] = self.block_size - 1  # S_comp: number of delta rows (excludes anchor)
            
            session["comp_res_k"][l][start_idx:start_idx+num_blocks] = res_k_padded[l_slice]
            session["comp_res_v"][l][start_idx:start_idx+num_blocks] = res_v_padded[l_slice]
            for b_idx in range(num_blocks):
                session["comp_res_n"][l][start_idx + b_idx] = n_res_batch[l * num_blocks + b_idx]
            if "comp_res_mask" in session:
                session["comp_res_mask"][l][start_idx:start_idx+num_blocks] = res_mask[l_slice]
                
            session["num_blocks"][l] = start_idx + num_blocks
            
            # Copy remaining dense tokens
            K_all = K_all_layers[l]
            V_all = V_all_layers[l]
            K_dense = K_all[:, :, N_comp:, :]
            V_dense = V_all[:, :, N_comp:, :]
            L_dense = L - N_comp
            
            session["dense_keys"][l][0, :, :L_dense] = K_dense.squeeze(0)
            session["dense_values"][l][0, :, :L_dense] = V_dense.squeeze(0)
            session["dense_lens"][l] = L_dense
            session["dense_lens_mx"][l] = mx.array(L_dense, dtype=mx.int32)
            
            rc = session.get("_res_cache")
            if rc is not None:
                rc.pop(l, None)
                
        # 8. Parallel evaluate all targets
        eval_targets = []
        for l in range(self.num_layers):
            eval_targets.extend([
                session["comp_U"][l],
                session["comp_VK"][l],
                session["comp_VV"][l],
                session["comp_anc_k"][l],
                session["comp_anc_v"][l],
                session["comp_min_k"][l],
                session["comp_max_k"][l],
                session["comp_res_k"][l],
                session["comp_res_v"][l],
                session["dense_keys"][l],
                session["dense_values"][l],
            ])
            if "comp_res_mask" in session:
                eval_targets.append(session["comp_res_mask"][l])
        mx.eval(*eval_targets)

    def _get_or_create_session(self, session_id: str):
        """dict.get + create-on-miss. NOT sessions.setdefault(_create_empty_session()):
        setdefault evaluates its default EAGERLY, so the old form built a complete
        empty session (hundreds of mx.zeros tensors) on EVERY call — 28×/decode-token
        of pure allocation churn (~6.5 ms/token host-side, measured 2026-07-10)."""
        session = self.sessions.get(session_id)
        if session is None:
            # No prefill-length hint (init_session wasn't called): start SMALL and rely
            # on _ensure_block_capacity growth. The old fallback allocated the full
            # max_blocks=256 pool (~46 MB/layer with max_residual=128 — >1.3 GB on a
            # 28-layer model) even for a 1-token session.
            init_blocks = min(self.max_blocks,
                              max(1, int(os.environ.get("DIFFKV_INIT_BLOCKS", "16"))))
            session = self.sessions[session_id] = self._create_empty_session(init_blocks)
        return session

    def capture_prefill_kv(self, session_id: str, layer_idx: int, K: mx.array, V: mx.array):
        """Write incoming prefill KV chunk into the stashed lists for deferred compression."""
        session = self._get_or_create_session(session_id)
        session["prefill_K_chunks"][layer_idx].append(K)
        session["prefill_V_chunks"][layer_idx].append(V)
        if self._lego_prefill:
            self._lego_capture_stream(session, layer_idx, K, V)

    def _lego_capture_stream(self, session: Dict, layer_idx: int, K: mx.array, V: mx.array):
        """Maintain the lego raw state per layer as prefill chunks stream in:

        * SINKS — a raw copy of the first sink_blocks*block_size tokens. The flush
          compresses block 0 like every other block, but StreamingLLM sinks must
          stay exact (attended by every chunk).
        * RING — a rolling raw copy of the LAST ~_lego_ring tokens, trimmed from
          the front in block_size multiples so its start stays block-aligned
          (far blocks = whole blocks below ring_start). This is the chunk's exact
          local neighborhood; everything older is attended via lego pieces.

        Chunks arrive in position order within a session lifetime (clear_session
        wipes this state), so appending until full / rolling forward is correct."""
        lens = session.get("lego_sink_len")
        if lens is None:
            # Record where this capture stream begins. A cached-prefix
            # continuation re-initialises lego state mid-sequence (the decode
            # boundary freed it); its first rows are NOT position 0, so they must
            # never be treated as sinks — leaving the sinks unfilled keeps
            # _lego_session_ready() False and the continuation on the raw path
            # (matching existing continuation behavior). ring_start likewise
            # starts at the stream's absolute position so block alignment holds.
            first_abs = 0
            if self.position_ids is not None:
                try:
                    first_abs = int(self.position_ids[0, 0])
                except Exception:
                    first_abs = 0
            session["lego_stream_start"] = first_abs
            lens = session["lego_sink_len"] = [0] * self.num_layers
            session["lego_sink_k"] = [None] * self.num_layers
            session["lego_sink_v"] = [None] * self.num_layers
            session["lego_ring_k"] = [None] * self.num_layers
            session["lego_ring_v"] = [None] * self.num_layers
            session["lego_ring_start"] = [first_abs] * self.num_layers
        L = int(K.shape[2])
        # 1. Sinks (only for streams that begin at position 0)
        target = self._sp_sink_blocks * self.block_size
        cur = lens[layer_idx]
        if cur < target and session.get("lego_stream_start", 0) == 0:
            take = min(target - cur, L)
            k_slice = K[0, :, :take, :]   # [H_kv, take, D]
            v_slice = V[0, :, :take, :]
            if session["lego_sink_k"][layer_idx] is None:
                session["lego_sink_k"][layer_idx] = k_slice
                session["lego_sink_v"][layer_idx] = v_slice
            else:
                session["lego_sink_k"][layer_idx] = mx.concatenate(
                    [session["lego_sink_k"][layer_idx], k_slice], axis=1)
                session["lego_sink_v"][layer_idx] = mx.concatenate(
                    [session["lego_sink_v"][layer_idx], v_slice], axis=1)
            lens[layer_idx] = cur + take
        # 2. Ring: preallocated [H_kv, 2*ring, D] buffer, appended IN PLACE per
        # chunk (mx.array __setitem__ — the same proven pattern as the dense
        # window and the fused decode buffer). The logical ring is
        # buf[:, front:n]; trimming just advances `front` (no copy), and only
        # when the write head reaches capacity is the live span copied back to
        # the buffer front — one ring-sized copy per ring-worth of tokens,
        # instead of the initial per-chunk full concat (~8x less traffic at
        # chunk 512 / ring 4096, which showed up as +8% on 16k prefill).
        bufs = session.get("lego_ring_bufs")
        if bufs is None:
            bufs = session["lego_ring_bufs"] = {}
        st = bufs.get(layer_idx)
        if st is None:
            # ring + 4 chunks of slack: compaction every ~3 chunks. (Was 2*ring —
            # ~74 MB more resident across 28 layers at ring 4096 for no benefit:
            # MLX __setitem__ is functional, the RHS slice reads the pre-update
            # array, so the self-copy below is safe even when src/dst overlap.)
            cap = self._lego_ring + 4 * L
            st = bufs[layer_idx] = {
                "k": mx.zeros((self.kv_heads, cap, self.head_dim), dtype=K.dtype),
                "v": mx.zeros((self.kv_heads, cap, self.head_dim), dtype=V.dtype),
                "front": 0, "n": 0, "cap": cap,
            }
        if st["n"] + L > st["cap"]:
            span = st["n"] - st["front"]
            if span + L > st["cap"]:
                # A later caller raised the chunk size past what the buffer was
                # sized for — reallocate (rare; buffers are sized off chunk 0).
                new_cap = self._lego_ring + 4 * L
                nk = mx.zeros((self.kv_heads, new_cap, self.head_dim), dtype=K.dtype)
                nv = mx.zeros((self.kv_heads, new_cap, self.head_dim), dtype=V.dtype)
                nk[:, :span, :] = st["k"][:, st["front"]:st["n"], :]
                nv[:, :span, :] = st["v"][:, st["front"]:st["n"], :]
                st["k"], st["v"], st["cap"] = nk, nv, new_cap
            else:
                # Compact: move the live span to the buffer front (functional
                # __setitem__ — overlap-safe, see the sizing note above).
                st["k"][:, :span, :] = st["k"][:, st["front"]:st["n"], :]
                st["v"][:, :span, :] = st["v"][:, st["front"]:st["n"], :]
            st["front"], st["n"] = 0, span
        st["k"][:, st["n"]:st["n"] + L, :] = K[0]
        st["v"][:, st["n"]:st["n"] + L, :] = V[0]
        st["n"] += L
        overflow = (st["n"] - st["front"]) - self._lego_ring
        if overflow > 0:
            trim = min(((overflow + self.block_size - 1) // self.block_size) * self.block_size,
                       st["n"] - st["front"])
            st["front"] += trim
            session["lego_ring_start"][layer_idx] += trim
        session["lego_ring_k"][layer_idx] = st["k"][:, st["front"]:st["n"], :]
        session["lego_ring_v"][layer_idx] = st["v"][:, st["front"]:st["n"], :]

    def _lego_session_ready(self, session: Dict, layer_idx: int, cur_start: int) -> bool:
        """Whether this chunk should take the lego prefill path (see the flag note
        in __init__). Sticky once engaged — after the raw prompt cache is dropped
        there is no falling back to a raw-history path."""
        if session.get("lego_engaged"):
            return True
        if cur_start < self._lego_min_ctx:
            return False
        lens = session.get("lego_sink_len")
        if lens is None or lens[layer_idx] < self._sp_sink_blocks * self.block_size:
            return False
        # Engage only when at least one whole COMPRESSED block lies below the ring
        # (the far field lego actually covers); before that, raw attention is both
        # exact and no more expensive.
        far_nb = session["lego_ring_start"][layer_idx] // self.block_size
        if far_nb <= self._sp_sink_blocks:
            return False
        if session["num_blocks"][layer_idx] < far_nb:
            return False
        ok = session.get("_lego_ok")
        if ok is None:
            # Engage only when the FULL prompt (registered up front by
            # register_prefill_tokens) will decode compressed — dense decode
            # reads the raw prompt cache that lego stops maintaining.
            total = len(session.get("token_ids") or [])
            ok = total > 0 and _resolve_compressed_decode(total)
            session["_lego_ok"] = ok
        return bool(ok)

    def compress_prefill_kv(self, session_id: str):
        pass

    def ingest_streaming(self, session_id: str, layer_idx: int, k: mx.array, v: mx.array):
        session = self._get_or_create_session(session_id)
        dense_len = session["dense_lens"][layer_idx]
        
        session["dense_keys"][layer_idx][0, :, dense_len:dense_len + 1] = k.squeeze(0)
        session["dense_values"][layer_idx][0, :, dense_len:dense_len + 1] = v.squeeze(0)
        session["dense_lens"][layer_idx] += 1
        session["dense_lens_mx"][layer_idx] = mx.array(session["dense_lens"][layer_idx], dtype=mx.int32)
        
        self._compress_eligible_blocks(session_id, layer_idx)
        # mx.eval(session["dense_keys"][layer_idx], session["dense_values"][layer_idx])

    def _compress_block(self, session: Dict, layer_idx: int, start: int):
        """Compress a single block starting at `start` in the dense buffer
        and store its low-rank representation in the compressed arrays."""
        num_blocks = session["num_blocks"][layer_idx]
        max_b = session.get("max_blocks", self.max_blocks)
        if num_blocks >= max_b:
            # Try growing the pools first (hint miss); only evict at the global cap.
            max_b = self._ensure_block_capacity(session, num_blocks + 1)
        if num_blocks >= max_b:
            # Safety: drop oldest compressed block by shifting (rare)
            session["comp_U"][layer_idx][:-1]     = session["comp_U"][layer_idx][1:]
            session["comp_VK"][layer_idx][:-1]    = session["comp_VK"][layer_idx][1:]
            session["comp_VV"][layer_idx][:-1]    = session["comp_VV"][layer_idx][1:]
            session["comp_anc_k"][layer_idx][:-1] = session["comp_anc_k"][layer_idx][1:]
            session["comp_anc_v"][layer_idx][:-1] = session["comp_anc_v"][layer_idx][1:]
            session["comp_min_k"][layer_idx][:-1] = session["comp_min_k"][layer_idx][1:]
            session["comp_max_k"][layer_idx][:-1] = session["comp_max_k"][layer_idx][1:]
            session["comp_res_k"][layer_idx][:-1] = session["comp_res_k"][layer_idx][1:]
            session["comp_res_v"][layer_idx][:-1] = session["comp_res_v"][layer_idx][1:]
            session["comp_res_n"][layer_idx][:-1] = session["comp_res_n"][layer_idx][1:]
            if "comp_res_mask" in session:
                session["comp_res_mask"][layer_idx][:-1] = session["comp_res_mask"][layer_idx][1:]
            num_blocks = max_b - 1

        block_k = session["dense_keys"][layer_idx][0, :, start:start + self.block_size]
        block_v = session["dense_values"][layer_idx][0, :, start:start + self.block_size]

        anchor_k = block_k[:, 0, :]
        anchor_v = block_v[:, 0, :]

        deltas_k = block_k[:, 1:, :] - mx.expand_dims(anchor_k, 1)
        deltas_v = block_v[:, 1:, :] - mx.expand_dims(anchor_v, 1)

        S_comp = self.block_size - 1
        deltas_k_2d = deltas_k.transpose(1, 0, 2).reshape(S_comp, -1)
        deltas_v_2d = deltas_v.transpose(1, 0, 2).reshape(S_comp, -1)

        # V-side rebalancing for the joint K|V SVD.
        # Keep v_gain computation entirely lazy in the MLX graph — one .item() sync
        # at the end instead of the prior eK.item() + eV.item() (two syncs).
        v_scale_on = os.environ.get("DIFFKV_V_SCALE", "1") != "0"
        v_gain = 1.0
        if v_scale_on:
            eK = mx.sum(deltas_k_2d.astype(mx.float32)**2)
            eV = mx.sum(deltas_v_2d.astype(mx.float32)**2)
            both_ok  = mx.logical_and(eV > 1e-12, eK > 1e-12)
            gain_raw = mx.sqrt(eK / mx.maximum(eV, mx.array(1e-12)))
            gain_clamped = mx.minimum(mx.maximum(gain_raw, mx.array(1.0)), mx.array(10000.0))
            v_gain = float(mx.where(both_ok, gain_clamped, mx.array(1.0)).item())  # 1 sync

            if v_gain > 1.0:
                deltas_v_2d_scaled = deltas_v_2d * v_gain
                deltas_2d = mx.concatenate([deltas_k_2d, deltas_v_2d_scaled], axis=1)
            else:
                deltas_2d = mx.concatenate([deltas_k_2d, deltas_v_2d], axis=1)
        else:
            deltas_2d = mx.concatenate([deltas_k_2d, deltas_v_2d], axis=1)

        token_norms = mx.linalg.norm(deltas_2d, axis=-1, keepdims=True)
        token_norms = mx.maximum(token_norms, 1e-5)
        deltas_normalized = deltas_2d / token_norms

        U_k, Vh_k, svd_scale, k_rank = compress_mlx_block(deltas_normalized, self.rank)
        # mx.eval(token_norms) removed: U_k is now an MLX tensor (not NumPy-backed),
        # so there is no graph boundary here. The multiply stays fully lazy.
        U_k = U_k * token_norms

        U_padded  = mx.pad(U_k,  [(0, 0), (0, self.rank - k_rank)])
        Vh_padded = mx.pad(Vh_k, [(0, self.rank - k_rank), (0, 0)])

        VK_flat = Vh_padded[:, :self.kv_heads * self.head_dim]
        VV_flat = Vh_padded[:, self.kv_heads * self.head_dim:]
        if v_scale_on and v_gain > 1.0:
            # The SVD ran on v_gain-scaled V; baking 1/v_gain here makes decode's
            # reconstruction produce raw-space V with no kernel changes.
            VV_flat = VV_flat / v_gain

        VK = VK_flat.reshape(self.rank, self.kv_heads, self.head_dim).transpose(1, 0, 2)
        VV = VV_flat.reshape(self.rank, self.kv_heads, self.head_dim).transpose(1, 0, 2)

        # ── SVD Residual Correction ──────────────────────────────────────────
        # Reconstruct block deltas from low-rank SVD components entirely in MLX on GPU
        recon_delta = mx.matmul(U_padded, Vh_padded) * svd_scale
        recon_delta_k = recon_delta[:, :self.kv_heads * self.head_dim]
        recon_delta_v = recon_delta[:, self.kv_heads * self.head_dim:]
        if v_scale_on and v_gain > 1.0:
            # Unscale V reconstruction to raw space for error computation
            recon_delta_v = recon_delta_v / v_gain

        errors_k = mx.linalg.norm(deltas_k_2d - recon_delta_k, axis=-1)
        errors_v = mx.linalg.norm(deltas_v_2d - recon_delta_v, axis=-1)

        # Rank residual candidates in the BALANCED space: weight V error by v_gain
        errors_v_balanced = errors_v * v_gain
        joint_errors = mx.sqrt(errors_k**2 + errors_v_balanced**2)

        # Content-aware residual capture / token boosting
        tok_boost_env = os.environ.get("DIFFKV_RESIDUAL_TOKEN_BOOST")
        tok_boost = 8.0
        if tok_boost_env is not None:
            try:
                tok_boost = float(tok_boost_env)
            except ValueError:
                pass

        n_boosted_rows = 0
        if os.environ.get("DIFFKV_DBG_TABLE") == "1":
            print(f"[DBG_TABLE/gate-stream] tok_boost={tok_boost} "
                  f"has_token_ids={'token_ids' in session} "
                  f"len={len(session.get('token_ids', []))} "
                  f"num_blocks={num_blocks} S_comp={S_comp}", flush=True)
        if tok_boost > 1.0 and "token_ids" in session and len(session["token_ids"]) > 0:
            abs_start = num_blocks * self.block_size
            tids = session["token_ids"][abs_start + 1 : abs_start + self.block_size]
            if len(tids) == S_comp:
                counts = session.get("token_counts", {})
                total_tokens = len(session["token_ids"])
                
                # Decode each token to string
                tok_strs = [self.tokenizer.decode([tid]) for tid in tids]
                
                # Classify core information tokens (digits, uppercase words/identifiers)
                is_core = []
                for s in tok_strs:
                    s_clean = s.strip()
                    has_digit = any(c.isdigit() for c in s_clean)
                    is_upper = s_clean.isupper() and s_clean.isalpha() and len(s_clean) >= 2
                    is_core.append(has_digit or is_upper or s_clean == '-' or s_clean == '_')
                
                # Classify prose tokens
                is_prose = []
                for s in tok_strs:
                    s_clean = s.strip()
                    if not s_clean:
                        is_prose.append(True)
                        continue
                    if s_clean in ('.', ',', ';', '?', '!', ':', '"', "'", '(', ')', '[', ']', '{', '}'):
                        is_prose.append(True)
                        continue
                    if s_clean.isalpha():
                        if s_clean.islower() or (s_clean.istitle() and len(s_clean) > 1):
                            is_prose.append(True)
                            continue
                    is_prose.append(False)
                
                boost_multipliers = [1.0] * S_comp
                in_segment = False
                segment_indices = []
                
                for i in range(S_comp):
                    if not is_prose[i]:
                        if not in_segment:
                            in_segment = True
                            segment_indices.append([i])
                        else:
                            segment_indices[-1].append(i)
                    else:
                        in_segment = False
                
                boosted_segs = []
                for seg in segment_indices:
                    contains_core = any(is_core[i] for i in seg)
                    if contains_core:
                        boosted_segs.append([tok_strs[i] for i in seg])
                        for i in seg:
                            tid = tids[i]
                            count = counts.get(tid, 1)
                            import math
                            idf = math.log(max(total_tokens, 2) / (count + 0.1))
                            rarity_weight = max(1.0, min(idf, 6.0))
                            boost_multipliers[i] = tok_boost * (rarity_weight / 2.0)

                # Owner capture: the fact's entity name gets the same
                # exact-residual treatment as its value (see helper doc).
                _apply_owner_capture(boost_multipliers, segment_indices, is_core,
                                     tok_strs, tids, counts, total_tokens, tok_boost)

                # Table capture: whole table-like lines (incl. header/unit
                # cells) enter the exact set with priority (see helper doc).
                _n_tab_stream = _apply_table_capture(boost_multipliers, tok_strs, tids,
                                                     counts, total_tokens, tok_boost)
                if _n_tab_stream and os.environ.get("DIFFKV_DBG_TABLE") == "1":
                    print(f"[DBG_TABLE/stream] block {num_blocks}: marked={_n_tab_stream}",
                          flush=True)

                # Apply window boost (Phase 2: contiguous runs)
                final_boosts = list(boost_multipliers)
                W = 2
                for idx in range(S_comp):
                    if boost_multipliers[idx] > 1.0:
                        for j in range(max(0, idx - W), min(S_comp, idx + W + 1)):
                            final_boosts[j] = max(final_boosts[j], boost_multipliers[idx])
                boost_multipliers = final_boosts

                # Relational edge capture: pin the connectives that BIND the
                # captured nodes into the exact set (see helper doc).
                _apply_relational_capture(boost_multipliers, tok_strs, tids,
                                          counts, total_tokens, tok_boost)
                n_boosted_rows = sum(1 for m in boost_multipliers if m > 1.0)

                boost_arr = mx.array(boost_multipliers, dtype=joint_errors.dtype)
                joint_errors = joint_errors * boost_arr

        # OPT-A: Adaptive residual budget — MLX/Python wrapper port.
        norms_k = mx.linalg.norm(deltas_k_2d, axis=-1)
        norms_v = mx.linalg.norm(deltas_v_2d, axis=-1)
        rel_error_k = errors_k / mx.maximum(norms_k, 1e-8)
        rel_error_v = errors_v / mx.maximum(norms_v, 1e-8)

        # MLX lacks mx.median; sort and retrieve the middle element.
        sorted_k = mx.sort(rel_error_k)
        sorted_v = mx.sort(rel_error_v)
        median_err_k = float(sorted_k[S_comp // 2].item())
        median_err_v = float(sorted_v[S_comp // 2].item())
        max_median_err = max(median_err_k, median_err_v)

        n_res = self.max_residual
        if max_median_err < 0.05:
            # Easy block (prose filler): cap at 8 residuals
            n_res = min(8, n_res)
        elif max_median_err < 0.15:
            # Medium block: cap at 16 residuals
            n_res = min(16, n_res)
        # Budget floor for boosted rows (value + owner + window glue must all
        # fit — coverage-aware, see the prefill-path comment).
        res_v_only, cov_frac = _capture_policy_env()
        if n_boosted_rows:
            try:
                _floor_margin = int(os.environ.get("DIFFKV_RESIDUAL_FLOOR_MARGIN", "4"))
            except ValueError:
                _floor_margin = 4
            _n_cov = 0
            if cov_frac > 0.0 and self.max_residual > 0:
                _n_cov = min(self.max_residual, max(1, int(round(cov_frac * self.max_residual))))
            n_res = max(n_res, min(self.max_residual, n_boosted_rows + _n_cov + _floor_margin))
        if self.max_residual > 0:
            capture_scores = joint_errors
            cov_bonus = _coverage_bonus(S_comp, self.max_residual, cov_frac)
            if cov_bonus is not None:
                capture_scores = capture_scores.astype(mx.float32) + cov_bonus
            # Select the top n_res elements
            top_k_indices = mx.argsort(capture_scores)[-n_res:][::-1]
            if cov_bonus is not None:
                # Ranked rows FIRST, coverage appended (budget-based quota) —
                # the residual head is the block's routing signature; see the
                # prefill-path comment / lowrank.cpp coverage note.
                row = np.asarray(top_k_indices)
                cov_set = np.zeros(S_comp, dtype=bool)
                cov_set[np.asarray(cov_bonus) > 0] = True
                m = cov_set[row]
                ranked, covs = row[~m], row[m]
                n_cov_r = 0
                if covs.size and n_res > 0 and cov_frac > 0.0:
                    n_cov_r = int(min(covs.size, max(1, round(cov_frac * n_res))))
                n_rank_r = max(0, n_res - n_cov_r)
                row = np.concatenate([ranked[:n_rank_r], covs[:n_cov_r],
                                      ranked[n_rank_r:], covs[n_cov_r:]])[:n_res]
                top_k_indices = mx.array(row)
            top_k = top_k_indices  # compatibility alias for mask writing below

            block_k_t = block_k.transpose(1, 0, 2)
            block_v_t = block_v.transpose(1, 0, 2)
            res_k_active = mx.take(block_k_t, top_k_indices + 1, axis=0)
            res_v_active = mx.take(block_v_t, top_k_indices + 1, axis=0)
            if res_v_only:
                recon_k_rows = (recon_delta_k.reshape(S_comp, self.kv_heads, self.head_dim)
                                + mx.expand_dims(anchor_k, 0))
                res_k_active = mx.take(recon_k_rows, top_k_indices, axis=0).astype(res_v_active.dtype)

            # Pad active residuals with zeros to keep static self.max_residual shape intact
            pad_len = self.max_residual - n_res
            if pad_len > 0:
                zero_k = mx.zeros((pad_len, self.kv_heads, self.head_dim), dtype=res_k_active.dtype)
                zero_v = mx.zeros((pad_len, self.kv_heads, self.head_dim), dtype=res_v_active.dtype)
                res_k_padded = mx.concatenate([res_k_active, zero_k], axis=0)
                res_v_padded = mx.concatenate([res_v_active, zero_v], axis=0)
            else:
                res_k_padded = res_k_active
                res_v_padded = res_v_active
        else:
            top_k = None
            res_k_padded = mx.zeros((self.max_residual, self.kv_heads, self.head_dim), dtype=block_k.dtype)
            res_v_padded = mx.zeros((self.max_residual, self.kv_heads, self.head_dim), dtype=block_v.dtype)
            n_res = 0


        session["comp_U"][layer_idx][num_blocks]     = U_padded
        session["comp_VK"][layer_idx][num_blocks]    = VK
        session["comp_VV"][layer_idx][num_blocks]    = VV
        session["comp_anc_k"][layer_idx][num_blocks] = anchor_k
        session["comp_anc_v"][layer_idx][num_blocks] = anchor_v
        # Element-wise key min/max over the block's tokens — for top-K routing.
        session["comp_min_k"][layer_idx][num_blocks] = mx.min(block_k, axis=1)
        session["comp_max_k"][layer_idx][num_blocks] = mx.max(block_k, axis=1)
        session["comp_scale"][layer_idx][num_blocks]   = svd_scale
        session["comp_seq_len"][layer_idx][num_blocks] = self.block_size - 1  # S_comp: number of delta rows (excludes anchor)
        
        session["comp_res_k"][layer_idx][num_blocks] = res_k_padded
        session["comp_res_v"][layer_idx][num_blocks] = res_v_padded
        session["comp_res_n"][layer_idx][num_blocks] = n_res
        # Mark which delta positions are kept exact (top_k indexes into the S_comp
        # deltas, aligned with the kernel's delta_s axis) so decode can drop their
        # lossy SVD twin from the sparse pool.
        if "comp_res_mask" in session:
            mask_val = mx.zeros((self.block_size - 1,), dtype=mx.bool_)
            if top_k is not None:
                mask_val[top_k] = True
            session["comp_res_mask"][layer_idx][num_blocks] = mask_val
        
        session["num_blocks"][layer_idx] = num_blocks + 1
        # Invalidate the cached residual gather for this layer: the block set
        # (and, at max_blocks, the block ordering via the shift above) changed.
        rc = session.get("_res_cache")
        if rc is not None:
            rc.pop(layer_idx, None)
        # Same for the decode-cache. Normally redundant (num_blocks changed → the
        # nb check re-routes), but at the max_blocks cap the shift above changes
        # block CONTENTS while nb stays constant — without this pop the decode
        # cache would keep attending the pre-shift blocks for a full interval.
        ck = session.get("_cache_kv")
        if ck is not None:
            ck.pop(layer_idx, None)

    def _flush_oldest_block(self, session: Dict, layer_idx: int):
        """Compress the oldest block_size tokens from the dense buffer and
        shift the remaining tokens to the front."""
        dense_len = session["dense_lens"][layer_idx]
        self._compress_block(session, layer_idx, start=0)
        # Shift remaining tokens left by block_size
        remaining = dense_len - self.block_size
        if remaining > 0:
            session["dense_keys"][layer_idx][0, :, :remaining]   = session["dense_keys"][layer_idx][0, :, self.block_size:dense_len]
            session["dense_values"][layer_idx][0, :, :remaining] = session["dense_values"][layer_idx][0, :, self.block_size:dense_len]
        session["dense_keys"][layer_idx][0, :, remaining:dense_len]   = 0.0
        session["dense_values"][layer_idx][0, :, remaining:dense_len] = 0.0
        session["dense_lens"][layer_idx] = remaining
        session["dense_lens_mx"][layer_idx] = mx.array(remaining, dtype=mx.int32)
        # Materialise all pending ops immediately so the lazy graph of slice
        # shifts does not chain dependencies across blocks.
        mx.eval(
            session["dense_keys"][layer_idx],
            session["dense_values"][layer_idx],
        )
        mx.clear_cache()

    def _compress_eligible_blocks(self, session_id: str, layer_idx: int):
        """Called after each decode step — flush blocks until dense window fits."""
        session = self.sessions[session_id]
        while session["dense_lens"][layer_idx] >= self.recency_window + self.block_size:
            self._flush_oldest_block(session, layer_idx)

    def _execute_decode_cache(self, session, layer_idx, q, dense_k, dense_v, dense_len, scale, gpk):
        """Decompress-and-cache decode (see DIFFKV_DECODE_CACHE). Materialises the routed blocks'
        K/V from the low-rank pool once per interval, caches it, and attends [cached blocks +
        exact residuals + dense window] with a single masked SDPA. Returns out [H_q, D].

        FUSED per-token path (DIFFKV_DECODE_FUSED=1, default ON, 2026-07-10): at route time the
        materialised blocks AND the dense window are written once into a persistent fused buffer
        [kv_heads, Lm + max_dense_len, D] with a static additive mask. Each subsequent token then
        only (a) appends its single freshly-ingested dense row into the buffer (an in-place
        one-row slice_update) and (b) runs ONE mx.fast.scaled_dot_product_attention launch over
        an exact-length view — no per-token concatenation, no per-token mask construction.
        Verified vs the legacy path: NIAH bench 4/4 (4k-32k), depths 3/3, multi-needle 1/1,
        synthetic parity (tests/test_decode_cache_fused_parity.py).
        Numerics notes:
          - Buffer dtype defaults to fp32 (DIFFKV_DECODE_FUSED_FP32, see the dial comment
            below): identical score arithmetic to the legacy path, which also ran fp32 —
            not by design but because comp_scale (an fp32 array) silently promoted the
            whole materialised cache. fp16 storage (=0) is ~3 ms/token faster but flipped
            the multi-needle case format, so it is opt-in.
          - Mask floor is -3e4 (fp16-safe) instead of -1e9; exp(-3e4 - lse) underflows to 0
            identically, and no row is ever fully masked (the current token is always live).
        DIFFKV_DECODE_FUSED=0 restores the old concat-per-token behavior exactly."""
        nb = session["num_blocks"][layer_idx]
        kv_heads, D, bs = self.kv_heads, self.head_dim, self.block_size
        S_comp = bs - 1
        fused = self._decode_fused
        NEGf = mx.array(-1e9, dtype=mx.float32)
        ZEROf = mx.array(0.0, dtype=mx.float32)
        cache = session.setdefault("_cache_kv", {})
        ent = cache.get(layer_idx)
        # Re-route on interval, OR whenever the block count changed (a block was flushed from the
        # dense window into the pool since the last route — else its content is attended nowhere).
        need_route = (ent is None) or (ent["steps"] >= self._decode_cache_interval) or (ent.get("nb") != nb)

        if need_route:
            U  = session["comp_U"][layer_idx][:nb]
            VK = session["comp_VK"][layer_idx][:nb]
            VV = session["comp_VV"][layer_idx][:nb]
            ak = session["comp_anc_k"][layer_idx][:nb]
            av = session["comp_anc_v"][layer_idx][:nb]
            sc = session["comp_scale"][layer_idx][:nb]
            csl = session["comp_seq_len"][layer_idx][:nb]
            rk = session["comp_res_k"][layer_idx][:nb]
            rv = session["comp_res_v"][layer_idx][:nb]
            res_n = mx.array(session["comp_res_n"][layer_idx][:nb], dtype=mx.int32)
            res_mask = session["comp_res_mask"][layer_idx][:nb] if "comp_res_mask" in session else None

            k_eff = self.topk_blocks
            if self.topk_blocks > 0 and self.topk_frac > 0.0:
                k_eff = max(self.topk_blocks, int(nb * self.topk_frac))
            if self.topk_blocks > 0 and nb > k_eff:
                R_route = min(self.route_residuals, self.max_residual)
                rvld = mx.expand_dims(mx.arange(R_route), 0) < mx.expand_dims(mx.minimum(res_n, R_route), 1)
                relevance = _block_relevance_residual(q, ak, rk[:, :R_route], rvld, scale, gpk)
                sel = mx.argsort(relevance)[-k_eff:]
                U = mx.take(U, sel, 0); VK = mx.take(VK, sel, 0); VV = mx.take(VV, sel, 0)
                ak = mx.take(ak, sel, 0); av = mx.take(av, sel, 0); sc = mx.take(sc, sel, 0)
                csl = mx.take(csl, sel, 0); rk = mx.take(rk, sel, 0); rv = mx.take(rv, sel, 0)
                res_n = mx.take(res_n, sel, 0)
                if res_mask is not None:
                    res_mask = mx.take(res_mask, sel, 0)
            K, R = U.shape[0], rk.shape[1]

            # Materialise: recon[t>0] = anchor + comp_scale*(U[t] @ V_basis); position 0 = anchor.
            delta_k = mx.einsum("bsr,bhrd->bhsd", U, VK) * sc.reshape(K, 1, 1, 1)
            delta_v = mx.einsum("bsr,bhrd->bhsd", U, VV) * sc.reshape(K, 1, 1, 1)
            ak_e = mx.expand_dims(ak, 2); av_e = mx.expand_dims(av, 2)
            full_k = mx.concatenate([ak_e, ak_e + delta_k], axis=2).transpose(1, 0, 2, 3).reshape(kv_heads, K * bs, D)
            full_v = mx.concatenate([av_e, av_e + delta_v], axis=2).transpose(1, 0, 2, 3).reshape(kv_heads, K * bs, D)
            res_k_all = rk.transpose(2, 0, 1, 3).reshape(kv_heads, K * R, D)
            res_v_all = rv.transpose(2, 0, 1, 3).reshape(kv_heads, K * R, D)

            pos = mx.arange(S_comp).reshape(1, S_comp)
            recon_valid = pos < csl.reshape(K, 1)
            if res_mask is not None:
                recon_valid = recon_valid & (~res_mask)          # drop low-rank twins of residuals
            block_valid = mx.concatenate([mx.ones((K, 1), dtype=mx.bool_), recon_valid], axis=1).reshape(K * bs)
            res_valid = (mx.arange(R).reshape(1, R) < res_n.reshape(K, 1)).reshape(K * R)

            mk = mx.concatenate([full_k, res_k_all], axis=1)
            mv = mx.concatenate([full_v, res_v_all], axis=1)
            # SPARSE_BIAS in the unified SDPA = an additive score bonus on the COMPRESSED-block
            # keys only (not residuals, not the dense window) — the same up-weighting the LSE
            # merge applies. auto mode uses its base as a fixed value (the unified SDPA has no
            # separate sparse/dense LSE to adapt on). 0.0 → bit-exact to the reference.
            _bias = _SPARSE_BIAS_BASE if _SPARSE_BIAS_MODE == "auto" else _SPARSE_BIAS
            block_add = mx.where(block_valid, mx.array(_bias, dtype=mx.float32), NEGf)
            res_add   = mx.where(res_valid, ZEROf, NEGf)
            block_add_mask = mx.concatenate([block_add, res_add])
            if fused:
                # Persistent fused buffer: [materialised blocks + residuals | dense window slots].
                # Stored in the activation dtype; reconstruction above stays fp32, only the
                # STORAGE is cast. The mask's dense segment is all-zeros because per-token
                # attention slices to the EXACT live length (no padding is ever visible).
                # DIFFKV_DECODE_FUSED_FP32 — buffer dtype dial (default "1" = fp32):
                #   "1" → K/V fp32, the legacy score arithmetic. REQUIRED for the exact
                #         multi-needle gate: full-fp16 storage flipped its case format
                #         (fp16-epsilon butterfly; bisected 2026-07-10 — the structural
                #         fused path with fp32 storage matches the legacy gate exactly).
                #   "0" → K/V fp16. ~7 ms/token faster (measured 35→48 tps @8k, 36→45 @32k,
                #         clean cells, compacted pool) and passes single-needle bench 4/4 +
                #         depths 3/3, but multi-needle returned 'OMEGA-7741-Delta' (content
                #         right, case flipped). Retrieval-speed dial only.
                # (K-fp32/V-fp16 mixed was tried and REJECTED: MLX SDPA drops to a
                # fallback kernel and it's slower than full fp32.)
                _fp16_buf = os.environ.get("DIFFKV_DECODE_FUSED_FP32", "1").strip().lower() in ("0", "off", "false")
                _fdt = q.dtype if _fp16_buf else mx.float32
                # COMPACTION: drop rows whose mask is -1e9 (padding + the masked SVD twins
                # of exact residuals — with max_residual=128 that is ~30% of the pool).
                # Their softmax weight is exactly 0, so removing them is EXACT; it just
                # stops the per-token SDPA from reading dead rows. One small host sync
                # (K*bs+K*R bools) per route interval, amortized.
                valid_np = np.array(mx.concatenate([block_valid, res_valid]))
                if not valid_np.all():
                    keep = mx.array(np.nonzero(valid_np)[0].astype(np.uint32))
                    mk = mx.take(mk, keep, axis=1)
                    mv = mx.take(mv, keep, axis=1)
                    block_add_mask = mx.take(block_add_mask, keep)
                mk = mk.astype(_fdt)
                mv = mv.astype(_fdt)
                am_static = mx.maximum(block_add_mask, -3e4).astype(_fdt)
                Lm = mk.shape[1]
                fk = mx.concatenate([mk, dense_k.astype(_fdt)], axis=1)
                fv = mx.concatenate([mv, dense_v.astype(_fdt)], axis=1)
                am = mx.concatenate([am_static, mx.zeros((dense_k.shape[1],), dtype=_fdt)])
                # Eager eval is deliberate: deferring materialisation to the token's
                # end-of-step eval was A/B'd (2026-07-10) and is WORSE (mean 26.2 vs
                # 23.9 ms/tok @8k) — the mega-graph on route tokens serializes badly.
                mx.eval(fk, fv, am)                               # materialise the cache now
                ent = {"fk": fk, "fv": fv, "am": am, "Lm": Lm, "steps": 0, "nb": nb,
                       "dl_synced": session["dense_lens"][layer_idx]}
            else:
                mx.eval(mk, mv, block_add_mask)                   # materialise the cache now
                ent = {"mk": mk, "mv": mv, "mask": block_add_mask, "steps": 0, "nb": nb}
            cache[layer_idx] = ent

        ent["steps"] += 1

        if fused:
            fk, fv, am, Lm = ent["fk"], ent["fv"], ent["am"], ent["Lm"]
            dl = session["dense_lens"][layer_idx]     # Python int — exact live length
            ds = ent["dl_synced"]
            if dl > ds:
                # Copy the row(s) ingested since the last sync (normally exactly 1: this
                # token) into the fused buffer. In-place one-row update, not a concat.
                fk[:, Lm + ds:Lm + dl, :] = dense_k[:, ds:dl, :]
                fv[:, Lm + ds:Lm + dl, :] = dense_v[:, ds:dl, :]
                ent["dl_synced"] = dl
            L = Lm + dl
            out = mx.fast.scaled_dot_product_attention(
                q.reshape(1, self.heads, 1, D),
                fk[:, :L].reshape(1, kv_heads, L, D),
                fv[:, :L].reshape(1, kv_heads, L, D),
                scale=scale, mask=am[:L].reshape(1, 1, 1, L))
            return out[0, :, 0, :]

        mk, mv, block_add_mask = ent["mk"], ent["mv"], ent["mask"]

        # Append the CURRENT dense window (changes every token) + its validity mask.
        fk = mx.concatenate([mk, dense_k], axis=1)
        fv = mx.concatenate([mv, dense_v], axis=1)
        dense_add = mx.where(mx.arange(dense_k.shape[1]) < dense_len, ZEROf, NEGf)
        add_mask = mx.concatenate([block_add_mask, dense_add])
        L = fk.shape[1]
        out = mx.fast.scaled_dot_product_attention(
            q.reshape(1, self.heads, 1, D), fk.reshape(1, kv_heads, L, D), fv.reshape(1, kv_heads, L, D),
            scale=scale, mask=add_mask.reshape(1, 1, 1, L))
        return out[0, :, 0, :]

    def execute_decode_attention(self, session_id: str, layer_idx: int, q_rot: mx.array, rope: Any, scale: float, num_key_value_groups: int) -> mx.array:
        session = self.sessions[session_id]

        q   = q_rot.squeeze(2).squeeze(0)   # [H_q, D]
        gpk = num_key_value_groups
        nb  = session["num_blocks"][layer_idx]  # Python int — used for slicing

        if self._decode_cache and nb > 0:
            dk = session["dense_keys"][layer_idx][0]
            dv = session["dense_values"][layer_idx][0]
            dl = session["dense_lens_mx"][layer_idx]
            out = self._execute_decode_cache(session, layer_idx, q, dk, dv, dl, scale, gpk)
            return mx.expand_dims(mx.expand_dims(out, 0), 2)

        dense_k   = session["dense_keys"][layer_idx][0]    # [kv_heads, max_dense, D]
        dense_v   = session["dense_values"][layer_idx][0]
        dense_len = session["dense_lens_mx"][layer_idx]

        # ── Top-K block routing ───────────────────────────────────────────────
        # When enabled and there are more compressed blocks than K, score every
        # block cheaply (Quest key min/max bound, no value reconstruction) and keep
        # only the K most relevant. Decode then runs the expensive value
        # reconstruction + exact-residual attention for K blocks instead of all nb,
        # so cost scales with K rather than total context. `topk_sel` (an mx.array of
        # selected block ids, kept lazy) carries the selection to the residual gather
        # below; None = attend all blocks.
        k_eff = self.topk_blocks
        if self.topk_blocks > 0 and self.topk_frac > 0.0:
            k_eff = max(self.topk_blocks, int(nb * self.topk_frac))
        # High-Quality Mode forces attend-all (no top-K prune) — the direct analog
        # of native's DIFFKV_HIGH_QUALITY_ROUTING attend-all path.
        use_topk = (self.topk_blocks > 0 and nb > k_eff) and not self._high_quality_routing

        # Host-cheap uniformity check (Python list, no GPU sync): blocks are only
        # ever compressed at exactly block_size, so n_res ≡ max_residual for all of
        # them. When uniform, the top-K residual gather is a fixed-width mx.take with
        # no host sync; the variable-length loop only runs in the rare non-uniform case.
        _res_n_list = session["comp_res_n"][layer_idx]
        all_blocks_full = (self.max_residual > 0 and nb > 0
                           and min(_res_n_list[:nb]) == self.max_residual)
        route_once = os.environ.get("DIFFKV_ROUTE_ONCE", "0") == "1" or getattr(self, "route_once", False)
        if layer_idx == 0:
            session["_route_once_sel"] = None

        if all_blocks_full:
            # Pad nb to the next power of 2 to stabilize compile shapes and avoid re-compilations
            nb_padded = 1 << (nb - 1).bit_length() if nb > 1 else 1
            nb_padded = min(nb_padded, session.get("max_blocks", self.max_blocks),
                           self._comp_res_n_const.shape[0])  # guard: never exceed pre-allocated const array

            comp_res_n_arr = self._comp_res_n_const[:nb_padded]
            S_comp = self.block_size - 1
            if self._res_exclude_svd and "comp_res_mask" in session:
                res_mask = session["comp_res_mask"][layer_idx][:nb_padded]
            else:
                res_mask = mx.zeros((nb_padded, S_comp), dtype=mx.bool_)
            
            cached_sel = session.get("_route_once_sel")
            use_cached_sel = route_once and cached_sel is not None
            if cached_sel is None:
                cached_sel = mx.zeros((1,), dtype=mx.int32)
                
            nb_actual_arr = mx.array([nb], dtype=mx.int32)
            out_combined, sel, lse_sparse, lse_dense, scores_sparse = _execute_decode_attention_compiled(
                q, dense_k, dense_v, dense_len,
                session["comp_U"][layer_idx][:nb_padded],
                session["comp_VK"][layer_idx][:nb_padded],
                session["comp_VV"][layer_idx][:nb_padded],
                session["comp_anc_k"][layer_idx][:nb_padded],
                session["comp_anc_v"][layer_idx][:nb_padded],
                session["comp_min_k"][layer_idx][:nb_padded],
                session["comp_max_k"][layer_idx][:nb_padded],
                session["comp_scale"][layer_idx][:nb_padded],
                session["comp_seq_len"][layer_idx][:nb_padded],
                session["comp_res_k"][layer_idx][:nb_padded],
                session["comp_res_v"][layer_idx][:nb_padded],
                comp_res_n_arr,
                res_mask,
                cached_sel,
                nb_actual_arr,
                scale, gpk, self.kv_heads, self.block_size, self.rank,
                self.max_dense_len, self.max_residual, self.route_residuals,
                k_eff, self.router, use_topk, use_cached_sel
            )
            if os.environ.get("DIFFKV_DBG_LSE_SHARE") == "1":
                # sigmoid(lse_s - lse_d): numerically stable — Qwen2.5's massive
                # activations push LSE magnitudes to ~1e4, so raw exp() overflows to
                # inf/inf = nan (which silently broke the 2026-07-03 D2A measurement).
                share_comp = mx.sigmoid(lse_sparse - lse_dense)
                max_share = float(mx.max(share_comp).item())
                avg_share = float(mx.mean(share_comp).item())
                top_block = int(sel[0].item()) if (sel is not None and sel.size > 0) else -1
                print(f"[LSE_SHARE] Layer {layer_idx}: max={max_share:.4f} avg={avg_share:.4f} top_block={top_block}", flush=True)
                if layer_idx in (0, 20):
                    max_h = int(mx.argmax(share_comp).item())
                    max_sh = float(share_comp[max_h].item())
                    print(f"[LSE_SHARE_INFO] Layer {layer_idx} Max-Share Head={max_h} share={max_sh:.4f} lse_sparse={float(lse_sparse[max_h].item()):.4f} lse_dense={float(lse_dense[max_h].item()):.4f}", flush=True)
                    print(f"[LSE_SHARE_INFO] (Ledger: Residuals are in the DENSE half for MLX)", flush=True)
                    
                    h_scores = scores_sparse[max_h]
                    session_token_ids = session.get("token_ids", [])
                    sel_list = [int(x) for x in sel.tolist()]
                    all_rows = []
                    for k, block_idx in enumerate(sel_list):
                        ap = block_idx * self.block_size
                        anc_score = float(h_scores[k * self.block_size].item())
                        
                        token_str = ""
                        if ap < len(session_token_ids) and self.tokenizer is not None:
                            token_str = f" ('{self.tokenizer.decode([session_token_ids[ap]])}')"
                            
                        all_rows.append({
                            "block_id": block_idx,
                            "row": -1,
                            "abs_pos": ap,
                            "score": anc_score,
                            "token_str": token_str
                        })
                        
                        for t in range(self.block_size - 1):
                            abs_pos = ap + 1 + t
                            t_score = float(h_scores[k * self.block_size + 1 + t].item())
                            
                            token_str = ""
                            if abs_pos < len(session_token_ids) and self.tokenizer is not None:
                                token_str = f" ('{self.tokenizer.decode([session_token_ids[abs_pos]])}')"
                                
                            all_rows.append({
                                "block_id": block_idx,
                                "row": t,
                                "abs_pos": abs_pos,
                                "score": t_score,
                                "token_str": token_str
                            })
                    all_rows.sort(key=lambda x: x["score"], reverse=True)
                    for i, r in enumerate(all_rows[:5]):
                        print(f"  [LSE_SHARE_ROW] #{i}: block_id={r['block_id']} row={r['row']} abs_pos={r['abs_pos']}{r['token_str']} score={r['score']:.4f}", flush=True)
            if route_once and not use_cached_sel and use_topk:
                session["_route_once_sel"] = sel
        else:
            if nb > 0 and use_topk:
                if route_once and session.get("_route_once_sel") is not None:
                    sel = session["_route_once_sel"]
                else:
                    if self.router == "residual" and self.max_residual > 0:
                        R = min(self.route_residuals, self.max_residual)
                        res_n = mx.array(session["comp_res_n"][layer_idx][:nb], dtype=mx.int32)  # [nb]
                        res_valid = mx.arange(R).reshape(1, -1) < mx.minimum(res_n, R).reshape(-1, 1)
                        relevance = _block_relevance_residual(
                            q,
                            session["comp_anc_k"][layer_idx][:nb],
                            session["comp_res_k"][layer_idx][:nb, :R],
                            res_valid,
                            scale, gpk,
                        )
                    else:
                        relevance = _block_relevance_minmax(
                            q,
                            session["comp_min_k"][layer_idx][:nb],
                            session["comp_max_k"][layer_idx][:nb],
                            scale, gpk,
                        )
                    sel = mx.argsort(relevance)[-k_eff:]               # [k_eff] selected block ids
                    if route_once:
                        session["_route_once_sel"] = sel
                topk_sel     = sel
                comp_U       = mx.take(session["comp_U"][layer_idx][:nb],       sel, axis=0)
                comp_VK      = mx.take(session["comp_VK"][layer_idx][:nb],      sel, axis=0)
                comp_VV      = mx.take(session["comp_VV"][layer_idx][:nb],      sel, axis=0)
                comp_anc_k   = mx.take(session["comp_anc_k"][layer_idx][:nb],   sel, axis=0)
                comp_anc_v   = mx.take(session["comp_anc_v"][layer_idx][:nb],   sel, axis=0)
                comp_scale   = mx.take(session["comp_scale"][layer_idx][:nb],   sel, axis=0)
                comp_seq_len = mx.take(session["comp_seq_len"][layer_idx][:nb], sel, axis=0)
            elif nb > 0:
                topk_sel     = None  # attend all blocks (use the nb-keyed residual cache)
                comp_U       = session["comp_U"][layer_idx][:nb]
                comp_VK      = session["comp_VK"][layer_idx][:nb]
                comp_VV      = session["comp_VV"][layer_idx][:nb]
                comp_anc_k   = session["comp_anc_k"][layer_idx][:nb]
                comp_anc_v   = session["comp_anc_v"][layer_idx][:nb]
                comp_scale   = session["comp_scale"][layer_idx][:nb]
                comp_seq_len = session["comp_seq_len"][layer_idx][:nb]
            else:
                topk_sel = None

            # ── Residual gather ───────────────────────────────────────────────────
            res_k_all = res_v_all = None
            total_res = 0
            if self.max_residual > 0 and nb > 0:
                if topk_sel is not None and all_blocks_full:
                    rk = mx.take(session["comp_res_k"][layer_idx][:nb], topk_sel, axis=0)
                    rv = mx.take(session["comp_res_v"][layer_idx][:nb], topk_sel, axis=0)
                    Ksel, Rw = rk.shape[0], rk.shape[1]
                    res_k_all = rk.transpose(2, 0, 1, 3).reshape(self.kv_heads, Ksel * Rw, self.head_dim)
                    res_v_all = rv.transpose(2, 0, 1, 3).reshape(self.kv_heads, Ksel * Rw, self.head_dim)
                    total_res = Ksel * Rw
                elif topk_sel is not None:
                    res_blocks = [int(i) for i in topk_sel.tolist()]
                    res_k_parts, res_v_parts = [], []
                    for bi in res_blocks:
                        n_res = session["comp_res_n"][layer_idx][bi]
                        if n_res > 0:
                            res_k_parts.append(session["comp_res_k"][layer_idx][bi, :n_res].transpose(1, 0, 2))
                            res_v_parts.append(session["comp_res_v"][layer_idx][bi, :n_res].transpose(1, 0, 2))
                    if res_k_parts:
                        res_k_all = mx.concatenate(res_k_parts, axis=1)
                        res_v_all = mx.concatenate(res_v_parts, axis=1)
                        total_res = res_k_all.shape[1]
                else:
                    rc = session.setdefault("_res_cache", {})
                    ent = rc.get(layer_idx)
                    if ent is not None and ent[0] == nb:
                        res_k_all, res_v_all, total_res = ent[1], ent[2], ent[3]
                    else:
                        res_k_parts, res_v_parts = [], []
                        for bi in range(nb):
                            n_res = session["comp_res_n"][layer_idx][bi]
                            if n_res > 0:
                                res_k_parts.append(session["comp_res_k"][layer_idx][bi, :n_res].transpose(1, 0, 2))
                                res_v_parts.append(session["comp_res_v"][layer_idx][bi, :n_res].transpose(1, 0, 2))
                        if res_k_parts:
                            res_k_all = mx.concatenate(res_k_parts, axis=1)   # [kv_heads, total_res, D]
                            res_v_all = mx.concatenate(res_v_parts, axis=1)
                            total_res = res_k_all.shape[1]
                            mx.eval(res_k_all, res_v_all)
                        rc[layer_idx] = (nb, res_k_all, res_v_all, total_res)

            if total_res > 0:
                dl = session["dense_lens"][layer_idx]
                dense_k_for_attn = mx.concatenate([res_k_all, dense_k], axis=1)
                dense_v_for_attn = mx.concatenate([res_v_all, dense_v], axis=1)
                dense_len_for_attn = mx.array(total_res + dl)
                current_max_dense_len = total_res + self.max_dense_len
                
                # Construct exact residuals mask + dense mask
                res_mask_attn = mx.ones((total_res,), dtype=mx.bool_)
                dense_mask_attn = mx.arange(self.max_dense_len) < dl
                dense_mask_combined = mx.concatenate([res_mask_attn, dense_mask_attn], axis=0)
            else:
                dense_k_for_attn = dense_k
                dense_v_for_attn = dense_v
                dense_len_for_attn = dense_len
                current_max_dense_len = self.max_dense_len
                dense_mask_combined = mx.arange(self.max_dense_len) < dense_len

            if nb == 0:
                out_combined = _dense_only_attention_static(
                    q, dense_k_for_attn, dense_v_for_attn, dense_len_for_attn,
                    scale, gpk, current_max_dense_len
                )
            else:
                S_comp = self.block_size - 1
                if self._res_exclude_svd and "comp_res_mask" in session:
                    _full_mask = session["comp_res_mask"][layer_idx][:nb]
                    res_mask = mx.take(_full_mask, topk_sel, axis=0) if topk_sel is not None else _full_mask
                else:
                    res_mask = mx.zeros((comp_U.shape[0], S_comp), dtype=mx.bool_)
                out_combined, _, _ = compute_decode_attention_static(
                    q, comp_U, comp_VK, comp_VV, comp_anc_k, comp_anc_v,
                    comp_scale, comp_seq_len, res_mask,
                    dense_k_for_attn, dense_v_for_attn, dense_mask_combined,
                    scale, gpk, self.kv_heads, self.block_size, self.rank,
                    current_max_dense_len,
                )

        if os.environ.get("DIFFKV_DBG_NAN") == "1":
            mx.eval(out_combined)
            qn = bool(mx.any(mx.isnan(q)).item())
            on = bool(mx.any(mx.isnan(out_combined)).item())
            print(f"[DBG] L={layer_idx} nb={nb} "
                  f"dl={int(session['dense_lens'][layer_idx])} q_nan={qn} out_nan={on}",
                  flush=True)

        return mx.expand_dims(mx.expand_dims(out_combined, 0), 2)

def scaled_dot_product_attention_mlx_basic(q: mx.array, k: mx.array, v: mx.array, scale: float, mask: Optional[Any] = None) -> mx.array:
    gpk = q.shape[1] // k.shape[1]
    if gpk > 1:
        k = mx.repeat(k, gpk, axis=1)
        v = mx.repeat(v, gpk, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L = q.shape[2]
            r = mx.arange(L)[:, None]
            c = mx.arange(L)[None, :]
            mask_arr = mx.where(r >= c, 0.0, -float("inf"))
            scores = scores + mask_arr
        else:
            scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v

def _resolve_compressed_decode(seq_len: int) -> bool:
    """Decide whether decode should use the DiffKV compressed sparse kernel.

    DIFFKV_COMPRESSED_DECODE:
      "1"/"on"/"true"   → always compressed (force the sparse path). THE DEFAULT —
                          DiffKV's sparse decode engages from token 1 so the
                          architecture is always exercised. TRADE-OFF: at short
                          context this is slower than fused dense (~16 vs ~36 tps
                          @4k) and pre-allocates the bounded block pool, with no
                          accuracy change — the memory/reach win only materializes
                          at long context. Accepted by design ("engage from the
                          start"); flip to "auto" if you want the adaptive policy.
      "0"/"off"/"false" → always dense (force exact full-KV attention)
      "auto"            → OPT-IN adaptive: dense below DIFFKV_COMPRESSED_MIN_CTX
                          (default 16384), sparse at/above it. Avoids the
                          short-context regression; use when raw short-prompt
                          throughput matters more than always exercising DiffKV.
    """
    mode = os.environ.get("DIFFKV_COMPRESSED_DECODE", "1").strip().lower()
    if mode in ("1", "on", "true", "yes"):
        return True
    if mode in ("0", "off", "false", "no"):
        return False
    threshold = int(os.environ.get("DIFFKV_COMPRESSED_MIN_CTX", "16384"))
    return seq_len >= threshold

def attention_forward(self, x: mx.array, mask: Optional[Any] = None, cache: Optional[Any] = None) -> mx.array:
    """Patched Qwen2 attention that:
    - During PREFILL: uses the native MLX KV cache (via `cache`) so that
      every chunk attends correctly over all preceding tokens.
      Also captures the K/V into DiffKV dense store for later decode use.
    - During DECODE (L==1): bypasses native cache entirely and uses our
      DiffKV compressed+dense attention.
    """
    if not hasattr(self, "kv_manager"):
        return self.original_call(x, mask, cache)

    B, L, D = x.shape
    manager = self.kv_manager
    layer_idx = self.layer_idx

    session_ids = manager.active_session_ids
    position_ids = manager.position_ids

    queries = self.q_proj(x)
    keys    = self.k_proj(x)
    values  = self.v_proj(x)

    queries = queries.reshape(B, L, self.n_heads,    -1).transpose(0, 2, 1, 3)
    keys    = keys.reshape(   B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values  = values.reshape( B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if B == 1:
        # Fast path: plain int offset (no per-layer mx.array creation) and no
        # single-element concatenate — decode calls this 28×/token.
        offset0 = int(position_ids[0, 0]) if position_ids is not None else 0
        queries_rot = self.rope(queries, offset=offset0)    # [1, H_q, L, D]
        keys_rot    = self.rope(keys,    offset=offset0)    # [1, H_kv, L, D]
    else:
        queries_rot_list = []
        keys_rot_list    = []
        for b_idx in range(B):
            offset = int(position_ids[b_idx, 0]) if position_ids is not None else 0
            queries_rot_list.append(self.rope(queries[b_idx:b_idx+1], offset=offset))
            keys_rot_list.append(  self.rope(keys[   b_idx:b_idx+1], offset=offset))

        queries_rot = mx.concatenate(queries_rot_list, axis=0)  # [B, H_q, L, D]
        keys_rot    = mx.concatenate(keys_rot_list,    axis=0)  # [B, H_kv, L, D]

    is_decode = (L == 1)

    if is_decode:
        # ── DECODE PATH ──
        # Two modes (selected by the DIFFKV_COMPRESSED_DECODE env flag):
        #   "1" → the attention output is produced by the DiffKV fused
        #         compressed+dense kernel (manager.execute_decode_attention →
        #         compute_decode_attention_static). The query is scored against
        #         every compressed block in low-rank space (never decompressing
        #         KV) plus the dense recency window, then LSE-combined. This is
        #         the real DiffKV sparse decode path.
        #   else → exact full-KV MLX attention over the native cache (the prior
        #         numerically-exact baseline). out_b is computed AFTER the ingest
        #         loop in both modes; the compressed path requires the current
        #         token to be ingested first so the query attends to history+self.
        # Route decode through the real DiffKV compressed+dense sparse attention
        # or exact full-KV MLX attention. The decision is resolved ONCE at the
        # prefill→decode boundary (MLXQwenModel.__call__) and stored per cache_key
        # so that this attention path and the cache-retention decision agree.
        # Fallback (direct attention use without the patched model): resolve from
        # the current sequence length.
        cache_key = tuple(session_ids)
        pm = getattr(manager, "patched_model", None)
        _decode_map = getattr(pm, "_decode_compressed", None) if pm is not None else None
        if _decode_map is not None and cache_key in _decode_map:
            _use_compressed_decode = _decode_map[cache_key]
        else:
            _seq_len = int(position_ids[0, 0]) if position_ids is not None else 0
            _use_compressed_decode = _resolve_compressed_decode(_seq_len)

        # Ingest the current token into the DiffKV store (architecture intact, and
        # required first for the compressed path so it can attend to itself).
        for b_idx in range(B):
            sid = session_ids[b_idx]
            if sid != "dummy_session":
                manager.ingest_streaming(
                    sid, layer_idx,
                    keys_rot[b_idx:b_idx+1],
                    values[ b_idx:b_idx+1]
                )

                # ── Factual store query (MLX decode path, layer 0 only) ──────
                # MLX attention bypasses diffkv_attention.py so the factual store
                # is never queried there. Use unrotated K at layer 0 as proxy Q
                # (same approach as C++ main.cpp). Results populate the factual
                # state fields read by the serving loop on the NEXT decode step.
                if layer_idx == 0:
                    try:
                        pool = getattr(manager, 'native_pool', None)
                        factual_store = getattr(manager, "_factual_stores", {}).get(sid)
                        srl_state = manager.get_srl_state(sid)
                        if (factual_store is not None and pool is not None
                                and pool.W_proj is not None and srl_state is not None
                                and factual_store.entries):
                            import numpy as _np
                            import torch as _torch
                            # keys: [B, H_kv, L=1, D] — extract [H_kv, D] for this batch element
                            k_np = _np.array(keys[b_idx, :, 0, :])
                            k_torch = _torch.from_numpy(k_np).float()

                            matching_entries = []
                            srl_state.current_step_factual_tokens = set()
                            srl_state.current_step_factual_sequences = []
                            srl_state.current_step_max_similarity = 0.0
                            srl_state.current_step_sequence_entity_ids = []
                            srl_state.current_step_sequence_is_prime = []
                            srl_state.current_step_sequence_prefixes = []

                            if srl_state.factual_anchor_q is None:
                                srl_state.factual_anchor_q = k_torch.detach().clone()
                                
                                # ── Early Entity Binding ─────────────────────────────────────────
                                try:
                                    query_toks = set(getattr(srl_state, "current_query_tokens", []))
                                    inv_index = getattr(srl_state, "inverted_index", None)
                                    if query_toks and factual_store is not None and inv_index is not None:
                                        important_query_toks = query_toks & inv_index.important_vocab
                                        prime_matches = []
                                        for fe in factual_store.entries:
                                            if getattr(fe, "is_prime", False):
                                                fe_important = set(fe.tokens) & inv_index.important_vocab
                                                overlap = len(important_query_toks & fe_important)
                                                if overlap >= 1:
                                                    prime_matches.append((fe.start_idx, overlap))
                                        if len(prime_matches) == 1:
                                            srl_state.current_entity_id = prime_matches[0][0]
                                            srl_state.dual_entity_mode = False
                                        elif len(prime_matches) >= 2:
                                            prime_matches.sort(key=lambda x: x[1], reverse=True)
                                            srl_state.dual_entity_mode = True
                                            srl_state.dual_entity_ids = [
                                                prime_matches[0][0],
                                                prime_matches[1][0],
                                            ]
                                            srl_state.comparison_entities = list(srl_state.dual_entity_ids)
                                            srl_state.comparison_active_idx = 0
                                            srl_state.comparison_covered = set()
                                            srl_state.current_entity_id = srl_state.comparison_entities[0]
                                except Exception:
                                    pass

                            q_for_factual = 0.20 * k_torch + 0.80 * srl_state.factual_anchor_q.to(k_torch.device)

                            _qbias = None
                            if getattr(srl_state, "dual_entity_mode", False) and getattr(srl_state, "dual_entity_ids", None):
                                _qbias = set(srl_state.dual_entity_ids)
                            elif getattr(srl_state, "current_entity_id", -1) != -1:
                                _qbias = {srl_state.current_entity_id}

                            # ── Positional query→value linking (MLX) ───────────────
                            # The descriptor match surfaces repeated FILLER on real
                            # docs (proven: it biased "and confirm that…" not the
                            # answer). Instead, bind the query's DISTINCTIVE (high-IDF)
                            # tokens to WHERE they occur in the document, and surface
                            # the fact spans co-located with them — i.e. connect the
                            # queried entity to its own value span, not filler. Only
                            # falls back to descriptor matching when no such anchor
                            # exists. This is what makes the store help, not derail.
                            matching_entries = None
                            try:
                                _inv = getattr(srl_state, "inverted_index", None)
                                _qtoks = getattr(srl_state, "current_query_tokens", [])
                                if (_inv is not None and _qtoks
                                        and getattr(_inv, "occurrences", None)
                                        and getattr(_inv, "idf", None)):
                                    _IDF_MIN = float(os.environ.get("DIFFKV_FACTUAL_IDF_MIN", "3.0"))
                                    _WIN = int(os.environ.get("DIFFKV_FACTUAL_WINDOW", "40"))
                                    # Max TOTAL occurrences for an anchor token. Block-IDF
                                    # alone is fooled when a whole table sits in one block
                                    # (shared words like "module"/"key" get high block-IDF
                                    # despite repeating); also require the token to be
                                    # genuinely rare document-wide so only distinctive
                                    # names (occur ~1-2×) anchor, not repeated schema words.
                                    _MAX_OCC = int(os.environ.get("DIFFKV_FACTUAL_MAX_OCC", "4"))
                                    _anchors = []
                                    for _qt in set(_qtoks):
                                        _occ = _inv.occurrences.get(_qt)
                                        # Distinctiveness = RARE (few total occurrences).
                                        # Block-IDF alone is fragile: on a short doc a
                                        # name split across the registry + question block
                                        # dips below IDF_MIN even though it occurs ~twice.
                                        # So very-rare tokens (≤2 occ) anchor regardless of
                                        # block-IDF; MAX_OCC stays the primary gate.
                                        if _occ and len(_occ) <= _MAX_OCC and (
                                                len(_occ) <= 2 or _inv.idf.get(_qt, 0.0) >= _IDF_MIN):
                                            _anchors.extend(p for (_s, p, _r) in _occ)
                                    if _anchors:
                                        # For each anchor take the SINGLE NEAREST fact
                                        # span (min distance to the span, 0 if inside),
                                        # not every span in the window — otherwise a
                                        # dense table (facts <window apart) surfaces all
                                        # of them and the bias can't discriminate.
                                        # Skip spans that are mostly QUERY tokens — those
                                        # are the question/instruction text at the tail
                                        # (the query's distinctive token also occurs there,
                                        # so its anchor would otherwise pull them in as
                                        # noise instead of the actual fact span).
                                        _qset = set(_qtoks)
                                        _pos_map = {}
                                        _primary_i, _primary_d = -1, _WIN + 1
                                        for _p in _anchors:
                                            _best_i, _best_d = -1, _WIN + 1
                                            for _i, _e in enumerate(factual_store.entries):
                                                _s0 = getattr(_e, "start_idx", -1)
                                                _e0 = getattr(_e, "end_idx", _s0)
                                                if _s0 < 0:
                                                    continue
                                                _et = getattr(_e, "tokens", None)
                                                if _et and (len(set(_et) & _qset) / len(_et)) > 0.5:
                                                    continue
                                                _d = 0 if (_s0 <= _p <= _e0) else min(abs(_s0 - _p), abs(_e0 - _p))
                                                if _d < _best_d:
                                                    _best_d, _best_i = _d, _i
                                            if _best_i >= 0 and _best_d <= _WIN:
                                                _pos_map[_best_i] = factual_store.entries[_best_i]
                                                if _best_d < _primary_d:
                                                    _primary_d, _primary_i = _best_d, _best_i
                                        if _pos_map:
                                            for _e in _pos_map.values():
                                                _e.current_sim = 1.0
                                            matching_entries = list(_pos_map.values())
                                            # Align entity binding with the positional
                                            # result: the NEAREST co-located fact defines
                                            # the queried entity. Overrides the raw-overlap
                                            # early-binding, which goes dual/entity-0 on
                                            # repetitive prompts and locks the wrong entity.
                                            if _primary_i >= 0:
                                                _peid = getattr(factual_store.entries[_primary_i], "entity_id", -1)
                                                if _peid != -1:
                                                    srl_state.current_entity_id = _peid
                                                    srl_state.dual_entity_mode = False
                                                    srl_state.dual_entity_ids = []
                            except Exception:
                                matching_entries = None

                            # When positional linking pinned the queried entity's own
                            # fact span(s), inject ONLY those — skip neighbor/triple
                            # expansion, which on a dense table pulls in ADJACENT rows'
                            # keys and re-muddies the bias.
                            _positional_used = matching_entries is not None

                            if matching_entries is None:
                                matching_entries = factual_store.query(
                                    Q=q_for_factual,
                                    W_proj=pool.W_proj,
                                    threshold=0.3,
                                    active_slots=None,
                                    query_entity_bias=_qbias,
                                )

                            if matching_entries:
                                for entry in matching_entries:
                                    srl_state.current_step_factual_tokens.update(entry.tokens)
                                    if entry.tokens not in srl_state.current_step_factual_sequences:
                                        srl_state.current_step_factual_sequences.append(entry.tokens)
                                    
                                    # RC1 — inject triple sequences from prime entries.
                                    if getattr(entry, "is_prime", False) and not _positional_used:
                                        for triple_seq in getattr(entry, "triple_sequences", []):
                                            if triple_seq and triple_seq not in srl_state.current_step_factual_sequences:
                                                srl_state.current_step_factual_sequences.append(triple_seq)
                                                srl_state.current_step_factual_tokens.update(triple_seq)

                                    # ── 1-hop neighbor injection ────────────────────────────
                                    for nb_idx, nb_weight in (zip(entry.neighbors, entry.weights) if not _positional_used else []):
                                        if nb_weight >= 0.35 and nb_idx < len(factual_store.entries):
                                            nb_e = factual_store.entries[nb_idx]
                                            if nb_e.tokens and nb_e.tokens not in srl_state.current_step_factual_sequences:
                                                srl_state.current_step_factual_tokens.update(nb_e.tokens)
                                                srl_state.current_step_factual_sequences.append(nb_e.tokens)
                                            if getattr(nb_e, "is_prime", False):
                                                for triple_seq in getattr(nb_e, "triple_sequences", []):
                                                    if triple_seq and triple_seq not in srl_state.current_step_factual_sequences:
                                                        srl_state.current_step_factual_sequences.append(triple_seq)
                                                        srl_state.current_step_factual_tokens.update(triple_seq)
                                            
                                            # ── 2-hop neighbor injection ────────────────
                                            for nb2_idx, nb2_weight in zip(nb_e.neighbors, nb_e.weights):
                                                if nb2_weight >= 0.50 and nb2_idx < len(factual_store.entries):
                                                    nb2_e = factual_store.entries[nb2_idx]
                                                    if nb2_e.tokens and nb2_e.tokens not in srl_state.current_step_factual_sequences:
                                                        srl_state.current_step_factual_tokens.update(nb2_e.tokens)
                                                        srl_state.current_step_factual_sequences.append(nb2_e.tokens)

                                # Coherence Cap sorting & truncation (Solution 6)
                                session_config = getattr(manager, "session_configs", {}).get(sid, {})
                                base_coherence = session_config.get("coherence_cap", 8)
                                num_active = 1
                                if getattr(srl_state, "dual_entity_mode", False):
                                    num_active = 2
                                coherence_cap = base_coherence + 4 * num_active

                                seq_id_to_score = {}
                                for fe_e in matching_entries:
                                    f_sim = getattr(fe_e, "current_sim", 0.0)
                                    if f_sim > 0:
                                        seq_id_to_score[tuple(fe_e.tokens)] = f_sim
                                        # 1-hop neighbors inherit score
                                        for nb_idx, nb_w in zip(fe_e.neighbors, fe_e.weights):
                                            if nb_w >= 0.35 and nb_idx < len(factual_store.entries):
                                                nb_toks = tuple(factual_store.entries[nb_idx].tokens)
                                                if nb_toks not in seq_id_to_score:
                                                    seq_id_to_score[nb_toks] = f_sim * nb_w
                                        # Triple sequences inherit prime's score
                                        if getattr(fe_e, "is_prime", False):
                                            for ts in getattr(fe_e, "triple_sequences", []):
                                                seq_id_to_score[tuple(ts)] = f_sim

                                srl_state.current_step_factual_sequences.sort(
                                    key=lambda s: seq_id_to_score.get(tuple(s), 0.0), reverse=True
                                )
                                srl_state.current_step_factual_sequences = srl_state.current_step_factual_sequences[:coherence_cap]
                                srl_state.current_step_factual_tokens = set()
                                for s in srl_state.current_step_factual_sequences:
                                    srl_state.current_step_factual_tokens.update(s)

                                # ── Entity-Subgraph Tagging ───────────────────────────────────
                                entry_meta = {}
                                for fe in factual_store.entries:
                                    tup = tuple(fe.tokens)
                                    entry_meta[tup] = (
                                        getattr(fe, "entity_id", -1),
                                        getattr(fe, "is_prime", False),
                                        getattr(fe, "prefix_tokens", []),
                                    )
                                triple_to_entity = {}
                                for fe in factual_store.entries:
                                    if getattr(fe, "is_prime", False):
                                        p_entity = getattr(fe, "entity_id", -1)
                                        for ts in getattr(fe, "triple_sequences", []):
                                            triple_to_entity[tuple(ts)] = p_entity

                                entity_ids = []
                                is_prime_flags = []
                                seq_prefixes = []
                                for seq in srl_state.current_step_factual_sequences:
                                    tup = tuple(seq)
                                    if tup in entry_meta:
                                        eid, isp, pref = entry_meta[tup]
                                    elif tup in triple_to_entity:
                                        eid, isp, pref = triple_to_entity[tup], False, []
                                    else:
                                        eid, isp, pref = -1, False, []
                                    entity_ids.append(eid)
                                    is_prime_flags.append(isp)
                                    seq_prefixes.append(list(pref))
                                srl_state.current_step_sequence_entity_ids = entity_ids
                                srl_state.current_step_sequence_is_prime = is_prime_flags
                                srl_state.current_step_sequence_prefixes = seq_prefixes

                                sims = [getattr(e, "current_sim", 0.0) for e in matching_entries]
                                if sims:
                                    srl_state.current_step_max_similarity = max(sims)

                            if os.environ.get("DIFFKV_FACTUAL_DBG") == "1":
                                try:
                                    _tk = getattr(manager, "tokenizer", None)
                                    _seqs = srl_state.current_step_factual_sequences[:6]
                                    _dec = [(_tk.decode(s) if _tk else s) for s in _seqs]
                                    print(f"[FDBG] eid={getattr(srl_state,'current_entity_id',-1)} "
                                          f"dual={getattr(srl_state,'dual_entity_mode',False)} "
                                          f"maxsim={getattr(srl_state,'current_step_max_similarity',0):.2f} "
                                          f"nseq={len(srl_state.current_step_factual_sequences)} seqs={_dec}", flush=True)
                                except Exception:
                                    pass
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[DiffKV MLX Exception] SRL update failed: {e}")
                # ────────────────────────────────────────────────────────────

        # ── Compute the attention output AFTER the ingest loop ──
        if _use_compressed_decode:
            outs = []
            for b_idx in range(B):
                sid = session_ids[b_idx]
                if sid == "dummy_session":
                    # No DiffKV state for this element — fall back to exact attention.
                    if cache is not None:
                        ak, av = _cache_fetch(
                            cache, keys_rot[b_idx:b_idx + 1], values[b_idx:b_idx + 1])
                    else:
                        ak, av = keys_rot[b_idx:b_idx + 1], values[b_idx:b_idx + 1]
                    outs.append(mx.fast.scaled_dot_product_attention(
                        queries_rot[b_idx:b_idx + 1], ak, av,
                        scale=self.scale, mask=mask))
                else:
                    outs.append(manager.execute_decode_attention(
                        sid, layer_idx, queries_rot[b_idx:b_idx + 1],
                        self.rope, self.scale, self.n_heads // self.n_kv_heads))
            out_b = mx.concatenate(outs, axis=0)
        else:
            if cache is not None:
                all_k, all_v = _cache_fetch(cache, keys_rot, values)
            else:
                all_k, all_v = keys_rot, values
            out_b = mx.fast.scaled_dot_product_attention(
                queries_rot, all_k, all_v, scale=self.scale, mask=mask)

        output = out_b.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    else:
        # ── PREFILL PATH ── use native MLX cache for correct causal attention,
        #                   then also capture K/V into DiffKV store.
        #
        # The original qwen2.py does:
        #   keys, values = cache.update_and_fetch(keys, values)
        #   output = scaled_dot_product_attention(queries, keys, values, ...)
        #
        # update_and_fetch accumulates all past KV into the cache and returns
        # the full [1, kv_heads, total_seq_len, head_dim] tensor, so every
        # chunk attends over ALL previous tokens correctly.

        # ── LEGO streaming prefill (DIFFKV_LEGO_PREFILL) ──
        # Once engaged, the chunk attends [raw sinks | routed compressed far blocks |
        # raw recency ring | self] entirely from the session state; the raw prompt
        # cache is NOT updated (and is dropped by MLXQwenModel.__call__), bounding
        # prefill raw KV by O(sinks + ring + chunk) instead of O(T).
        _lego_out = None
        if manager._lego_prefill and B == 1 and session_ids[0] != "dummy_session":
            _lego_sess = manager.sessions.get(session_ids[0])
            _cur_start_lego = int(position_ids[0, 0]) if position_ids is not None else 0
            if _lego_sess is not None and manager._lego_session_ready(
                    _lego_sess, layer_idx, _cur_start_lego):
                if layer_idx == 0:
                    _lego_sess["lego_engaged"] = True
                _lego_out = _lego_prefill_attend(
                    manager, _lego_sess, layer_idx,
                    queries_rot, keys_rot, values, _cur_start_lego,
                    self.scale, self.n_heads // self.n_kv_heads, manager._sp_dbg,
                )

        # DIFFKV_LEGO_SHADOW=1 — parity diagnostic: compute the lego attention but
        # USE the raw path's output (cache keeps updating), printing lego-vs-exact
        # error per layer with identical inputs. Isolates attend-time bugs from
        # compounding capture drift.
        _lego_shadow = _lego_out is not None and os.environ.get("DIFFKV_LEGO_SHADOW", "0") == "1"
        if _lego_out is not None and not _lego_shadow:
            out_b = _lego_out
        else:
            if cache is not None:
                all_k, all_v = _cache_fetch(cache, keys_rot, values)
            else:
                all_k, all_v = keys_rot, values
            if _lego_shadow:
                _a = _lego_out.astype(mx.float32).reshape(-1)
                _ref = mx.fast.scaled_dot_product_attention(
                    queries_rot, all_k, all_v, scale=self.scale, mask=mask)
                _b = _ref.astype(mx.float32).reshape(-1)
                _cos = mx.sum(_a * _b) / (mx.sqrt(mx.sum(_a * _a)) * mx.sqrt(mx.sum(_b * _b)) + 1e-9)
                _md = mx.max(mx.abs(_a - _b))
                if layer_idx in (0, 13, 27):
                    # Also profile the VALIDATED sparse prefill against the same
                    # dense reference — the acceptance bar for lego's error.
                    _cur0 = int(position_ids[0, 0])
                    _sp = _sparse_prefill_attend(
                        queries_rot, all_k, all_v, _cur0,
                        self.scale, self.n_heads // self.n_kv_heads,
                        manager.block_size, manager._sp_window, manager._sp_sink_blocks,
                        manager._sp_kmin, manager._sp_frac, False)
                    _c = _sp.astype(mx.float32).reshape(-1)
                    _cos_sp = mx.sum(_c * _b) / (mx.sqrt(mx.sum(_c * _c)) * mx.sqrt(mx.sum(_b * _b)) + 1e-9)
                    _md_sp = mx.max(mx.abs(_c - _b))
                    print(f"[LEGO-PAR] cur={_cur0} l={layer_idx} "
                          f"lego: cos={float(_cos):.6f} max|d|={float(_md):.4f}  "
                          f"sparse: cos={float(_cos_sp):.6f} max|d|={float(_md_sp):.4f}", flush=True)

            # ── DSA/NSA-style sparse prefill (DIFFKV_SPARSE_PREFILL) ──
            # Attend to [sink blocks + top-K routed history blocks + recency window + self]
            # instead of the full accumulated KV, once the chunk is far enough in that there
            # is prunable history. Default OFF; verified via niah_recall before any default flip.
            _T = all_k.shape[2]
            _cur_start = _T - L
            if (manager._sparse_prefill and _cur_start >= manager._sp_min_ctx):
                out_b = _sparse_prefill_attend(
                    queries_rot, all_k, all_v, _cur_start,
                    self.scale, self.n_heads // self.n_kv_heads,
                    manager.block_size, manager._sp_window, manager._sp_sink_blocks,
                    manager._sp_kmin, manager._sp_frac, manager._sp_dbg,
                )
            else:
                out_b = mx.fast.scaled_dot_product_attention(
                    queries_rot,
                    all_k,
                    all_v,
                    scale=self.scale,
                    mask=mask
                )

        # 2. Capture ONLY the current chunk's K/V into DiffKV store
        #    (all_k/all_v grow with every chunk; we store incrementally)
        for b_idx in range(B):
            sid = session_ids[b_idx]
            if sid == "dummy_session":
                continue
            manager.capture_prefill_kv(
                sid, layer_idx,
                keys_rot[b_idx:b_idx+1],
                values[ b_idx:b_idx+1]
            )
            # Stash UNROTATED K/V (layers 0 + middle only) for the optional factual
            # store — its descriptors must share the unrotated layer-0 space the
            # decode-time query uses. No-op unless DIFFKV_FACTUAL_STORE is enabled.
            manager.capture_factual_prefill_kv(
                sid, layer_idx,
                keys[b_idx:b_idx+1],
                values[b_idx:b_idx+1]
            )

        output = out_b.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

class MLXQwenModel:
    def __init__(self, mlx_model, manager):
        self.mlx_model = mlx_model
        self.manager = manager
        self._diffkv_session_ids = ["default"]
        # Per-session KVCache lists kept alive across prefill+decode.
        self._prefill_caches: dict = {}
        # Tracks whether the previous call was a prefill, per cache_key.
        # Used to fire mx.clear_cache() exactly once at the prefill→decode boundary.
        self._prev_was_prefill: dict = {}
        # Decode path decision (compressed sparse vs exact dense) resolved once at
        # the prefill→decode boundary, per cache_key. Read by attention_forward.
        self._decode_compressed: dict = {}

    def _prefill_forward_last_only(self, inputs_mx: mx.array, prefill_cache):
        """Prefill forward that computes the LM head for the LAST position only.

        The stock model call applies lm_head to every position of the chunk —
        a [L, vocab] matmul (512×151936 ≈ 16% of total prefill FLOPs at 1.5B)
        whose result is then round-tripped to fp32 numpy/torch (~600 MB of
        transient host memory per 512-token chunk). During prefill only the
        final position's logits are ever consumed, so run the backbone for the
        full chunk (KV capture needs every position) but the head on hidden[-1:]
        only. Bit-identical to slicing the full logits afterwards.

        Speculative decoding is the one consumer of full multi-token logits
        (draft verification); it sets keep_prefill_cache, which routes back to
        the stock full-logits call.
        """
        m = self.mlx_model
        backbone = getattr(m, "model", None)
        if getattr(self, "keep_prefill_cache", False) or backbone is None:
            return m(inputs_mx, cache=prefill_cache)
        hidden = backbone(inputs_mx, cache=prefill_cache)   # [1, L, H]
        last = hidden[:, -1:, :]
        if getattr(getattr(m, "args", None), "tie_word_embeddings", False):
            return backbone.embed_tokens.as_linear(last)
        lm_head = getattr(m, "lm_head", None)
        if lm_head is not None:
            return lm_head(last)
        return backbone.embed_tokens.as_linear(last)

    def _get_or_create_prefill_cache(self, cache_key: tuple, total_tokens: int = 0):
        if cache_key not in self._prefill_caches:
            from mlx_lm.models.cache import make_prompt_cache, KVCache, QuantizedKVCache
            # DIFFKV_PREFILL_CACHE_BITS: bit-width for the prompt KV cache during prefill.
            #   16 = standard float16 KVCache (1.4 GB at 12k for 3B model)
            #    8 = 8-bit QuantizedKVCache  (~700 MB — DEFAULT, saves 700 MB)
            #    4 = 4-bit QuantizedKVCache  (~350 MB — saves 1.05 GB, slightly lossy)
            # Dequantization is done transparently by _cache_fetch() in attention_forward
            # so all downstream code sees plain float16 mx.array tensors as before.
            # Restore default to 16 (float16) for full precision and to prevent loops.
            # Users can override this to 8 or 4 via DIFFKV_PREFILL_CACHE_BITS if needed.
            _cache_bits = int(os.environ.get("DIFFKV_PREFILL_CACHE_BITS", "16"))
            if _cache_bits in (4, 8):
                cache_list = [
                    QuantizedKVCache(group_size=64, bits=_cache_bits)
                    for _ in range(len(self.mlx_model.layers))
                ]
            else:
                # float16 — pre-size to avoid concatenation reallocs.
                cache_list = make_prompt_cache(self.mlx_model)
                if total_tokens > 0:
                    step = max(256, ((total_tokens + 255) // 256) * 256)
                    for c in cache_list:
                        if isinstance(c, KVCache):
                            c.step = step
            self._prefill_caches[cache_key] = cache_list
        return self._prefill_caches[cache_key]

    def __call__(self, input_ids: torch.Tensor, position_ids: torch.Tensor, use_cache: bool = True):
        inputs_np = input_ids.detach().cpu().numpy()
        inputs_mx = mx.array(inputs_np)

        self.manager.active_session_ids = self._diffkv_session_ids
        self.manager.position_ids = position_ids.detach().cpu().numpy() if position_ids is not None else None

        is_prefill = (input_ids.shape[1] > 1)
        cache_key = tuple(self._diffkv_session_ids)

        if is_prefill:
            # Pass the accumulated prefill cache so each chunk attends over
            # ALL previous tokens — this gives correct causal hidden states.
            # total_seq_len hint: position_ids tells us the absolute end position of
            # the final chunk. We pass it at first-chunk time so the cache is
            # pre-sized and avoids concatenation reallocs on subsequent chunks.
            _total_hint = 0
            if position_ids is not None:
                try:
                    _total_hint = int(position_ids[0, -1].item()) + 1
                except Exception:
                    pass
            prefill_cache = self._get_or_create_prefill_cache(cache_key, total_tokens=_total_hint)
            logits_mx = self._prefill_forward_last_only(inputs_mx, prefill_cache)

            # ── LEGO prefill: the raw prompt cache is dead weight once engaged ──
            # Engaged chunks attend the DiffKV session state only (sinks + compressed
            # blocks + dense tail) and never update the cache, so drop it now instead
            # of carrying a full-context raw copy to the decode boundary. This is THE
            # memory win of lego mode: prefill raw KV stops growing with T.
            if (self.manager._lego_prefill
                    and os.environ.get("DIFFKV_LEGO_SHADOW", "0") != "1"
                    and not getattr(self, "keep_prefill_cache", False)
                    and cache_key in self._prefill_caches):
                for _sid in self._diffkv_session_ids:
                    _s = self.manager.sessions.get(_sid)
                    if _s is not None and _s.get("lego_engaged"):
                        mx.eval(logits_mx)
                        self._prefill_caches.pop(cache_key, None)
                        mx.clear_cache()
                        break
        else:
            # ── Prefill → Decode transition ──────────────────────────────────
            # MLX's allocator holds onto the peak GQA-expanded K/V tensors from
            # the final prefill chunk (e.g. [1, 12, 8192, 128] × 28 layers × 2
            # ≈ 1.4 GB). These are no longer needed once decode begins.
            # mx.clear_cache() releases them back to the OS immediately.
            if self._prev_was_prefill.get(cache_key, True):
                # Resolve the decode path ONCE here, at the prefill→decode
                # boundary, from the prompt length. Storing it per cache_key keeps
                # the attention layers and the cache-retention choice consistent.
                seq_len = int(position_ids[0, 0].item()) if position_ids is not None else 0
                use_compressed = _resolve_compressed_decode(seq_len)
                self._decode_compressed[cache_key] = use_compressed

                # Build the factual store / SRL state from the captured prefill KV
                # BEFORE the first decode step queries it. No-op unless enabled.
                for _sid in self._diffkv_session_ids:
                    if _sid != "dummy_session":
                        # print(f"[DBG] __call__ Finalizing SRL index...", flush=True)
                        self.manager.finalize_srl_index(_sid)

                # print(f"[DBG] __call__ transition: mx.eval() + mx.clear_cache() + gc.collect()", flush=True)
                mx.eval()          # flush any pending lazy ops first
                mx.clear_cache()   # return peak activation memory to OS
                import gc; gc.collect()
                if use_compressed and not getattr(self, "keep_prefill_cache", False):
                    # Compressed decode runs entirely on the DiffKV store
                    # (compressed blocks + dense recency window), so the full
                    # native prefill KV cache is no longer needed. Drop it so
                    # decode-time memory reflects the DiffKV footprint, not a
                    # retained full-context cache.
                    self._prefill_caches.pop(cache_key, None)
                    # Lego prefill state (raw sinks + recency-ring buffers, up to
                    # ~250 MB at ring 4096 on a 28-layer model) is prefill-only —
                    # decode attends the compressed store. Free it, and clear the
                    # engaged flag so a later cached-prefix continuation doesn't
                    # try to attend freed buffers (it falls back to the raw path,
                    # matching the existing continuation behavior).
                    for _sid in self._diffkv_session_ids:
                        _s = self.manager.sessions.get(_sid)
                        if _s is not None and _s.get("lego_engaged"):
                            for _k in ("lego_ring_bufs", "lego_ring_k", "lego_ring_v",
                                       "lego_sink_k", "lego_sink_v", "lego_sink_len",
                                       "lego_ring_start"):
                                _s.pop(_k, None)
                            _s["lego_engaged"] = False
                            _s.pop("_lego_ok", None)
                    mx.clear_cache(); gc.collect()

            # Decode: keep the same cache alive so decode tokens attend over
            # the full prefill context + all previously decoded tokens.
            # (When compressed decode dropped it above, this is None and the
            # patched attention uses the DiffKV store instead.)
            decode_cache = self._prefill_caches.get(cache_key)
            # print(f"[DBG] __call__ is_prefill=False: shape={input_ids.shape} starting model forward...", flush=True)
            logits_mx = self.mlx_model(inputs_mx, cache=decode_cache)
            # print(f"[DBG] __call__ model forward completed.", flush=True)

        self._prev_was_prefill[cache_key] = is_prefill

        # print(f"[DBG] __call__ mx.eval(logits_mx) starting...", flush=True)
        mx.eval(logits_mx)
        # print(f"[DBG] __call__ mx.eval(logits_mx) completed.", flush=True)

        logits_np = np.array(logits_mx.astype(mx.float32))
        logits_py = torch.from_numpy(logits_np).to(device=input_ids.device)

        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
                self.past_key_values = None

        return ModelOutput(logits_py)



class MLXDiffKVWrapper:
    def __init__(
        self, 
        model_id: str,
        config: Dict[str, Any],
        device: str = None,
        quantization_config: Any = None,
        torch_dtype: Any = None,
        lazy: bool = False,
    ):
        self.model_id = model_id
        self.config = config or {}
        self.lazy = lazy
        self.is_mlx = True
        
        self.block_size = self.config.get("block_size", 256)
        self.rank = self.config.get("rank", 32)
        self.micro_block_size = self.config.get("micro_block_size", 256)
        self.device = "mps"
        
        self.tokenizer = None
        self.stop_token_ids = set()
        self.model = None
        self.manager = None
        self.active_session = None
        self._session_token_ids = {}

        if not self.lazy:
            self.ensure_loaded()

    def ensure_loaded(self):
        if self.model is not None:
            return

        # Bound the MLX buffer cache. During chunked prefill every chunk's
        # tensors have new shapes (context grows), so cached buffers from
        # earlier chunks are never reused and the cache grows superlinearly:
        # measured 4.07 GB dead cache (~6.1 GB total allocator footprint) on a
        # 13.2k-token prefill. A 1 GB limit cut peak to ~3.0 GB AND made the
        # same prefill 15% faster (31.7s -> 27.1s) on the 8 GB M3 machine.
        # DIFFKV_CACHE_LIMIT_GB=0 disables the cap.
        cache_gb = os.environ.get("DIFFKV_CACHE_LIMIT_GB", "1")
        try:
            cache_bytes = int(float(cache_gb) * 1e9)
        except ValueError:
            cache_bytes = 0
        if cache_bytes > 0:
            _set_limit = getattr(mx, "set_cache_limit", None) or getattr(mx.metal, "set_cache_limit", None)
            if _set_limit is not None:
                _set_limit(cache_bytes)

        model_id = self.model_id
        quant = self.config.get("quantization")
        
        preset = self.config.get("preset", os.environ.get("DIFFKV_PRESET", "mid")).lower()
        if preset == "low" and not quant and not os.environ.get("DIFFKV_QUANTIZATION"):
            quant = "int4"
            print("[DiffKV MLX] Low preset: auto-enabling 4-bit quantization")

        if quant in ("int4", "int8") and not model_id.startswith("mlx-community/"):
            parts = model_id.split("/")
            if len(parts) == 2:
                org, name = parts
                suffix = "4bit" if quant == "int4" else "8bit"
                model_id = f"mlx-community/{name}-{suffix}"
                print(f"[DiffKV MLX] Loading quantized model: {model_id}")

        print(f"[DiffKV MLX] Loading model via mlx_lm: {model_id}...")
        t0 = time.time()
        model, tokenizer = mlx_load(model_id)
        print(f"[DiffKV MLX] Loaded model in {time.time() - t0:.2f}s")
        
        self.tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        
        self.stop_token_ids = set()
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            self.stop_token_ids.update(eos_id)
        elif isinstance(eos_id, int):
            self.stop_token_ids.add(eos_id)
            
        special_words = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
        for word in special_words:
            tok_id = self.tokenizer.convert_tokens_to_ids(word)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tok_id)

        self.manager = MLXKVBlockManager(
            num_layers=len(model.layers),
            heads=model.model.layers[0].self_attn.n_heads,
            kv_heads=model.model.layers[0].self_attn.n_kv_heads,
            head_dim=model.model.layers[0].self_attn.q_proj.weight.shape[0] // model.model.layers[0].self_attn.n_heads,
            rank=self.rank,
            block_size=self.block_size
        )
        # Hand the manager what FactualExactStore.build / setup_sas_and_eqa need
        # (no-ops unless DIFFKV_FACTUAL_STORE is enabled).
        self.manager.tokenizer = self.tokenizer
        self.manager._stop_token_ids = set(self.stop_token_ids)
        self._session_token_ids = self.manager._session_token_ids
        
        self._patch_attention_layers(model)
        self.model = MLXQwenModel(model, self.manager)
        self.model.keep_prefill_cache = (
            self.config.get("draft_model") is not None
            or os.environ.get("DIFFKV_SPECULATIVE", "0") == "1"
        )
        self.manager.patched_model = self.model

    def _patch_attention_layers(self, model):
        # Dynamically find and patch the attention class of the loaded model
        if len(model.model.layers) > 0:
            attn_class = model.model.layers[0].self_attn.__class__
            if not hasattr(attn_class, "original_call"):
                attn_class.original_call = attn_class.__call__
            attn_class.__call__ = attention_forward
        
        for layer_idx, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = layer_idx
            layer.self_attn.kv_manager = self.manager

    def close(self):
        if self.manager is not None:
            self.manager.sessions.clear()
            self.manager = None
        self.model = None
        self.tokenizer = None
        mx.metal.clear_cache()

    def stop(self):
        self.close()

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        query_text: Optional[str] = None,
    ) -> str:
        self.ensure_loaded()
        session_id = self.active_session or "default"

        # Squeeze prompt tokenization
        prompt_ids = self.tokenizer.encode(prompt)

        # Entity-binding hint: the part of the prompt that is the actual question.
        # Used by the factual store to bind decode to the queried entity (no-op
        # unless the factual store is enabled). Falls back to the uncached tail.
        if query_text and getattr(self.manager, "_factual_enabled", False):
            try:
                self.manager._pending_query[session_id] = self.tokenizer.encode(query_text)
            except Exception:
                pass
        
        # Check cache reuse
        cached_len = 0
        if session_id in self._session_token_ids:
            stored_ids = self._session_token_ids[session_id]
            if len(stored_ids) > 0 and len(stored_ids) < len(prompt_ids):
                if prompt_ids[:len(stored_ids)] == stored_ids:
                    cached_len = len(stored_ids)
                    print(f"[DiffKV MLX Wrapper] Reusing {cached_len} cached tokens!")
                    
        if cached_len == 0:
            self.manager.clear_session(session_id)
            self._session_token_ids[session_id] = []
            new_prompt_ids = prompt_ids
        else:
            new_prompt_ids = prompt_ids[cached_len:]

        self.manager.init_session(session_id, prefill_len=cached_len + len(new_prompt_ids))
        self.manager.register_prefill_tokens(session_id, torch.tensor(new_prompt_ids, dtype=torch.long))
        self.model._diffkv_session_ids = [session_id]

        # ── Pre-size the KVCache to the full prompt length BEFORE the loop. ──
        # The default KVCache grows in 256-token steps via mx.concatenate.
        # Pre-sizing here ensures the backing buffer is allocated once, so all
        # chunked forward passes write in-place with no concatenation reallocs.
        _total_prefill_len = cached_len + len(new_prompt_ids)
        _cache_key = tuple([session_id])
        self.model._get_or_create_prefill_cache(_cache_key, total_tokens=_total_prefill_len)

        # ── Chunked Prefill ──
        # We process the prompt in 512-token chunks. After EACH chunk we:
        #   1. Run SVD compression on completed blocks (compress_deferred_prefill_blocks)
        #   2. mx.eval() — force MLX to run the lazy graph for this chunk so its
        #      intermediate activation tensors (GQA projections, softmax buffers,
        #      rotary embeddings) are materialized and their graph references dropped.
        #   3. mx.clear_cache() — flush MLX allocator's free-but-retained buffers
        #      back to the OS before the next chunk allocates on top of them.
        # WITHOUT steps 2+3 all chunks' transient tensors pile up simultaneously in
        # the MLX allocator cache, making peak RAM proportional to the full prompt
        # length instead of one chunk — the root cause of the 7–8 GB spike at 12k.
        import gc as _gc
        PREFILL_CHUNK = 512
        output = None
        total_chunks = (len(new_prompt_ids) + PREFILL_CHUNK - 1) // PREFILL_CHUNK
        for i, chunk_start in enumerate(range(0, len(new_prompt_ids), PREFILL_CHUNK)):
            chunk = new_prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
            clen = len(chunk)
            abs_start = cached_len + chunk_start

            chunk_tensor = torch.tensor([chunk], dtype=torch.long)
            pos_tensor = torch.tensor([list(range(abs_start, abs_start + clen))], dtype=torch.long)

            output = self.model(chunk_tensor, pos_tensor)
            self.manager.compress_deferred_prefill_blocks(session_id)

            # ── Per-chunk memory release ──────────────────────────────────────
            # Materialize this chunk's output so the graph and its intermediate
            # tensors are fully evaluated (and thus eligible for GC).
            mx.eval(output.logits)
            # Return freed MLX allocator buffers to OS so the next chunk doesn't
            # stack on top of the previous chunk's transient memory.
            mx.clear_cache()
            _gc.collect()

        # Complete sequence prefill done
        generated = prompt_ids.copy()
        srl_state = self.manager.get_srl_state(session_id)
        if srl_state is not None:
            srl_state.vsl_active_candidates = []
            srl_state.vsl_consecutive_helpers = 0
            srl_state.factual_anchor_q = None
            srl_state.current_entity_id = -1
            srl_state.dual_entity_mode = False
            srl_state.dual_entity_ids = []
            
        # ── Decoding loop ──
        cur_pos = cached_len + len(new_prompt_ids)
        # print(f"[DBG] generate: entering decoding loop. Reading initial logits...", flush=True)
        logits = output.logits[0, -1].cpu().numpy()
        # print(f"[DBG] generate: initial logits read.", flush=True)
        
        # Helper sampling
        def sample_logits(logits, temp, top_p):
            if temp <= 0.01:
                return int(np.argmax(logits))
            scaled = logits / temp
            # Softmax
            exp_logits = np.exp(scaled - np.max(scaled))
            probs = exp_logits / np.sum(exp_logits)
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cum_probs = np.cumsum(sorted_probs)
                cutoff = np.where(cum_probs > top_p)[0]
                if len(cutoff) > 0:
                    probs[sorted_indices[cutoff[0]+1:]] = 0.0
                    probs = probs / np.sum(probs)
            return int(np.random.choice(len(probs), p=probs))

        sfa_active = False
        for _ in range(max_new_tokens):
            # ── Repetition-loop detection (mirrors batch_engine.py Fix 2) ──────
            # Detect tight token-level loops every 10 new tokens.
            # On detection, widen the penalty window and boost the strength.
            # After 40 tokens without recovery, force-stop generation.
            _new_tokens = generated[len(prompt_ids):]  # tokens produced in this call
            _n_new = len(_new_tokens)
            _loop_detected = getattr(self, "_mlx_loop_detected", False)
            _loop_idx = getattr(self, "_mlx_loop_idx", None)

            if not _loop_detected and _n_new >= 30 and _n_new % 10 == 0:
                # 1. Exact match check for period K (10 to 120)
                _exact_loop = False
                for K in range(10, min(120, len(_new_tokens) // 2)):
                    if _new_tokens[-K:] == _new_tokens[-2*K:-K]:
                        _exact_loop = True
                        break
                
                # 2. Unique N-gram ratio check over a wider window (up to 256 tokens)
                _ratio_loop = False
                _window_size = min(256, len(_new_tokens))
                _window = _new_tokens[-_window_size:]
                _ng = 5
                if len(_window) >= _ng + 1:
                    _ngrams = [tuple(_window[i:i + _ng]) for i in range(len(_window) - _ng + 1)]
                    _counts = Counter(_ngrams)
                    _unique_ratio = len(_counts) / len(_ngrams)
                    if _unique_ratio < 0.40:
                        _ratio_loop = True
                
                if _exact_loop or _ratio_loop:
                    _loop_detected = True
                    self._mlx_loop_detected = True
                    self._mlx_loop_idx = _n_new
                    print(
                        f"[DiffKV MLX] WARNING: repetition loop detected at token "
                        f"{_n_new}. Escalating penalty window to 256 tokens and strength to 1.3x.",
                        file=sys.stderr
                    )

            if _loop_detected:
                if _loop_idx is None:
                    self._mlx_loop_idx = _n_new
                elif _n_new - _loop_idx >= 40:
                    print(
                        "[DiffKV MLX] WARNING: repetition loop persisted for 40 tokens "
                        "after detection \u2014 forcing EOS.",
                        file=sys.stderr
                    )
                    break

            # Repetition penalty (widened window when a loop is active)
            _pen_window = 256 if _loop_detected else 64
            _pen_val = max(repetition_penalty, 1.3) if _loop_detected else repetition_penalty
            if _pen_val != 1.0:
                # Numeric/separator exemption (2026-07-13): digits carry
                # semantics, not fluency — penalizing them corrupts faithful
                # reproduction of numeric content. Measured (CLI direct mode,
                # 12k paper + planted 6-row table, temp 0): at the default
                # 1.15 every digit is argmax-suppressed after the first row —
                # the reply is a header plus EMPTY cells and the model claims
                # the table "is not provided in your original document" —
                # while --repetition-penalty 1.0 reads the identical
                # compressed state 5-6/6. THIS loop is the live sampler on
                # macOS (the engine delegates decode here); mirrors
                # _filter_penalty_ids (batch_engine), the hf wrapper loop, and
                # rep_exempt_cache (native main.cpp). Suspended during loop
                # recovery so the escalated penalty still breaks digit loops.
                # DIFFKV_REP_PENALTY_PROTECT_NUMERIC=0 restores.
                _protect_numeric = (not _loop_detected and
                                    os.environ.get("DIFFKV_REP_PENALTY_PROTECT_NUMERIC", "1") == "1")
                # Table-line suspension (mirror of batch_engine._in_table_line):
                # while the current output line (plus the line above — row
                # starts count) is table-like, suspend the penalty entirely;
                # verbatim table rows can't survive ANY penalized token
                # (measured: empty '| | | |' cells with digits exempt but
                # glue penalized). Loop recovery overrides.
                if _protect_numeric and generated:
                    if not hasattr(self, "_rep_decode_strs"):
                        self._rep_decode_strs = {}
                    _seps = _nums = _nl = _n = 0
                    for _tid in reversed(generated):
                        _n += 1
                        if _n > 64:
                            break
                        _s = self._rep_decode_strs.get(_tid)
                        if _s is None:
                            _s = self._rep_decode_strs[_tid] = self.tokenizer.decode([_tid])
                        if "\n" in _s:
                            _nl += 1
                            if _nl >= 2:
                                break
                            continue
                        _sc = _s.strip()
                        if _sc in ("|", "&"):
                            _seps += 1
                            _nums += 1
                        elif any(c.isdigit() for c in _sc):
                            _nums += 1
                        if _seps >= 2 or _nums >= 3:
                            _pen_val = 1.0
                            break
                if not hasattr(self, "_rep_exempt_tokens"):
                    self._rep_exempt_tokens = {}
                for tok_id in set(generated[-_pen_window:]):
                    if _protect_numeric:
                        _ex = self._rep_exempt_tokens.get(tok_id)
                        if _ex is None:
                            _txt = self.tokenizer.decode([tok_id])
                            _ex = any(c.isdigit() for c in _txt) or _txt.strip() in ("|", "&")
                            self._rep_exempt_tokens[tok_id] = _ex
                        if _ex:
                            continue
                    if logits[tok_id] > 0:
                        logits[tok_id] /= _pen_val
                    else:
                        logits[tok_id] *= _pen_val
                        
            # Apply Factual Logit Bias
            srl_state = getattr(self.manager, "_session_srl", {}).get(session_id)
            if srl_state is not None:
                # Helper token set (needed for penalty and VSL masking below)
                from native_core.srl.factual_alignment import get_helper_token_ids
                helper_ids = get_helper_token_ids(self.tokenizer)

                # +7.0 factual token bias (raised from +3).
                # EXCLUDE tokens already emitted THIS generation: a flat per-token
                # bias re-boosts a value every step, so once "5198" is out it keeps
                # winning → "5198-5198-…" loops. Skipping emitted tokens lets the
                # +10 transition bias carry in-order progression and then release,
                # so a value/sequence is emitted once. (A repeated token that
                # legitimately recurs is still reachable via the transition bias.)
                if getattr(srl_state, "current_step_factual_tokens", None):
                    current_entity = getattr(srl_state, "current_entity_id", -1)
                    entity_ids = getattr(srl_state, "current_step_sequence_entity_ids", [])
                    is_prime_list = getattr(srl_state, "current_step_sequence_is_prime", [])
                    _emitted_gen = set(generated[len(prompt_ids):])

                    if current_entity != -1:
                        entity_factual_tokens = set()
                        for i, seq in enumerate(srl_state.current_step_factual_sequences):
                            seq_eid = entity_ids[i] if i < len(entity_ids) else -1
                            seq_is_prime = is_prime_list[i] if i < len(is_prime_list) else False
                            if seq_eid == -1 or seq_eid == current_entity or seq_is_prime:
                                entity_factual_tokens.update(seq)
                        for tok_id in entity_factual_tokens:
                            if tok_id < len(logits) and tok_id not in _emitted_gen:
                                logits[tok_id] += 7.0
                    else:
                        for tok_id in srl_state.current_step_factual_tokens:
                            if tok_id < len(logits) and tok_id not in _emitted_gen:
                                logits[tok_id] += 7.0

                # +7.0 VSL active-candidate boost
                active_candidates = getattr(srl_state, "vsl_active_candidates", [])
                if active_candidates:
                    for suffix in active_candidates:
                        if suffix and suffix[0] < len(logits):
                            logits[suffix[0]] += 7.0

                # -3.5 anti-hallucination penalty — threshold lowered 0.55→0.4,
                # magnitude raised -2.5→-3.5, active_candidates requirement removed.
                if (getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.4
                        and not getattr(srl_state, "dual_entity_mode", False)
                        and getattr(srl_state, "current_step_factual_tokens", None)):
                    factual_set = srl_state.current_step_factual_tokens
                    _vocab = len(logits)
                    _excl = np.array([t for t in list(factual_set) + list(helper_ids) if 0 <= t < _vocab], dtype=np.int64)
                    _penalty_mask = np.ones(_vocab, dtype=bool)
                    if len(_excl) > 0:
                        _penalty_mask[_excl] = False
                    logits[_penalty_mask] -= 3.5

                # +10.0 transition bias (raised from +4)
                last_token = generated[-1] if generated else None
                if last_token is not None and getattr(srl_state, "current_step_factual_sequences", None):
                    transition_candidates = set()
                    current_entity = getattr(srl_state, "current_entity_id", -1)
                    entity_ids = getattr(srl_state, "current_step_sequence_entity_ids", [])
                    for i, seq in enumerate(srl_state.current_step_factual_sequences):
                        seq_entity = entity_ids[i] if i < len(entity_ids) else -1
                        if current_entity != -1 and seq_entity != -1 and seq_entity != current_entity:
                            continue  # skip cross-entity transitions
                        for idx, tok in enumerate(seq[:-1]):
                            if tok == last_token:
                                transition_candidates.add(seq[idx + 1])
                    for tok_id in transition_candidates:
                        if tok_id < len(logits):
                            logits[tok_id] += 10.0

            # Apply Dynamic Temperature Scaling (Option 1)
            effective_temperature = temperature
            if srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55:
                max_sim = srl_state.current_step_max_similarity
                effective_temperature = temperature * (1.0 - max_sim * 0.95)

            # SFA threshold aligned to 0.55: at 0.3 almost every topical entry matches,
            # activating the VSL and forcing generation from a mixed-category token set.
            # At 0.55 only high-confidence, specific retrieval triggers the constraint.
            sfa_active = (
                srl_state is not None
                and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55
                and bool(getattr(srl_state, "current_step_factual_sequences", None))
            )

            # LM-VSL (Logit Masking) — guard against empty sequences; without this
            # get_allowed_tokens_vsl returns only helper words, locking generation.
            if sfa_active:
                from native_core.srl.factual_alignment import (
                    get_allowed_tokens_vsl, get_structural_helper_token_ids)
                structural_helper_ids = get_structural_helper_token_ids(self.tokenizer)
                allowed_ids = get_allowed_tokens_vsl(
                    srl_state, helper_ids,
                    structural_helper_ids=structural_helper_ids,
                    sfa_active=True,
                )
                mask = np.ones_like(logits, dtype=bool)
                mask[list(allowed_ids)] = False
                factual_toks = getattr(srl_state, "current_step_factual_tokens", None)
                if factual_toks:
                    valid_factual_toks = [t for t in factual_toks if 0 <= t < logits.shape[-1]]
                    mask[valid_factual_toks] = False
                
                max_sim = getattr(srl_state, "current_step_max_similarity", 0.0)
                if max_sim >= 0.70:
                    logits[mask] = -1e10   # hard: verbatim extraction mode
                else:
                    logits[mask] -= 7.0    # soft: guided but escapable

            next_id = sample_logits(logits, effective_temperature, top_p)

            if srl_state is not None and os.environ.get("DIFFKV_FACTUAL_DBG") == "1":
                print(f"[Python DEBUG] Step {_n_new} next_id_val={next_id} token={repr(self.tokenizer.decode([next_id]))} max_sim={getattr(srl_state, 'current_step_max_similarity', 0.0):.4f} sfa_active={sfa_active}", flush=True)

            # Strict Factual Alignment (SFA) State Update and Loop Check
            if srl_state is not None:
                from native_core.srl.factual_alignment import update_vsl_state, get_helper_token_ids
                helper_ids = get_helper_token_ids(self.tokenizer)
                update_vsl_state(next_id, srl_state, helper_ids)
                
                if sfa_active and getattr(srl_state, "vsl_consecutive_helpers", 0) >= 16:
                    uncertainty_suffix = " [uncertain: details missing in source]"
                    uncertainty_tokens = self.tokenizer.encode(uncertainty_suffix, add_special_tokens=False)
                    for t_id in uncertainty_tokens:
                        generated.append(t_id)
                        self.manager.register_prefill_tokens(session_id, torch.tensor([t_id], dtype=torch.long))
                    break

            generated.append(next_id)
            self.manager.register_prefill_tokens(session_id, torch.tensor([next_id], dtype=torch.long))

            if srl_state is not None and hasattr(srl_state, "save_step_state"):
                srl_state.save_step_state(len(generated))

            # Factual Early Stopping (Option 2 Extension)
            stop_generation = False
            if max_new_tokens < 64 and srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.5:
                if getattr(srl_state, "current_step_factual_sequences", None):
                    for seq in srl_state.current_step_factual_sequences:
                        if len(seq) >= 5 and len(generated) >= len(seq):
                            if generated[-len(seq):] == list(seq):
                                stop_generation = True
                                break
            if stop_generation:
                break
            
            if next_id in self.stop_token_ids:
                break
                
            # print(f"[DBG] generate: decode step {_n_new+1}/{max_new_tokens} starting...", flush=True)
            input_ids = torch.tensor([[next_id]], dtype=torch.long)
            pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long)
            
            output = self.model(input_ids, pos_tensor)
            logits = output.logits[0, -1].cpu().numpy()
            # print(f"[DBG] generate: decode step {_n_new+1}/{max_new_tokens} completed.", flush=True)
            
            cur_pos += 1

        # Clear loop detection state for this session after generation completes
        self._mlx_loop_detected = False
        self._mlx_loop_idx = None
        self._session_token_ids[session_id] = generated
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return _normalize_references(decoded)

    def rollback_session(self, session_id: str, target_len: int, clear_srl: bool = False):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.rollback_session(session_id, target_len, clear_srl=clear_srl)
        if session_id in self._session_token_ids:
            self._session_token_ids[session_id] = self._session_token_ids[session_id][:target_len]

    def clone_session(self, src_sid: str, dst_sid: str):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.clone_session(src_sid, dst_sid)
        if src_sid in self._session_token_ids:
            self._session_token_ids[dst_sid] = list(self._session_token_ids[src_sid])

    def clear_session(self, session_id: str):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.clear_session(session_id)
        self._session_token_ids.pop(session_id, None)
