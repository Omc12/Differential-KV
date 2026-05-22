"""
patch_kv_manager.py
Removes all dead adaptive-rank code from kv_runtime_manager.py now that
RESEARCH_PROTOTYPES has been archived.
"""
import re, sys

PATH = r"ACTIVE_RUNTIME\native_core\kv_runtime_manager.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

original_len = len(src)

# ── 1. Remove RESEARCH_PROTOTYPES import block ────────────────────────────────
# Lines 27-42: comment + try/except block
pattern1 = re.compile(
    r"# \u2500\u2500 Salvaged P1: Adaptive rank selector \u2500+\n"
    r"try:.*?_AdaptiveRankSelector = None\n\n",
    re.DOTALL
)
src, n1 = pattern1.subn("", src)

# ── 2. Remove adaptive_rank init block inside __init__ ────────────────────────
pattern2 = re.compile(
    r"        # \u2500\u2500 Salvaged P1: Adaptive Rank Selector \u2500+\n"
    r"        self\._adaptive_rank = adaptive_rank and _ADAPTIVE_RANK\n"
    r"        if self\._adaptive_rank:\n"
    r"            self\._rank_selector = _AdaptiveRankSelector\(\n"
    r"                rank_buckets=\[4, 8, 16, 32\],\n"
    r'                method="variance",\n'
    r"            \)\n"
    r"        else:\n"
    r"            self\._rank_selector = None\n\n",
    re.DOTALL
)
src, n2 = pattern2.subn("", src)

# ── 3. Fix docstring of _compress_block_sync ──────────────────────────────────
old_docstring = (
    '    def _compress_block_sync(self, block: KVBlock,\n'
    '                             k: torch.Tensor, v: torch.Tensor):\n'
    '        """Synchronous SVD compression (used by AsyncCompressor worker).\n'
    '        \n'
    '        Uses AdaptiveRankSelector (salvaged from RESEARCH_PROTOTYPES) when available\n'
    '        to pick the minimal rank that preserves 95%+ KV delta variance.\n'
    '        Falls back to self.rank when adaptive selector is unavailable.\n'
    '        """\n'
)
new_docstring = (
    '    def _compress_block_sync(self, block: KVBlock,\n'
    '                             k: torch.Tensor, v: torch.Tensor):\n'
    '        """Synchronous SVD compression (used by AsyncCompressor worker).\n'
    '        Fixed rank -- simple, stable, predictable.\n'
    '        """\n'
)
n3 = 0
if old_docstring in src:
    src = src.replace(old_docstring, new_docstring, 1)
    n3 = 1

# ── 4. Remove adaptive rank selection branch inside _compress_block_sync ──────
pattern4 = re.compile(
    r"        # \u2500\u2500 Adaptive rank selection \(P1 salvage\) \u2500+\n"
    r"        if self\._adaptive_rank and self\._rank_selector is not None:\n"
    r"            rank = self\._rank_selector\.select_rank\(deltas\)\n"
    r"        else:\n"
    r"            rank = self\.rank\n\n",
)
src, n4 = pattern4.subn("        rank = self.rank\n", src)

# ── 5. Fix rank initializer comment ───────────────────────────────────────────
n5 = 0
old5 = "        self.rank                 = 8    # fallback fixed rank"
new5 = "        self.rank                 = rank  # fixed, set at construction"
if old5 in src:
    src = src.replace(old5, new5, 1)
    n5 = 1

# ── 6. Remove stale comment on dense_recency_blocks ──────────────────────────
n6 = 0
old6 = "        self.dense_recency_blocks = 1    # Phase 24.5: reduced from 2 to 1"
new6 = "        self.dense_recency_blocks = 1"
if old6 in src:
    src = src.replace(old6, new6, 1)
    n6 = 1

# ── 7. Remove adaptive_rank parameter from __init__ signature ─────────────────
n7 = 0
old7 = "        adaptive_rank:       bool  = True,\n        streaming_ingest:    bool  = True,   # Phase 24.5: enable sparse-first ingest\n        micro_block_size:    int   = 16,     # Phase 24.5: compress every N tokens\n"
new7 = "        streaming_ingest:    bool  = True,\n        micro_block_size:    int   = 16,\n        rank:                int   = 8,\n"
if old7 in src:
    src = src.replace(old7, new7, 1)
    n7 = 1

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print(f"Patch complete:")
print(f"  [1] Removed RESEARCH_PROTOTYPES import block: {n1} match(es)")
print(f"  [2] Removed adaptive_rank init block:         {n2} match(es)")
print(f"  [3] Fixed _compress_block_sync docstring:     {n3} match(es)")
print(f"  [4] Removed adaptive rank branch:             {n4} match(es)")
print(f"  [5] Fixed rank initializer comment:           {n5} match(es)")
print(f"  [6] Removed Phase 24.5 comment:               {n6} match(es)")
print(f"  [7] Removed adaptive_rank param:              {n7} match(es)")
print(f"  File size: {original_len} -> {len(src)} bytes")
