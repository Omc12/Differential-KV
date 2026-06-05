"""
native_core/srl/inverted_index.py

Lexical inverted token index: token_id → list of pool slot IDs.

This catches blocks relevant to the current query via exact token match,
complementing the semantic ANN search (which uses meaning, not spelling).

Example: if the user asks about "GPT-4", the inverted index returns all
blocks that mentioned "GPT" or "4" during the conversation, even if those
blocks' semantic descriptors didn't cluster near the query descriptor.

Build:
  - Runs over full prompt token IDs (CPU, Python loop)
  - Takes top-N most frequent non-stop tokens per block
  - ~0.05s for a 25K-token prompt

Storage: ~240KB for 5000 unique terms × avg 12 blocks per term
"""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch


@dataclass
class InvertedTokenIndex:
    """
    Maps token_id → sorted list of pool_slot_ids that contain that token.
    CPU-resident (dict lookup, no GPU needed).
    """
    index:           Dict[int, List[int]]   # token_id → [pool_slot_id, ...]
    important_vocab: Set[int]               # token IDs indexed (excludes stop words)


def build_inverted_index(
    token_ids:        torch.Tensor,     # [seq_len] full prompt token IDs (CPU)
    slot_ids:         List[int],        # pool slot IDs in chronological order
    block_size:       int,              # tokens per block (including anchor)
    stop_token_ids:   Set[int],         # precomputed stop word token IDs
    top_n_per_block:  int = 20,         # most important tokens to index per block
) -> InvertedTokenIndex:
    """
    Build a lexical inverted index from prompt token IDs.

    Each block covers `block_size` tokens (including anchor). The index maps
    important token IDs to the list of pool slots that contain them.

    Args:
        token_ids:       Full prompt token ID sequence, CPU tensor [seq_len]
        slot_ids:        Pool slot IDs in chronological order (len = N_blocks)
        block_size:      Tokens per block (anchor + active, e.g. 256+1=257)
        stop_token_ids:  Token IDs to exclude (stop words, punctuation, etc.)
        top_n_per_block: Max distinct tokens to index per block

    Returns:
        InvertedTokenIndex ready for decode-time lookup
    """
    token_list = token_ids.tolist()
    seq_len    = len(token_list)
    N_blocks   = len(slot_ids)

    index: Dict[int, List[int]] = defaultdict(list)

    for i, slot in enumerate(slot_ids):
        # Block i covers tokens [i*block_size, (i+1)*block_size)
        start = i * block_size
        end   = min(start + block_size, seq_len)
        if start >= seq_len:
            break

        block_toks = token_list[start:end]

        # Count non-stop token frequencies in this block
        freq = Counter(t for t in block_toks if t not in stop_token_ids)

        # Take top_n most frequent
        for tok, _ in freq.most_common(top_n_per_block):
            index[tok].append(slot)

    # Deduplicate (a slot shouldn't appear twice for the same token)
    deduped = {tok: list(dict.fromkeys(slots)) for tok, slots in index.items()}

    return InvertedTokenIndex(
        index           = deduped,
        important_vocab = set(deduped.keys()),
    )


def lookup(
    inv_index:       InvertedTokenIndex,
    query_token_ids: List[int],
) -> Set[int]:
    """
    Return the set of pool slot IDs that contain any of the query tokens.

    Args:
        inv_index:       The InvertedTokenIndex for this session
        query_token_ids: List of token IDs to look up (recent generated tokens)

    Returns:
        Set of pool slot IDs (unordered)
    """
    result: Set[int] = set()
    vocab = inv_index.important_vocab
    idx   = inv_index.index
    for tok in query_token_ids:
        if tok in vocab and tok in idx:
            result.update(idx[tok])
    return result
