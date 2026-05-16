
import torch
from typing import Dict, List, Optional

class SymbolicAnchorIndex:
    """
    PPOSAH Phase 20.6A: Exact Symbolic Anchor Index.
    Replaces fuzzy alignment with absolute-index lookup.
    """
    def __init__(self):
        self.index: Dict[int, int] = {} # absolute_index -> token_id
        self.spans: List[tuple] = [] # list of (start, end)
        self.active_root_start: int = -1

    def add_span(self, start: int, end: int, tokens: torch.Tensor):
        self.spans.append((start, end))
        token_ids = tokens.flatten().tolist()
        for i, tid in enumerate(token_ids):
            self.index[start + i] = tid

    def get_expected_token(self, absolute_pos: int) -> Optional[int]:
        return self.index.get(absolute_pos)

    def find_span_for_index(self, index: int) -> Optional[tuple]:
        for s, e in self.spans:
            if s <= index <= e:
                return (s, e)
        return None

    def reset(self):
        self.index = {}
        self.spans = []
        self.active_root_start = -1

class ConfirmedSpanRootTracker:
    """
    PPOSAH Phase 20.6A: Confirmed Span Root Tracker.
    Tracks which symbolic span the decoder is currently 'locked' onto.
    """
    def __init__(self):
        self.root_start: int = -1
        self.current_offset: int = 0
        self.is_confirmed: bool = False

    def confirm(self, root_start: int):
        self.root_start = root_start
        self.current_offset = 0
        self.is_confirmed = True

    def step(self):
        if self.is_confirmed:
            self.current_offset += 1

    def reset(self):
        self.root_start = -1
        self.current_offset = 0
        self.is_confirmed = False
