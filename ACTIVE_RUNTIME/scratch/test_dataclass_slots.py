from dataclasses import dataclass
from typing import Optional, Any

@dataclass(slots=True)
class TestBlock:
    anchor_idx: int
    _U: Optional[str] = None
    pool: Any = None
    pool_idx: Optional[int] = None

    @property
    def U(self):
        if self._U is not None:
            return self._U
        if self.pool_idx is not None and self.pool is not None:
            return f"from pool slot {self.pool_idx}"
        return None

    @U.setter
    def U(self, val):
        self._U = val

tb = TestBlock(anchor_idx=0)
print("Initial:", tb.U)
tb.U = "explicit"
print("After explicit set:", tb.U)
tb._U = None
tb.pool = "dummy_pool"
tb.pool_idx = 42
print("After pool link:", tb.U)
