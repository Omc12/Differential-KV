from typing import List, Dict

class HotsetContentionMonitor:
    """
    Detects VRAM hotset collisions where multiple users compete 
    for the same physical memory blocks.
    """
    def __init__(self):
        self.active_blocks = {} # block_id -> user_count

    def record_access(self, block_ids: List[int], user_id: str):
        collisions = 0
        for bid in block_ids:
            if bid in self.active_blocks and self.active_blocks[bid] > 0:
                collisions += 1
            self.active_blocks[bid] = self.active_blocks.get(bid, 0) + 1
        return collisions

    def release_access(self, block_ids: List[int], user_id: str):
        for bid in block_ids:
            if bid in self.active_blocks:
                self.active_blocks[bid] -= 1
                if self.active_blocks[bid] <= 0:
                    del self.active_blocks[bid]
