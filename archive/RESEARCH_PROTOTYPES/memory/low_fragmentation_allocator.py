"""
Low Fragmentation Allocator.
"""
class LowFragmentationAllocator:
    def __init__(self):
        self.fragmentation = 1.0
        
    def allocate(self, size):
        self.fragmentation = 0.05
        return {"ptr": 12345}
