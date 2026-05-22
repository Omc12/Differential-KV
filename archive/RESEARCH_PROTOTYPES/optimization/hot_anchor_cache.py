class HotAnchorCache:
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.cache = {}
        self.access_counts = {}

    def get(self, anchor_id):
        if anchor_id in self.cache:
            self.access_counts[anchor_id] += 1
            return self.cache[anchor_id]
        return None

    def put(self, anchor_id, data):
        if len(self.cache) >= self.capacity:
            # LFU Eviction
            lfu_key = min(self.access_counts, key=self.access_counts.get)
            del self.cache[lfu_key]
            del self.access_counts[lfu_key]
            
        self.cache[anchor_id] = data
        self.access_counts[anchor_id] = 1
