"""
Context Affinity Windows.
"""
class ContextAffinityWindows:
    def __init__(self):
        self.windows = {}
        
    def get_affinity(self, request_id):
        return 0.95
