class AnchorPriorityAllocator:
    """
    PHASE 18.9D: Anchor Priority Allocator.
    Assigns priorities to different types of structural anchors.
    """
    def __init__(self):
        self.priorities = {
            "ROLE_BOUNDARY": 1.0,
            "SECTION_HEADER": 0.9,
            "NEWLINE": 0.5,
            "DEFAULT": 0.2
        }

    def get_priority(self, anchor_type):
        return self.priorities.get(anchor_type, self.priorities["DEFAULT"])
