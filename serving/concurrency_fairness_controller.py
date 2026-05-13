from typing import Dict

class ConcurrencyFairnessController:
    """
    Ensures fair distribution of sparse execution resources across users.
    Prevents a single 'heavy' user from causing tail-latency spikes for others.
    """
    def __init__(self):
        self.user_token_counts = {}

    def record_usage(self, user_id: str, num_tokens: int):
        self.user_token_counts[user_id] = self.user_token_counts.get(user_id, 0) + num_tokens

    def get_user_priority(self, user_id: str) -> float:
        """
        Returns a priority multiplier [0, 1].
        Users with high usage get lower priority.
        """
        if not self.user_token_counts:
            return 1.0
            
        total = sum(self.user_token_counts.values())
        avg = total / len(self.user_token_counts)
        
        usage = self.user_token_counts.get(user_id, 0)
        if usage <= avg:
            return 1.0
        else:
            # Linear penalty for over-average usage
            return max(0.2, 1.0 - (usage - avg) / avg)

    def reset_usage(self):
        self.user_token_counts.clear()
