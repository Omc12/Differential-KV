from typing import Dict, List

class ConcurrencyFairnessController:
    """
    PHASE 7.5C: Concurrency Fairness Controller
    Ensures that no single user starves others during high retrieval 
    contention by implementing a deficit round-robin or similar fairness scheme.
    """
    def __init__(self, quantum: int = 10):
        self.user_quotas: Dict[int, int] = {} # user_id -> remaining_quantum
        self.quantum = quantum
        self.active_users: List[int] = []

    def register_request(self, user_id: int):
        """Signals a request from a user."""
        if user_id not in self.user_quotas:
            self.user_quotas[user_id] = self.quantum
            self.active_users.append(user_id)

    def can_consume(self, user_id: int) -> bool:
        """Checks if a user has remaining quota for this round."""
        if user_id not in self.user_quotas:
            return True
        return self.user_quotas[user_id] > 0

    def consume_resource(self, user_id: int, cost: int = 1):
        """Decrements a user's quota after a resource-heavy operation."""
        if user_id in self.user_quotas:
            self.user_quotas[user_id] -= cost

    def reset_quotas(self):
        """Refills all active user quotas for the next round."""
        for user_id in self.user_quotas:
            self.user_quotas[user_id] = self.quantum
            
    def get_fair_user_list(self) -> List[int]:
        """Returns users who still have quota."""
        return [uid for uid, quota in self.user_quotas.items() if quota > 0]
