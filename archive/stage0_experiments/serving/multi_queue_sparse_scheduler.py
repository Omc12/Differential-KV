import time
from typing import List, Dict, Any

class MultiQueueSparseScheduler:
    """
    Manages concurrent users with independent sparse budgets and fairness control.
    """
    def __init__(self, max_concurrency: int = 8):
        self.max_concurrency = max_concurrency
        self.queues = {i: [] for i in range(max_concurrency)}
        self.active_users = set()
        self.start_times = {}

    def add_request(self, user_id: int, request_id: str):
        if user_id not in self.active_users:
            self.active_users.add(user_id)
            self.start_times[user_id] = time.perf_counter()
        self.queues[user_id % self.max_concurrency].append(request_id)

    def get_batch_config(self, user_id: int):
        # Dynamically adjust sparse budget based on concurrency
        # To maintain fairness and prevent OOM
        concurrency = len(self.active_users)
        base_budget = 0.1
        adjusted_budget = base_budget / (concurrency ** 0.5)
        return {"sparse_budget": max(0.01, adjusted_budget)}

    def complete_request(self, user_id: int, request_id: str):
        if request_id in self.queues[user_id % self.max_concurrency]:
            self.queues[user_id % self.max_concurrency].remove(request_id)
        if not self.queues[user_id % self.max_concurrency]:
            if user_id in self.active_users:
                self.active_users.remove(user_id)

    def get_scaling_efficiency(self, single_user_tps: float, multi_user_tps: float):
        # Efficiency = (Multi-user TPS / (Single-user TPS * Num Users))
        num_users = len(self.active_users)
        if num_users == 0 or single_user_tps == 0:
            return 0.0
        return multi_user_tps / (single_user_tps * num_users)
