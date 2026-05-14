class ReinforcementBudgetController:
    """
    PHASE 18.9D: Reinforcement Budget Controller.
    Ensures that anchor-relative reinforcement does not exceed the fidelity budget.
    Supports priority levels for critical structural markers.
    """
    def __init__(self, max_reinforced_tokens=1024):
        self.max_tokens = max_reinforced_tokens
        self.current_count = 0
        self.critical_count = 0

    def request_protection(self, num_tokens, priority=1):
        """
        Request protection for a set of tokens.
        Priority 1: Normal reinforcement (limited by budget)
        Priority 2: Critical structural markers (allowed slight overflow)
        """
        if priority >= 2:
            self.critical_count += num_tokens
            return True
            
        if self.current_count + num_tokens <= self.max_tokens:
            self.current_count += num_tokens
            return True
        return False

    def get_utilization(self):
        return (self.current_count + self.critical_count) / self.max_tokens if self.max_tokens > 0 else 1.0

    def reset(self):
        self.current_count = 0
        self.critical_count = 0
