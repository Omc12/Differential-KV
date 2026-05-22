class ContextPressureBalancer:
    def __init__(self, max_memory_budget):
        self.max_memory_budget = max_memory_budget
        self.current_pressure = 0

    def balance(self, num_active_streams, context_lengths):
        # Adjust sparsity levels and eviction policies based on global memory pressure
        total_projected = sum(context_lengths)
        if total_projected > self.max_memory_budget:
            # Increase sparsity constraints
            return "high_pressure"
        return "normal"
