class BoundedRecursionController:
    """
    Manages recursion depth and explicit recursion state.
    Prevents uncontrolled recursive state growth and hidden persistence.
    """
    def __init__(self, max_recursion_depth: int = 3):
        self.max_recursion_depth = max_recursion_depth
        self.current_depth = 0
        self.call_stack = []

    def enter_recursion(self, scope_name: str, state_summary: str):
        """Enters a new level of recursion."""
        if self.current_depth >= self.max_recursion_depth:
            raise RecursionError(f"Max recursion depth {self.max_recursion_depth} reached in {scope_name}.")
        
        self.call_stack.append({
            "scope": scope_name,
            "depth": self.current_depth,
            "state_summary": state_summary
        })
        self.current_depth += 1

    def exit_recursion(self):
        """Exits the current level of recursion."""
        if self.current_depth > 0:
            self.call_stack.pop()
            self.current_depth -= 1

    def get_stack_trace(self) -> str:
        """Returns a string representation of the explicit recursion stack."""
        return " -> ".join([f"[{s['depth']}] {s['scope']}" for s in self.call_stack])

    def reset(self):
        """Resets the recursion controller."""
        self.current_depth = 0
        self.call_stack = []
