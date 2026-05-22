from typing import Callable, Any, Dict
from memory.execution_state_memory import ExecutionStateMemory

class FakeContinuityDestroyer:
    """
    Tests reset robustness by attempting to inject fake continuity signals
    and verifying that a hard reset correctly wipes them.
    """
    def __init__(self, memory: ExecutionStateMemory):
        self.memory = memory

    def inject_fake_continuity(self, key: str, data: Any):
        """Injects fake continuity data into memory."""
        self.memory.update_state(f"FAKE_{key}", data)

    def verify_wipe(self) -> bool:
        """
        Performs a hard reset and verifies that no keys (including fake ones) remain.
        Returns True if the wipe was successful.
        """
        self.memory.clear()
        # After clear, the memory.state should be empty and file should not exist
        return len(self.memory.state) == 0

    def test_continuity_leak(self, operation_fn: Callable) -> bool:
        """
        Runs an operation, then checks if any state leaked into a fresh instance.
        """
        # 1. Run operation
        operation_fn()
        
        # 2. Create fresh instance of memory with same path
        fresh_memory = ExecutionStateMemory(self.memory.storage_path)
        fresh_memory.load()
        
        # 3. If fresh_memory has data without explicit restoration, it's a leak
        return len(fresh_memory.state) > 0
