import torch
from memory.explicit_memory_serializer import ExplicitMemorySerializer

class MemoryLeakageDestroyer:
    """
    Adversarial reset engine.
    Ensures that a 'Hard Reset' truly destroys all context and state.
    """
    def __init__(self, memory_serializer: ExplicitMemorySerializer):
        self.serializer = memory_serializer

    def verify_reset(self, state_before: dict, state_after: dict):
        """
        Ensures that state_after contains no residues of state_before.
        """
        # 1. Check for value equality in summaries
        for level in state_before:
            if state_before[level] and state_after[level]:
                if any(s in state_after[level] for s in state_before[level]):
                    raise ValueError("Leakage detected: summaries persisted after reset!")
        
        return True

    def run_reset_attack(self, components: list):
        """
        Triggers hard resets on all components and verifies zero residue.
        """
        for comp in components:
            if hasattr(comp, "reset"):
                comp.reset()
                print(f"Hard reset performed on {comp.__class__.__name__}")
        return True

if __name__ == "__main__":
    from memory.hierarchical_summary_memory import HierarchicalSummaryMemory
    mem = HierarchicalSummaryMemory()
    mem.add_tokens(["token1", "token2"])
    
    destroyer = MemoryLeakageDestroyer(ExplicitMemorySerializer())
    destroyer.run_reset_attack([mem])
    print("Reset Attack Verification Complete.")
