import torch
import hashlib

class HiddenStateAuditor:
    """
    Audits hidden states by hashing them to detect accidental persistence or reuse.
    """
    def __init__(self):
        self.hashes = {}

    def audit(self, hidden_states, run_id, step):
        """
        hidden_states: torch.Tensor
        run_id: str
        step: int
        """
        # Detach and move to CPU for hashing to avoid CUDA state interference
        data = hidden_states.detach().cpu().numpy().tobytes()
        h = hashlib.sha256(data).hexdigest()
        
        # Check if this hash has been seen in a DIFFERENT run at the same step
        # (Very unlikely for a truly fresh run with randomized seeds)
        if step in self.hashes:
            for prev_run_id, prev_hash in self.hashes[step].items():
                if prev_run_id != run_id and prev_hash == h:
                    print(f"CRITICAL ERROR: Hidden state collision detected between {run_id} and {prev_run_id} at step {step}!")
                    return False
        
        if step not in self.hashes:
            self.hashes[step] = {}
        self.hashes[step][run_id] = h
        return True

class CacheContaminationDetector:
    """
    Monitors VRAM and system memory for unexpected growth or leftover allocations.
    """
    def __init__(self):
        self.initial_vram = 0
        if torch.cuda.is_available():
            self.initial_vram = torch.cuda.memory_allocated()

    def check(self):
        if torch.cuda.is_available():
            current_vram = torch.cuda.memory_allocated()
            leak = current_vram - self.initial_vram
            if leak > 0:
                print(f"WARNING: VRAM Leak detected: {leak / 1024**2:.2f} MB")
                return False
        return True
