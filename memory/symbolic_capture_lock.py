import torch

class SymbolicCaptureLock:
    """
    PHASE 20.2: Symbolic Capture Lock & Contiguous Payload Preservation.
    Once a symbolic 'lead-in' (keyword/anchor) is detected via salience, 
    this module 'locks' preservation for the next N tokens to ensure 
    the randomized payload is preserved regardless of its individual salience.
    """
    def __init__(self, lock_duration: int = 16, trigger_threshold: float = 2.0):
        self.lock_duration = lock_duration
        self.trigger_threshold = trigger_threshold
        self.lock_counters = None # [batch_size]

    def update_locks(self, salience_scores: torch.Tensor) -> torch.Tensor:
        """
        Updates lock states based on new salience scores.
        Returns:
            lock_mask: Binary mask of currently locked tokens.
        """
        batch, q_len = salience_scores.shape
        if self.lock_counters is None:
            self.lock_counters = torch.zeros(batch, device=salience_scores.device, dtype=torch.long)
            
        lock_mask = torch.zeros_like(salience_scores, dtype=torch.bool)
        
        for t in range(q_len):
            # Check for new triggers
            triggers = salience_scores[:, t] > self.trigger_threshold
            self.lock_counters[triggers] = self.lock_duration
            
            # Apply lock
            lock_mask[:, t] = self.lock_counters > 0
            
            # Decrement counters
            self.lock_counters = (self.lock_counters - 1).clamp(min=0)
            
        return lock_mask

    def reset(self):
        self.lock_counters = None
