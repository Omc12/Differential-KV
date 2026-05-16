import torch
from typing import Optional, Tuple, Any

class DenseReferenceComparator:
    """
    SGC Phase 39.1 RESET: Dense Reference Comparator.
    Orchestrates periodic full-dense passes for semantic validation.
    """
    def __init__(self, sampling_interval: int = 10):
        self.sampling_interval = sampling_interval
        self.step_counter = 0

    def should_run_reference(self) -> bool:
        """Determines if the current step requires a dense reference pass."""
        self.step_counter += 1
        return (self.step_counter % self.sampling_interval) == 0

    def run_dense_pass(self, model: torch.nn.Module, input_ids: torch.Tensor, past_key_values: Optional[Any] = None) -> torch.Tensor:
        """
        Executes a single forward pass in forced-dense mode.
        NOTE: This expects the model to respect a 'force_dense' flag in the config or context.
        """
        # We assume the caller handles the 'force_dense' context management
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            return outputs.logits
