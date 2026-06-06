from abc import ABC, abstractmethod
from typing import List, Any, Optional

class DraftModelPlugin(ABC):
    """
    Abstract Base Class for Draft Model Plugins used in speculative decoding.
    
    Licensees can implement this class to integrate any custom draft model
    (e.g., Jacobi/lookahead drafts, eagle, custom small transformers, MLX-powered drafts)
    with the DiffKV continuous batch engine.
    """
    
    @abstractmethod
    def init_session(self, session_id: str, prefill_len: int) -> None:
        """Initialize draft cache for a new/existing session."""
        pass
        
    @abstractmethod
    def generate_candidates(
        self,
        session_id: str,
        last_token: int,
        prefix_len: int,
        num_candidates: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[int]:
        """
        Generate num_candidates draft token candidates.
        Returns a list of token IDs.
        """
        pass
        
    @abstractmethod
    def rollback_session(self, session_id: str, target_len: int) -> None:
        """Roll back draft cache to target_len tokens."""
        pass
        
    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Release any draft cache resources for the session."""
        pass
