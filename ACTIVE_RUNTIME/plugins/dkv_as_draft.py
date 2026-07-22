import torch
from typing import List, Any
from .base import DraftModelPlugin

class DKVAsDraftPlugin(DraftModelPlugin):
    """
    Wraps a small DKV model instance to act as a draft model plugin
    for a larger main model.
    """
    def __init__(self, draft_wrapper: Any):
        self.draft_wrapper = draft_wrapper
        
    def init_session(self, session_id: str, prefill_len: int) -> None:
        draft_session_id = session_id + "_draft"
        if hasattr(self.draft_wrapper.manager, "init_session"):
            self.draft_wrapper.manager.init_session(draft_session_id, prefill_len=prefill_len)
            
    def generate_candidates(
        self,
        session_id: str,
        last_token: int,
        prefix_len: int,
        num_candidates: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[int]:
        draft_session_id = session_id + "_draft"
        candidate_tokens = []
        current_input = last_token
        
        for step in range(num_candidates):
            input_ids = torch.tensor([[current_input]], dtype=torch.long, device=self.draft_wrapper.device)
            position_ids = torch.tensor([[prefix_len - 1 + step]], dtype=torch.long, device=self.draft_wrapper.device)
            
            # Inject session ID for draft attention routing
            self.draft_wrapper.model._dkv_session_ids = [draft_session_id]
            
            with torch.no_grad():
                out = self.draft_wrapper.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True
                )
            logits = out.logits[0, -1, :]
            
            # Greedy decoding for draft candidates
            next_draft_token = logits.argmax().item()
            candidate_tokens.append(next_draft_token)
            current_input = next_draft_token
            
        return candidate_tokens
        
    def rollback_session(self, session_id: str, target_len: int) -> None:
        draft_session_id = session_id + "_draft"
        if hasattr(self.draft_wrapper.manager, "rollback_session"):
            self.draft_wrapper.manager.rollback_session(draft_session_id, target_len)
            
    def clear_session(self, session_id: str) -> None:
        draft_session_id = session_id + "_draft"
        if hasattr(self.draft_wrapper.manager, "clear_session"):
            self.draft_wrapper.manager.clear_session(draft_session_id)
