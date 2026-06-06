import os
import torch
import torch.nn.functional as F
from typing import List, Any
from .base import DraftModelPlugin

class SpeculativeDecodingPlugin:
    """
    Plugin that implements speculative decoding verification logic,
    consuming candidates from a DraftModelPlugin instance.
    """
    def __init__(self, main_wrapper: Any, draft_plugin: DraftModelPlugin, num_candidates: int = 4):
        self.main_wrapper = main_wrapper
        self.draft_plugin = draft_plugin
        self.num_candidates = num_candidates
        
    def run_step(self, req: Any, step_start: float, sample_fn: Any, emit_fn: Any) -> List[int]:
        device = self.main_wrapper.device
        session_id = req.session_id
        
        prefix_len = req.total_seq_len
        last_token = req.generated_ids[-1] if req.generated_ids else req.prompt_ids[-1]
        
        # 1. Initialize draft session
        self.draft_plugin.init_session(session_id, prefix_len)
        
        # 2. Generate candidates from draft plugin
        candidate_tokens = self.draft_plugin.generate_candidates(
            session_id=session_id,
            last_token=last_token,
            prefix_len=prefix_len,
            num_candidates=self.num_candidates,
            temperature=req.temperature,
        )
        
        if not candidate_tokens:
            # Fallback to standard eager decoding
            return []
            
        # 3. Main Model: Run parallel verification pass
        verification_input = [last_token] + candidate_tokens
        verification_input_tensor = torch.tensor([verification_input], dtype=torch.long, device=device)
        verification_position_tensor = torch.arange(
            prefix_len - 1, prefix_len - 1 + len(verification_input),
            dtype=torch.long, device=device
        ).unsqueeze(0)
        
        # Inject main session ID
        self.main_wrapper.model._diffkv_session_ids = [session_id]
        
        self.main_wrapper.model._disable_lm_head_slicing = True
        try:
            with torch.no_grad():
                out = self.main_wrapper.model(
                    input_ids=verification_input_tensor,
                    position_ids=verification_position_tensor,
                    use_cache=True
                )
        finally:
            self.main_wrapper.model._disable_lm_head_slicing = False
            
        main_logits = out.logits[0]  # [len(verification_input), vocab_size]
        
        # 4. Verification Loop
        accepted_tokens = []
        correction_token = None
        temperature = req.temperature
        
        for i in range(len(candidate_tokens)):
            target_token = candidate_tokens[i]
            pred_logits = main_logits[i]
            
            # Greedy verification for temperature=0.0
            if temperature == 0.0:
                pred_token = pred_logits.argmax().item()
                if pred_token == target_token:
                    accepted_tokens.append(target_token)
                    if target_token in self.main_wrapper.stop_token_ids or target_token == self.main_wrapper.tokenizer.eos_token_id:
                        break
                else:
                    correction_token = pred_token
                    break
            else:
                # Sample-based verification (greedy fallback for plugin integration)
                pred_token = pred_logits.argmax().item()
                if pred_token == target_token:
                    accepted_tokens.append(target_token)
                    if target_token in self.main_wrapper.stop_token_ids or target_token == self.main_wrapper.tokenizer.eos_token_id:
                        break
                else:
                    correction_token = pred_token
                    break
                    
        # If all candidates accepted, sample correction token from the last logit
        if correction_token is None and (not accepted_tokens or (accepted_tokens[-1] not in self.main_wrapper.stop_token_ids and accepted_tokens[-1] != self.main_wrapper.tokenizer.eos_token_id)):
            last_logits = main_logits[-1]
            if temperature == 0.0:
                correction_token = last_logits.argmax().item()
            else:
                probs = F.softmax(last_logits / max(temperature, 1e-5), dim=-1)
                correction_token = torch.multinomial(probs, 1).item()
                
        # 5. Rollback caches to match the accepted length
        target_len = prefix_len + len(accepted_tokens)
        
        if hasattr(self.main_wrapper.manager, "rollback_session"):
            self.main_wrapper.manager.rollback_session(session_id, target_len)
            
        self.draft_plugin.rollback_session(session_id, target_len)
        
        # 6. Emit accepted tokens + correction token
        new_tokens = accepted_tokens + ([correction_token] if correction_token is not None else [])
        for token_id in new_tokens:
            req.generated_ids.append(token_id)
            emit_fn(req, token_id, step_start)
            
        return new_tokens
