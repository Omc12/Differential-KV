import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

class SpeculativeDecoder:
    def __init__(self, main_wrapper, draft_wrapper, num_candidates: int = 4):
        self.main_wrapper = main_wrapper
        self.draft_wrapper = draft_wrapper
        self.num_candidates = num_candidates
        
    def run_step(self, req: Any, step_start: float, sample_fn: Any, emit_fn: Any) -> List[int]:
        """
        Runs a single speculative decoding step for a single request.
        Generates candidates with the draft model, verifies them with the main model,
        rolls back both caches, and emits the accepted tokens + correction token.
        
        Returns the list of new token IDs generated/accepted in this step.
        """
        device = self.main_wrapper.device
        session_id = req.session_id
        draft_session_id = session_id + "_draft"
        
        # 1. Determine starting token and prefix length
        # prefix_len is the length of the cache before this speculation step
        prefix_len = req.total_seq_len
        
        # The last generated token (or last prompt token) is the input for the draft model
        last_token = req.generated_ids[-1] if req.generated_ids else req.prompt_ids[-1]
        
        # 2. Draft Model: Generate candidates
        candidate_tokens = []
        draft_logits_list = []
        
        current_input = last_token
        
        # Ensure draft session is initialized
        if hasattr(self.draft_wrapper.manager, "init_session"):
            self.draft_wrapper.manager.init_session(draft_session_id, prefill_len=prefix_len)
            
        for step in range(self.num_candidates):
            # Run one draft step
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
                
            logits = out.logits[0, -1, :]  # [vocab_size]
            draft_logits_list.append(logits.cpu())
            
            # Sample next draft candidate (greedy or using request settings)
            # Use greedy for draft generation to ensure high acceptance rate
            next_draft_token = logits.argmax().item()
            candidate_tokens.append(next_draft_token)
            current_input = next_draft_token

        # 3. Main Model: Run parallel verification pass
        # The input is the candidate tokens: [last_token] + candidate_tokens
        # We feed the entire list of candidates so that the KV cache for all accepted candidates
        # is computed and stored, avoiding gaps in the cache.
        verification_input = [last_token] + candidate_tokens
        
        verification_input_tensor = torch.tensor([verification_input], dtype=torch.long, device=device)
        verification_position_tensor = torch.arange(
            prefix_len - 1, prefix_len - 1 + len(verification_input),
            dtype=torch.long, device=device
        ).unsqueeze(0)
        
        # Inject main session ID
        self.main_wrapper.model._dkv_session_ids = [session_id]
        
        # Disable lm_head slicing during the verification pass
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
        
        # Check candidate tokens one by one
        for i in range(len(candidate_tokens)):
            target_token = candidate_tokens[i]
            pred_logits = main_logits[i]
            
            # Greedy verification for simplicity and speed, or temperature-based if specified
            if temperature == 0.0:
                pred_token = pred_logits.argmax().item()
                if os.environ.get("DKV_TELEMETRY", "0") == "1":
                    print(f"[Speculative Verification] step {i}: pred={pred_token} ({self.main_wrapper.tokenizer.decode([pred_token])!r}) vs draft={target_token} ({self.main_wrapper.tokenizer.decode([target_token])!r})")
                if pred_token == target_token:
                    accepted_tokens.append(target_token)
                    if target_token in self.main_wrapper.stop_token_ids or target_token == self.main_wrapper.tokenizer.eos_token_id:
                        break
                else:
                    correction_token = pred_token
                    break
            else:
                # Sample-based verification
                # Calculate probabilities
                main_probs = F.softmax(pred_logits / max(temperature, 1e-5), dim=-1)
                draft_probs = F.softmax(draft_logits_list[i] / max(temperature, 1e-5), dim=-1)
                
                p_main = main_probs[target_token].item()
                p_draft = draft_probs[target_token].item()
                
                # Speculative acceptance criterion
                r = torch.rand(1).item()
                if r < min(1.0, p_main / (p_draft + 1e-9)):
                    accepted_tokens.append(target_token)
                    if os.environ.get("DKV_TELEMETRY", "0") == "1":
                        print(f"[Speculative Verification] step {i} accepted: draft={target_token} ({self.main_wrapper.tokenizer.decode([target_token])!r})")
                    if target_token in self.main_wrapper.stop_token_ids or target_token == self.main_wrapper.tokenizer.eos_token_id:
                        break
                else:
                    # Reject and sample correction token from normalized difference distribution
                    diff_dist = torch.clamp(main_probs - draft_probs, min=0.0)
                    if diff_dist.sum() > 0:
                        diff_dist = diff_dist / diff_dist.sum()
                        correction_token = torch.multinomial(diff_dist, 1).item()
                    else:
                        correction_token = pred_logits.argmax().item()
                    if os.environ.get("DKV_TELEMETRY", "0") == "1":
                        print(f"[Speculative Verification] step {i} rejected: draft={target_token} ({self.main_wrapper.tokenizer.decode([target_token])!r}), correction={correction_token} ({self.main_wrapper.tokenizer.decode([correction_token])!r})")
                    break
                    
        # If all candidates accepted, sample correction token from the last logit
        if correction_token is None and (not accepted_tokens or (accepted_tokens[-1] not in self.main_wrapper.stop_token_ids and accepted_tokens[-1] != self.main_wrapper.tokenizer.eos_token_id)):
            # We need to run one step of the main model on candidate_tokens[-1] to get the next logits,
            # or sample from the last output logits of the verification pass!
            # Wait, the verification pass input was [last_token, t_1, t_2, t_3], so the last output logit
            # at index len(verification_input) - 1 is the prediction after t_3 (which is t_4).
            # So that is the prediction for candidate_tokens[-1].
            # Therefore, we can sample the correction token directly from the last logit of the verification pass!
            last_logits = main_logits[-1]
            if temperature == 0.0:
                correction_token = last_logits.argmax().item()
            else:
                probs = F.softmax(last_logits / max(temperature, 1e-5), dim=-1)
                correction_token = torch.multinomial(probs, 1).item()
                
        # 5. Rollback caches to match the accepted length
        # Both caches had candidate tokens appended.
        # Main cache length before this step: prefix_len - 1 (since last_token was NOT appended to KV cache yet).
        # Wait, let's verify if last_token was in the main cache.
        # At the start of this turn, req.total_seq_len represents the length of sequence including last_token.
        # But last_token's KV was NOT appended to the main KV cache yet.
        # So the main cache length was prefix_len - 1.
        # The verification pass fed [last_token] + candidate_tokens[:-1], which has length num_candidates.
        # So the main cache length grew to (prefix_len - 1) + num_candidates.
        # We want to keep the KV for prefix_len - 1 tokens + last_token + accepted_tokens.
        # The number of tokens we want to keep is (prefix_len - 1) + 1 + len(accepted_tokens) = prefix_len + len(accepted_tokens).
        # So target_len for both main and draft is exactly: prefix_len + len(accepted_tokens).
        target_len = prefix_len + len(accepted_tokens)
        
        if hasattr(self.main_wrapper.manager, "rollback_session"):
            self.main_wrapper.manager.rollback_session(session_id, target_len)
            
        if hasattr(self.draft_wrapper.manager, "rollback_session"):
            self.draft_wrapper.manager.rollback_session(draft_session_id, target_len)
            
        # 6. Emit accepted tokens + correction token
        new_tokens = accepted_tokens + ([correction_token] if correction_token is not None else [])
        for token_id in new_tokens:
            req.generated_ids.append(token_id)
            emit_fn(req, token_id, step_start)
            
        return new_tokens
