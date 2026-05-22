import torch
import time
from typing import Dict, Any, List

class RealOutputCorrelationEngine:
    """
    STAGE 4B.1.6 — ERCA Real Output Correlation Engine.
    Correlates actual generated tokens and text strings step-by-step with the raw 
    logits computed by the model's final lm_head linear projection on the GPU.
    """
    def __init__(self):
        self.hooks = []
        self.step_logits = []
        self.emitted_tokens = []
        self.passed = True
        self.violations = []

    def register_hooks(self, model: torch.nn.Module):
        """
        Registers a forward hook on the lm_head layer to extract step-by-step logits.
        """
        self.remove_hooks()
        self.step_logits.clear()
        self.emitted_tokens.clear()
        self.passed = True
        self.violations.clear()

        # Target lm_head layer
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            lm_head = getattr(model, "embed_out", None)

        if lm_head is not None:
            hook = lm_head.register_forward_hook(self._make_lm_head_hook())
            self.hooks.append(hook)

    def remove_hooks(self):
        """
        Removes the registered lm_head hook.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _make_lm_head_hook(self):
        def lm_head_hook(module, args, output):
            if output is not None:
                # Copy the last position's logit vector to CPU to prevent VRAM retention
                last_logits = output[0, -1, :].detach().cpu()
                self.step_logits.append(last_logits)
        return lm_head_hook

    def record_emitted_token(self, token_id: int, token_str: str):
        """
        Records the actual token string and ID returned by the tokenizer decoder.
        """
        self.emitted_tokens.append({
            "token_id": token_id,
            "token_str": token_str,
            "timestamp": time.time()
        })

    def verify_correlation(self) -> Dict[str, Any]:
        """
        Compares output token indexes against computed step logits to verify physical lineage.
        """
        if not self.step_logits or not self.emitted_tokens:
            self.passed = False
            self.violations.append("Logits or token records are missing. No execution data gathered.")
            return {
                "passed": False,
                "violations": self.violations,
                "matches_count": 0,
                "total_count": 0,
                "match_ratio": 0.0
            }

        matches = 0
        total = min(len(self.step_logits), len(self.emitted_tokens))
        details = []

        for idx in range(total):
            logits = self.step_logits[idx]
            emitted = self.emitted_tokens[idx]

            # The index with maximum logit value
            top_pred_id = int(logits.argmax(dim=-1).item())
            emitted_id = emitted["token_id"]

            logit_value = float(logits[emitted_id].item())
            is_valid_logit = not (torch.isnan(logits[emitted_id]) or torch.isinf(logits[emitted_id]))

            is_exact_match = (top_pred_id == emitted_id)
            if is_exact_match:
                matches += 1

            details.append({
                "step": idx,
                "emitted_token_id": emitted_id,
                "emitted_token_str": emitted["token_str"],
                "top_predicted_id": top_pred_id,
                "exact_match": is_exact_match,
                "logit_value": logit_value,
                "is_valid_logit": is_valid_logit
            })

        match_ratio = matches / total if total > 0 else 0.0

        for d in details:
            if not d["is_valid_logit"]:
                self.passed = False
                self.violations.append(f"Invalid logit value (NaN/Inf) at step {d['step']} for token '{d['emitted_token_str']}'")

        if total == 0:
            self.passed = False
            self.violations.append("Zero logits matched with generated outputs.")

        return {
            "passed": self.passed and len(self.violations) == 0,
            "violations": self.violations,
            "match_ratio": match_ratio,
            "matches_count": matches,
            "total_count": total,
            "details": details
        }
