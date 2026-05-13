import torch
from typing import List

class ModelAccuracyPreservation:
    """
    Evaluates how much model accuracy is preserved when using Differential KV.
    Uses metrics like perplexity or KL-divergence between dense and sparse outputs.
    """
    def __init__(self, dense_model, sparse_model, tokenizer):
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.tokenizer = tokenizer

    def compare_logits(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.dense_model.device)
        
        with torch.no_grad():
            dense_logits = self.dense_model(**inputs).logits
            sparse_logits = self.sparse_model(**inputs).logits
            
        # KL Divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(sparse_logits, dim=-1),
            torch.nn.functional.softmax(dense_logits, dim=-1),
            reduction='batchmean'
        )
        
        # Logit MSE
        mse = torch.mean((dense_logits - sparse_logits)**2)
        
        return {
            "kl_divergence": kl_div.item(),
            "logit_mse": mse.item()
        }
