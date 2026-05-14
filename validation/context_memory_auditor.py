import torch

class ContextMemoryAuditor:
    """
    PHASE 18.2F: Audits retrieval survival after sparse pruning.
    Tests if the model can still recall information from the 'pruned' past.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def audit_recall(self, prompt, expected_recall_phrase):
        """
        Runs generation and checks if the expected phrase is present.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
            
        generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        success = expected_recall_phrase.lower() in generated_text.lower()
        
        return {
            "success": success,
            "generated": generated_text,
            "expected": expected_recall_phrase
        }
