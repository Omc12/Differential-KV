import torch
from transformers import GenerationConfig

class HFGenerationBridge:
    """
    Bridges Differential KV runtime with HuggingFace generation utilities.
    Ensures that generation follows the real model's forward path.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "tokens": generated_tokens.tolist(),
            "input_len": inputs.input_ids.shape[1],
            "output_len": len(generated_tokens)
        }

    def stream_generate(self, prompt, max_new_tokens=50):
        """
        Stub for streaming generation to be used in live serving.
        """
        # Implementation would use TextStreamer or similar
        pass
