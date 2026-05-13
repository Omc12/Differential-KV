import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class QwenBaselineRunner:
    """
    Runs baseline benchmarks for Qwen models using standard HF implementation.
    Establishes the 'Dense' ground truth for comparison.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )

    def run_inference(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        duration = time.perf_counter() - start
        
        token_count = outputs.shape[1] - inputs.input_ids.shape[1]
        tps = token_count / duration
        
        return {
            "text": self.tokenizer.decode(outputs[0], skip_special_tokens=True),
            "tokens": token_count,
            "latency": duration,
            "tps": tps
        }
