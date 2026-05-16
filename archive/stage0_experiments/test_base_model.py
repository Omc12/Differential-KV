import torch
from models.qwen7b_real_loader import Qwen7BRealLoader
from transformers import AutoTokenizer

def test_base():
    loader = Qwen7BRealLoader()
    model = loader.load(attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"Generating for prompt: {prompt}")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"Result: {tokenizer.decode(outputs[0])}")

if __name__ == "__main__":
    test_base()
