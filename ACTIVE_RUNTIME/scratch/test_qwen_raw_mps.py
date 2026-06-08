import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps"
dtype = torch.float16

from test_sustainable_ai_prompt import PROMPT

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    + PROMPT + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="eager"
).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = hf_model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)
print("HF Output on MPS:")
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
