import os
import sys
import torch
import asyncio

async def main():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("\n--- Loading Standard HF (Dense) ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    
    with open("/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/scratch/berry_paper.txt", "r") as f:
        paper_text = f.read()
        
    messages1 = [
        {"role": "user", "content": f"Here is the paper reference:\n\n{paper_text}\n\nAcknowledge that you have received the paper by saying 'Paper received successfully.'"}
    ]
    prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    
    inputs1 = tokenizer(prompt1, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(**inputs1, max_new_tokens=128, temperature=0.0, do_sample=False)
    
    gen1 = tokenizer.decode(out1[0][inputs1.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("Response 1:", gen1)
    
    messages2 = [
        {"role": "user", "content": f"Here is the paper reference:\n\n{paper_text}\n\nAcknowledge that you have received the paper by saying 'Paper received successfully.'"},
        {"role": "assistant", "content": gen1},
        {"role": "user", "content": "Question: What are the codimensions of degeneracies for: (1) real symmetric hermitian matrices, (2) complex hermitian matrices, and (3) nonhermitian matrices? Compare their eigenvalue and eigenvector behavior around the degeneracy based on the paper."}
    ]
    prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=512, temperature=0.0, do_sample=False)
        
    gen2 = tokenizer.decode(out2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("\nResponse 2:")
    print(gen2)

if __name__ == "__main__":
    asyncio.run(main())
