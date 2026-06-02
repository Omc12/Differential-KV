import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""

long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])

def main():
    print("Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()

    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("Running baseline generation...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.0,
        )
    print(f"Done in {time.perf_counter() - t0:.2f}s")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("--- BASELINE RESPONSE ---")
    print(response)
    print("-------------------------")

if __name__ == "__main__":
    main()
