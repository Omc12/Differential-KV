import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.universal_engine import UniversalCompressionEngine

def extract_answer(completion):
    match = re.search(r"####\s*(-?\d+)", completion)
    if match: return match.group(1)
    numbers = re.findall(r"-?\d+", completion)
    return numbers[-1] if numbers else None

def generate_manual(model, tokenizer, input_ids, past_kv, max_new_tokens=100):
    last_token_id = input_ids[:, -1:]
    with torch.no_grad():
        outputs = model(last_token_id, past_key_values=past_kv, use_cache=True)
        curr_kv = outputs.past_key_values
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = []
        for _ in range(max_new_tokens):
            token = next_tok.item()
            if token == tokenizer.eos_token_id: break
            generated.append(token)
            outputs = model(next_tok, past_key_values=curr_kv, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_kv = outputs.past_key_values
    return tokenizer.decode(generated, skip_special_tokens=True)

def run_gsm8k_eval(models, modes, samples, output_path):
    results = {}
    few_shot_prompt = (
        "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
        "Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. #### 6\n"
        "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
        "Answer: There are 3 cars originally. Then 2 more cars arrived. 3 + 2 = 5. #### 5\n"
    )
    dataset = load_dataset("gsm8k", "main", split=f"test[:{samples}]")
    for model_id in models:
        print(f"\n>>> Loading Model: {model_id}")
        model_map = {"qwen2-0.5b": "Qwen/Qwen2-0.5B", "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "phi2": "microsoft/phi-2", "qwen2-1.5b": "Qwen/Qwen2-1.5B"}
        hf_id = model_map.get(model_id, model_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            engine = UniversalCompressionEngine(model, tokenizer)
        except Exception as e:
            print(f"Error loading {model_id}: {e}"); continue
        results[model_id] = {}
        for mode in modes:
            print(f"  Evaluating Mode: {mode}")
            correct, total = 0, 0
            for item in tqdm(dataset, desc=f"{model_id} | {mode}"):
                question = item["question"]
                gold_answer = extract_answer(item["answer"])
                prompt = few_shot_prompt + f"Question: {question}\nAnswer:"
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
                prefix_ids = input_ids[:, :-1]
                with torch.no_grad():
                    outputs = model(prefix_ids, use_cache=True)
                    past_kv_recon, stats = engine.compress_kv(outputs.past_key_values, mode)
                    completion = generate_manual(model, tokenizer, input_ids, past_kv_recon)
                    if extract_answer(completion) == gold_answer: correct += 1
                    total += 1
            results[model_id][mode] = {"accuracy": correct/total if total > 0 else 0, "correct": correct, "total": total, "compression_ratio": stats.get("ratio", 1.0)}
        del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f: json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen2-0.5b"])
    parser.add_argument("--modes", nargs="+", default=["fp16", "int8", "rank8", "sam", "actr", "lcg"])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="results/phase20/gsm8k_full.json")
    args = parser.parse_args()
    run_gsm8k_eval(args.models, args.modes, args.samples, args.output)
