import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase20.validation.compression_engine import UniversalCompressionEngine

def run_cross_arch_validation(model_id, modes, output_path):
    print(f"\n>>> Validating Architecture: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        engine = UniversalCompressionEngine(model, tokenizer)
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        return

    results = {"model_id": model_id, "modes": {}}
    
    test_prompt = "The quick brown fox jumps over the lazy dog. Scientific discovery requires rigorous validation and open peer review."
    
    for mode in modes:
        print(f"  Testing Mode: {mode}")
        input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            past_kv_recon, stats = engine.compress_kv(past_kv, mode)
            
            # Measure reconstruction error (MSE)
            recon_mse = 0
            count = 0
            for i in range(len(past_kv)):
                orig_k = past_kv[i][0]
                recon_k = past_kv_recon[i][0]
                recon_mse += torch.nn.functional.mse_loss(orig_k.float(), recon_k.float()).item()
                count += 1
            avg_mse = recon_mse / count
            
            results["modes"][mode] = {
                "avg_mse": avg_mse,
                "compression_ratio": stats.get("ratio", 1.0),
                "stability_score": 1.0 / (1.0 + avg_mse)
            }
            print(f"    MSE: {avg_mse:.6f} | Ratio: {stats.get('ratio', 1.0):.2f}x")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--modes", nargs="+", default=["fp16", "rank8", "sam", "actr", "lcg"])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    run_cross_arch_validation(args.model, args.modes, args.output)
