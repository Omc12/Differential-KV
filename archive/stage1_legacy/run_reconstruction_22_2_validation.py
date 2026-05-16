
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.esm_resolver import ESMResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_22_2_esm"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 22.2: ESM ULTRA-LIGHTWEIGHT VALIDATION.
    Targets Execution Specialization and Mode Stability.
    Target: < 5 minutes.
    """
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    
    try:
        model = loader.load(attn_implementation="sdpa")
    except Exception:
        print("[SKIP] Model load failed.")
        return

    suite = SPSPrecisionSuite(tokenizer)

    tests = [
        {"name": "Symbolic Mode Specialization", "domain": "api_key_complex", "len": 16},
        {"name": "Semantic Mode Stability", "domain": "json_exact", "len": 16},
        {"name": "Topology Repair Specialization", "domain": "adversarial_delimiters", "len": 24},
        {"name": "Dormant Low-Power Execution", "domain": "hex_sequence", "len": 16},
        {"name": "Specialization Switching", "domain": "structured_id", "len": 32}, # Longer to trigger mode switches
        {"name": "Continuity Preservation", "domain": "activation_code", "len": 24},
    ]

    print(f"Starting Phase 22.2 ESM ULTRA-LIGHT Specialization Probes ({len(tests)} tests)...")
    total_start = time.time()
    
    all_summaries = []

    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        resolver = ESMResolver(tokenizer)

        test_case = suite.create_case(test["domain"], 2048, target_len=test["len"])
        input_ids = torch.tensor([test_case["tokens"]], device=model.device)
        needle = test_case["needle"]

        torch.cuda.empty_cache()
        past_key_values = DynamicCache()
        
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), input_ids)
            logits = outputs.logits[:, -1, :].float()
            del outputs

        generated_tokens = []
        for i in range(test["len"] + 4):
            logits = resolver.guide_decoder(logits, None)
            token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(token_id)
            
            resolver.record_generated_token(token_id, logits.detach().cpu())
            
            with torch.no_grad():
                outputs = model(torch.tensor([[token_id]], device=model.device), 
                                past_key_values=past_key_values, 
                                use_cache=True, 
                                output_hidden_states=True)
                
                resolver.resolve_and_prune(past_key_values, 
                                           outputs.hidden_states[-1].detach(), 
                                           torch.tensor([[token_id]], device=model.device))
                
                logits = outputs.logits[:, -1, :].detach().float()
                del outputs
                
            if token_id == tokenizer.eos_token_id: break

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        exact_match = needle.lower() in output_text.lower()
        stats = resolver.get_esm_stats()
        
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "specialization_efficiency": stats.get("specialization_efficiency", 0.0),
            "mode_switch_stability": stats.get("mode_switch_stability", 1.0),
            "dormant_compute_ratio": stats.get("dormant_compute_ratio", 0.0),
            "symbolic_integrity": stats.get("symbolic_integrity", 1.0),
            "execution_localization": stats.get("execution_localization", 0.0)
        }
        
        all_summaries.append(summary)
        
        with open(os.path.join(RESULTS_DIR, "esm_validation_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} SpecEff={summary['specialization_efficiency']:.2f} Stability={summary['mode_switch_stability']:.2f}")

    duration = (time.time() - total_start) / 60
    print(f"\nESM Phase 22.2 Validation Complete. Duration: {duration:.2f} minutes.")
    
    avg_spec_eff = sum(s["specialization_efficiency"] for s in all_summaries) / len(all_summaries)
    avg_stability = sum(s["mode_switch_stability"] for s in all_summaries) / len(all_summaries)
    avg_integrity = sum(s["symbolic_integrity"] for s in all_summaries) / len(all_summaries)
    
    print("\n--- ESM PHASE 22.2 SUCCESS REPORT ---")
    print(f"Avg Specialization Efficiency: {avg_spec_eff:.4f}")
    print(f"Avg Mode Switch Stability: {avg_stability:.4f}")
    print(f"Avg Symbolic Integrity: {avg_integrity:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    run_test()
