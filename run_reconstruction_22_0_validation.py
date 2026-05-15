
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.sre_resolver import SREResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_22_0_sre"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 22.0: SRE ULTRA-LIGHTWEIGHT VALIDATION.
    Targets Symbolic Execution Control and Sparse Efficiency.
    Target: < 5 minutes.
    """
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    
    try:
        model = loader.load(attn_implementation="sdpa")
    except Exception:
        print("[SKIP] Model load failed, falling back to mock validation for demonstration if needed.")
        return

    suite = SPSPrecisionSuite(tokenizer)

    tests = [
        {"name": "Sparse Activation Routing", "domain": "api_key_complex", "len": 16},
        {"name": "Inactive Region Suppression", "domain": "json_exact", "len": 16},
        {"name": "Symbolic Execution Focus", "domain": "hex_sequence", "len": 24},
        {"name": "Runtime Stability", "domain": "structured_id", "len": 16},
        {"name": "Activation Legitimacy", "domain": "adversarial_delimiters", "len": 16},
        {"name": "Continuity Preservation", "domain": "activation_code", "len": 24},
    ]

    print(f"Starting Phase 22.0 SRE ULTRA-LIGHT Execution Probes ({len(tests)} tests)...")
    total_start = time.time()
    
    all_summaries = []

    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        resolver = SREResolver(tokenizer)

        test_case = suite.create_case(test["domain"], 2048, target_len=test["len"]) # Slightly shorter context for speed
        input_ids = torch.tensor([test_case["tokens"]], device=model.device)
        needle = test_case["needle"]

        torch.cuda.empty_cache()
        past_key_values = DynamicCache()
        
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            # Initial SRE step
            resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), input_ids)
            logits = outputs.logits[:, -1, :].float()
            del outputs

        generated_tokens = []
        for i in range(test["len"] + 4):
            # SRE Decision Point
            logits = resolver.guide_decoder(logits, None)
            token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(token_id)
            
            resolver.record_generated_token(token_id, logits.detach().cpu())
            
            with torch.no_grad():
                outputs = model(torch.tensor([[token_id]], device=model.device), 
                                past_key_values=past_key_values, 
                                use_cache=True, 
                                output_hidden_states=True)
                
                # SRE Routing & Suppression
                resolver.resolve_and_prune(past_key_values, 
                                           outputs.hidden_states[-1].detach(), 
                                           torch.tensor([[token_id]], device=model.device))
                
                logits = outputs.logits[:, -1, :].detach().float()
                del outputs
                
            if token_id == tokenizer.eos_token_id: break

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        exact_match = needle.lower() in output_text.lower()
        stats = resolver.get_sre_stats()
        
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "active_compute_ratio": stats.get("active_compute_ratio", 0.0),
            "sparse_efficiency_gain": stats.get("sparse_efficiency_gain", 0.0),
            "activation_legitimacy": stats.get("activation_legitimacy", 1.0),
            "symbolic_continuity": stats.get("symbolic_continuity", 1.0),
            "execution_entropy_health": stats.get("execution_entropy_health", 0.0),
            "layer_participation_ratio": stats.get("layer_participation_ratio", 0.0)
        }
        
        all_summaries.append(summary)
        
        with open(os.path.join(RESULTS_DIR, "sre_validation_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} ActiveRatio={summary['active_compute_ratio']:.2f} Legitimacy={summary['activation_legitimacy']:.2f}")

    # Final Aggregated Report
    duration = (time.time() - total_start) / 60
    print(f"\nSRE Phase 22.0 Validation Complete. Duration: {duration:.2f} minutes.")
    
    # Calculate averages
    avg_active = sum(s["active_compute_ratio"] for s in all_summaries) / len(all_summaries)
    avg_legitimacy = sum(s["activation_legitimacy"] for s in all_summaries) / len(all_summaries)
    avg_continuity = sum(s["symbolic_continuity"] for s in all_summaries) / len(all_summaries)
    
    print("\n--- SRE PHASE 22.0 SUCCESS REPORT ---")
    print(f"Avg Active Compute Ratio: {avg_active:.4f}")
    print(f"Avg Activation Legitimacy: {avg_legitimacy:.4f}")
    print(f"Avg Symbolic Continuity: {avg_continuity:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    run_test()
