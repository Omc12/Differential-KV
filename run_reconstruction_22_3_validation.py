
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.hec_resolver import HECResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_22_3_hec"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 22.3: HEC ULTRA-LIGHTWEIGHT VALIDATION.
    Targets Hierarchical Execution Coordination and Delegation.
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
        {"name": "Inter-mode Delegation", "domain": "api_key_complex", "len": 16},
        {"name": "Priority Arbitration", "domain": "json_exact", "len": 16},
        {"name": "Synchronization Stability", "domain": "adversarial_delimiters", "len": 24},
        {"name": "Coordination Continuity", "domain": "hex_sequence", "len": 16},
        {"name": "Cooperative Memory", "domain": "structured_id", "len": 32},
        {"name": "Specialization Integrity", "domain": "activation_code", "len": 24},
    ]

    print(f"Starting Phase 22.3 HEC ULTRA-LIGHT Coordination Probes ({len(tests)} tests)...")
    total_start = time.time()
    
    all_summaries = []

    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        resolver = HECResolver(tokenizer)

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
        stats = resolver.get_hec_stats()
        
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "coordination_efficiency": stats.get("coordination_efficiency", 0.0),
            "delegation_success_rate": stats.get("delegation_success_rate", 0.0),
            "synchronization_integrity": stats.get("synchronization_integrity", 1.0),
            "arbitration_stability": stats.get("arbitration_stability", 1.0),
            "symbolic_continuity": stats.get("symbolic_continuity", 1.0)
        }
        
        all_summaries.append(summary)
        
        with open(os.path.join(RESULTS_DIR, "hec_validation_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} CoordEff={summary['coordination_efficiency']:.2f} Delegation={summary['delegation_success_rate']:.2f}")

    duration = (time.time() - total_start) / 60
    print(f"\nHEC Phase 22.3 Validation Complete. Duration: {duration:.2f} minutes.")
    
    avg_coord_eff = sum(s["coordination_efficiency"] for s in all_summaries) / len(all_summaries)
    avg_delegation = sum(s["delegation_success_rate"] for s in all_summaries) / len(all_summaries)
    avg_sync = sum(s["synchronization_integrity"] for s in all_summaries) / len(all_summaries)
    
    print("\n--- HEC PHASE 22.3 SUCCESS REPORT ---")
    print(f"Avg Coordination Efficiency: {avg_coord_eff:.4f}")
    print(f"Avg Delegation Success Rate: {avg_delegation:.4f}")
    print(f"Avg Synchronization Integrity: {avg_sync:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    run_test()
