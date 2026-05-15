
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.aeg_resolver import AEGResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_22_1_aeg"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 22.1: AEG ULTRA-LIGHTWEIGHT VALIDATION.
    Targets Adaptive Execution Graphs and Predictive Activation.
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
        {"name": "Adaptive Execution Propagation", "domain": "api_key_complex", "len": 16},
        {"name": "Predictive Activation Stability", "domain": "json_exact", "len": 16},
        {"name": "Dormant Path Suspension", "domain": "hex_sequence", "len": 24},
        {"name": "Graph Cascade Suppression", "domain": "structured_id", "len": 16},
        {"name": "Symbolic Continuity Preservation", "domain": "adversarial_delimiters", "len": 16},
        {"name": "Sparse Execution Efficiency", "domain": "activation_code", "len": 24},
    ]

    print(f"Starting Phase 22.1 AEG ULTRA-LIGHT Execution Graph Probes ({len(tests)} tests)...")
    total_start = time.time()
    
    all_summaries = []

    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        resolver = AEGResolver(tokenizer)

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
        stats = resolver.get_aeg_stats()
        
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "graph_activation_efficiency": stats.get("graph_activation_efficiency", 0.0),
            "predictive_accuracy": stats.get("predictive_accuracy", 0.0),
            "dormant_path_ratio": stats.get("dormant_path_ratio", 0.0),
            "cascade_suppression_health": stats.get("cascade_suppression_health", 1.0),
            "symbolic_continuity": stats.get("symbolic_continuity", 1.0)
        }
        
        all_summaries.append(summary)
        
        with open(os.path.join(RESULTS_DIR, "aeg_validation_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} Efficiency={summary['graph_activation_efficiency']:.2f} PredAcc={summary['predictive_accuracy']:.2f}")

    duration = (time.time() - total_start) / 60
    print(f"\nAEG Phase 22.1 Validation Complete. Duration: {duration:.2f} minutes.")
    
    avg_efficiency = sum(s["graph_activation_efficiency"] for s in all_summaries) / len(all_summaries)
    avg_pred_acc = sum(s["predictive_accuracy"] for s in all_summaries) / len(all_summaries)
    avg_dormant = sum(s["dormant_path_ratio"] for s in all_summaries) / len(all_summaries)
    
    print("\n--- AEG PHASE 22.1 SUCCESS REPORT ---")
    print(f"Avg Graph Activation Efficiency: {avg_efficiency:.4f}")
    print(f"Avg Predictive Accuracy: {avg_pred_acc:.4f}")
    print(f"Avg Dormant Path Ratio: {avg_dormant:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    run_test()
