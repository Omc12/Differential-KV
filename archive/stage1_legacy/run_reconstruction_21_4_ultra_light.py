
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.lscp_resolver import LSCPResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_21_4_light"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 21.4: LSCP ULTRA-LIGHTWEIGHT VALIDATION.
    Optimized for rapid architectural iteration.
    Target: < 5-8 minutes.
    """
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    model = loader.load(attn_implementation="sdpa")
    suite = SPSPrecisionSuite(tokenizer)

    tests = [
        {"name": "Dormant Persistence", "domain": "api_key_complex", "len": 16},
        {"name": "Symbolic Resurrection", "domain": "hex_sequence", "len": 16},
        {"name": "Lineage After Dormancy", "domain": "structured_id", "len": 32},
        {"name": "Stale Recall Suppression", "domain": "json_exact", "len": 16},
        {"name": "Persistence Decay", "domain": "activation_code", "len": 32},
        {"name": "Topology Survival", "domain": "adversarial_delimiters", "len": 32},
    ]

    print(f"Starting Phase 21.4 LSCP ULTRA-LIGHT Batch Smoke Test ({len(tests)} tests)...")
    total_start = time.time()
    
    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        resolver = LSCPResolver(tokenizer)

        test_case = suite.create_case(test["domain"], 4096, target_len=test["len"])
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
        for i in range(test["len"] + 8):
            logits = resolver.guide_decoder(logits, None)
            token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(token_id)
            resolver.record_generated_token(token_id, logits.detach().cpu())
            with torch.no_grad():
                outputs = model(torch.tensor([[token_id]], device=model.device), past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), torch.tensor([[token_id]], device=model.device))
                logits = outputs.logits[:, -1, :].detach().float()
                del outputs
            if token_id == tokenizer.eos_token_id: break

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        exact_match = needle.lower() in output_text.lower()
        stats = resolver.get_lscp_summary()
        
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "resurrection_integrity": stats.get("resurrection_integrity", 0.0),
            "lineage_persistence": stats.get("lineage_persistence", 0),
            "stale_recall_rate": stats.get("stale_recall_rate", 0),
            "persistence_decay": stats.get("persistence_decay_health", 1.0)
        }
        
        with open(os.path.join(RESULTS_DIR, "lscp_persistence_summary.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} Resurrection={summary['resurrection_integrity']:.2f} Lineage={summary['lineage_persistence']}")

    duration = (time.time() - total_start) / 60
    print(f"LSCP ULTRA-LIGHT Test Complete. Total duration: {duration:.2f} minutes.")

if __name__ == "__main__":
    run_test()
