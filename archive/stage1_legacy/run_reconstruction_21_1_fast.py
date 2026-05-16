
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.sabeaf_resolver import SABEAFResolver
from runtime.hsha_resolver import HSHAResolver
from runtime.srl_resolver import SRLResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_21_1"
os.makedirs(RESULTS_DIR, exist_ok=True)

def fidelity(output, target):
    if not target: return 0.0
    target = target.lower(); output = output.lower()
    if target in output: return 1.0
    import editdistance
    best_dist = len(target); t_len = len(target)
    for i in range(len(output) - t_len + 1):
        window = output[i:i+t_len]
        dist = editdistance.eval(window, target)
        if dist < best_dist: best_dist = dist
    return max(0.0, 1.0 - (best_dist / t_len))

def run_test():
    # Load Model ONCE
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    model = loader.load(attn_implementation="sdpa")
    suite = SPSPrecisionSuite(tokenizer)

    tests = [
        {"mode": "srl_21_1", "ctx": 4096, "domain": "api_key_complex", "len": 64},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "hex_sequence", "len": 64},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "structured_id", "len": 64},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "json_exact", "len": 64},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "activation_code", "len": 64},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "adversarial_delimiters", "len": 64},
        {"mode": "hsha_21_0", "ctx": 4096, "domain": "api_key_complex", "len": 64},
        {"mode": "sabeaf_20_8", "ctx": 4096, "domain": "api_key_complex", "len": 64},
        {"mode": "dense", "ctx": 4096, "domain": "api_key_complex", "len": 64},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "propagation_chain", "len": 128},
        {"mode": "srl_21_1", "ctx": 4096, "domain": "json_reconstruction", "len": 64},
    ]

    print(f"Starting Phase 21.1 FAST Batch Smoke Test ({len(tests)} tests)...")
    total_start = time.time()
    
    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['mode']} | {test['domain']}...")
        
        # Setup Resolver
        if test["mode"] == "dense":
            resolver = None
        elif test["mode"] == "sabeaf_20_8":
            resolver = SABEAFResolver(tokenizer)
        elif test["mode"] == "hsha_21_0":
            resolver = HSHAResolver(tokenizer)
        elif test["mode"] == "srl_21_1":
            resolver = SRLResolver(tokenizer)
        else:
            resolver = None

        # Setup Case
        test_case = suite.create_case(test["domain"], test["ctx"], target_len=test["len"])
        input_ids = torch.tensor([test_case["tokens"]], device=model.device)
        needle = test_case["needle"]

        # Reset VRAM & Generation
        torch.cuda.empty_cache()
        past_key_values = DynamicCache()
        
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :].float()
            attentions = None
            del outputs

        generated_tokens = []
        gen_start = time.time()
        for i in range(test["len"] + 32):
            if resolver:
                logits = resolver.guide_decoder(logits, attentions)
            
            token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(token_id)
            
            if resolver:
                resolver.record_generated_token(token_id, logits.detach().cpu())
                
            with torch.no_grad():
                outputs = model(
                    torch.tensor([[token_id]], device=model.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True
                )
                if resolver:
                    resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), torch.tensor([[token_id]], device=model.device))
                
                logits = outputs.logits[:, -1, :].detach().float()
                attentions = torch.stack([a.detach() for a in outputs.attentions]) if outputs.attentions else None
                del outputs
            
            if i % 10 == 0:
                print(".", end="", flush=True)
            if token_id == tokenizer.eos_token_id: break
        print()

        # Results
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        fid_score = fidelity(output_text, needle)
        exact_match = needle.lower() in output_text.lower()
        tps = len(generated_tokens) / (time.time() - gen_start)
        
        res = {
            "mode": test["mode"], "ctx": test["ctx"], "domain": test["domain"], "prop_len": test["len"],
            "exact_match": exact_match, "fidelity": fid_score, "tps": tps
        }
        
        # Log results
        with open(os.path.join(RESULTS_DIR, "raw_symbolic_propagation.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
            
        if test["mode"] == "srl_21_1" and resolver:
            stats = resolver.get_srl_summary()
            with open(os.path.join(RESULTS_DIR, "raw_srl_metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({**res, **stats}) + "\n")

        print(f"  Result: EM={exact_match} Fid={fid_score:.3f} TPS={tps:.1f}")

    print(f"FAST Smoke Test Complete. Total duration: {(time.time() - total_start)/60:.2f} minutes.")

if __name__ == "__main__":
    run_test()
