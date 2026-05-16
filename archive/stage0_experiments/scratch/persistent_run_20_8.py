
import os
import sys
import gc
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

# Ensure project root is in path for local imports
PROJECT_ROOT = r"d:\Codes\Projects\Differential KV"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from validation.sps_precision_suite import SPSPrecisionSuite
from runtime.sabeaf_resolver import SABEAFResolver
from runtime.spslrif_resolver import SPSLRIFResolver
from runtime.pposah_resolver import PPOSAHResolver
from runtime.alfsr_resolver import ALFSRResolver

# results setup
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "reconstruction_20_8")
os.makedirs(RESULTS_DIR, exist_ok=True)
JSONL_PATH = os.path.join(RESULTS_DIR, "raw_symbolic_propagation.jsonl")

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    with open(os.path.join(RESULTS_DIR, "raw_wallclock_trace.log"), "a") as f:
        f.write(f"[{timestamp}] [PERSISTENT] {msg}\n")

def run_persistent_validation():
    log("Initializing Persistent High-Velocity Runner (Phase 20.8)...")
    
    # 1. Load Model Once
    # Use standard HuggingFace repo ID
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    log(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )
    model.eval()
    
    # 2. Setup Suite
    suite = SPSPrecisionSuite(tokenizer)
    
    # 3. Define Matrix (Post-20.7 Strategy: 8k Primary)
    contexts = [8192, 16384]
    prop_lengths = [64, 128]
    domains = ["hex_sequence", "api_key_complex", "propagation_chain", "delimiter_integrity", "structured_id"]
    modes = ["sabeaf_20_8", "spslrif_20_7", "pposah_20_6a", "dense"]
    
    total_runs = len(contexts) * len(prop_lengths) * len(domains) * len(modes)
    current_run = 0
    
    # 4. Loop through matrix
    for ctx in contexts:
        for prop_len in prop_lengths:
            for domain in domains:
                for mode in modes:
                    current_run += 1
                    log(f"[{current_run}/{total_runs}] Running: mode={mode} ctx={ctx} domain={domain} len={prop_len}")
                    
                    try:
                        # Initialize Resolver
                        if mode == "sabeaf_20_8":
                            resolver = SABEAFResolver(tokenizer)
                        elif mode == "spslrif_20_7":
                            resolver = SPSLRIFResolver(tokenizer)
                        elif mode == "pposah_20_6a":
                            resolver = PPOSAHResolver(tokenizer)
                        else: # dense
                            resolver = None
                        
                        # Get test data
                        test_case = suite.create_case(domain, ctx, target_len=prop_len)
                        input_ids = torch.tensor([test_case["tokens"]], device=model.device)
                        target = test_case["needle"]
                        
                        # Execution
                        start_time = time.time()
                        with torch.no_grad():
                            if resolver:
                                resolver.reset_generation_state()
                                past_key_values = DynamicCache()
                                
                                # Prefill
                                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_attentions=False)
                                logits = outputs.logits[:, -1, :].float()
                                attentions = None
                                
                                # Generate
                                generated_ids = []
                                current_token_id = None
                                
                                for i in range(prop_len + 32):
                                    if resolver:
                                        logits = resolver.guide_decoder(logits, attentions)
                                    
                                    token_id = torch.argmax(logits, dim=-1).item()
                                    generated_ids.append(token_id)
                                    
                                    if resolver:
                                        resolver.record_generated_token(token_id, logits.detach().cpu())
                                        
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
                                    
                                    if token_id == tokenizer.eos_token_id: break
                                    
                                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                            else:
                                # Dense baseline
                                outputs = model.generate(
                                    input_ids, 
                                    max_new_tokens=prop_len, 
                                    do_sample=False, 
                                    use_cache=True
                                )
                                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                        
                        duration = time.time() - start_time
                        
                        # Verify
                        target = test_data["target"].strip()
                        is_match = target in response.strip()
                        fidelity = 1.0 if is_match else 0.0 
                        tps = prop_len / duration
                        
                        # Save
                        result = {
                            "mode": mode, "ctx": ctx, "domain": domain, "prop_len": prop_len,
                            "exact_match": is_match, "fidelity": fidelity, "tps": tps
                        }
                        with open(JSONL_PATH, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        
                        log(f"  Result: Match={is_match} TPS={tps:.2f}")
                        
                    except Exception as e:
                        log(f"  FAILED Case {current_run}: {e}")
                    
                    # 5. Flush VRAM between runs to prevent 16k OOM bleed
                    gc.collect()
                    torch.cuda.empty_cache()

    log("Persistent Validation Suite Complete.")

if __name__ == "__main__":
    run_persistent_validation()
