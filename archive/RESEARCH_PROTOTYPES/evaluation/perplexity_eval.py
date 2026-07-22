"""
evaluation/perplexity_eval.py
Phase 8: Model Quality Validation — Perplexity on WikiText-2.
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.shared_basis import SharedBasisManager
from compression.adaptive import AdaptiveRankSelector
from compression.quantization import quantize_int8, dequantize_int8

class Phase8PerplexityEvaluator:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=self.device, 
            trust_remote_code=True
        )
        self.model.eval()
        self.sb_manager = SharedBasisManager()
        self.rank_selector = AdaptiveRankSelector(rank_buckets=[8, 16, 32, 64], method="energy")

    def get_wiki_data(self, n_samples=5, seq_len=2048):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
        encodings = self.tokenizer(text, return_tensors="pt")
        
        stride = seq_len
        samples = []
        for i in range(0, min(encodings.input_ids.size(1) - seq_len, n_samples * stride), stride):
            samples.append(encodings.input_ids[:, i:i+seq_len])
        return samples

    def compress_kv(self, past_kv: Any, mode: str, interval: int = 64) -> Tuple[Any, Dict[str, Any]]:
        """
        Compresses and reconstructs the KV cache according to the specified mode.
        Returns (reconstructed_kv, stats).
        """
        # Runtime Verification Logs
        is_cache_obj = hasattr(past_kv, "get_seq_length")
        print(f"[Audit] Mode: {mode} | Cache type: {'DynamicCache' if is_cache_obj else 'Tuple'} | seq_len: {past_kv.get_seq_length() if is_cache_obj else len(past_kv)}")

        if is_cache_obj:
            kv_list = list(past_kv)
        else:
            kv_list = past_kv

        if mode == "FP16":
            total_bytes = sum(x[0].numel() * 2 + x[1].numel() * 2 for x in kv_list)
            return past_kv, {"ratio": 1.0, "bytes": total_bytes}

        recon_kv_list = []
        total_orig_bytes = 0
        total_comp_bytes = 0
        
        for layer_idx, layer_data in enumerate(kv_list):
            k, v = layer_data[0], layer_data[1]
            b, h, s, d = k.shape
            feat_dim = h * d
            
            # Differential KV logic: 
            # 1. Take anchors at every 'interval'
            # 2. Compress deltas between anchors
            
            # For simplicity in this eval script, we treat the entire cache as one block for basis creation per layer
            # but we compress it in sub-blocks for adaptive rank selection.
            
            k_recon = k.clone().float()
            v_recon = v.clone().float()
            
            k_flat = k.transpose(1, 2).reshape(s, feat_dim).float()
            v_flat = v.transpose(1, 2).reshape(s, feat_dim).float()
            
            total_orig_bytes += k.numel() * 2 + v.numel() * 2
            
            layer_comp_bytes = 0
            
            for component_name, data_flat, recon_target in [("K", k_flat, k_recon), ("V", v_flat, v_recon)]:
                # Anchors
                anchor_indices = list(range(0, s, interval))
                layer_comp_bytes += len(anchor_indices) * feat_dim * 2 # Anchors are FP16
                
                # Basis for this layer/component
                # Use deltas to form basis
                deltas_all = []
                for i in range(len(anchor_indices)):
                    start = anchor_indices[i]
                    end = anchor_indices[i+1] if i+1 < len(anchor_indices) else s
                    if start + 1 < end:
                        deltas_all.append(data_flat[start+1:end] - data_flat[start])
                
                if not deltas_all:
                    continue
                    
                stacked_deltas = torch.cat(deltas_all, dim=0)
                
                if mode == "INT8-DKV":
                    # Quantize all deltas
                    q = quantize_int8(stacked_deltas)
                    recon_deltas = dequantize_int8(q)
                    # Reconstruct
                    curr_pos = 0
                    for i in range(len(anchor_indices)):
                        start = anchor_indices[i]
                        end = anchor_indices[i+1] if i+1 < len(anchor_indices) else s
                        if start + 1 < end:
                            n = end - (start + 1)
                            d_recon = recon_deltas[curr_pos:curr_pos+n]
                            recon_target[0, :, start+1:end, :] = (data_flat[start] + d_recon).reshape(n, h, d).transpose(0, 1)
                            curr_pos += n
                    layer_comp_bytes += stacked_deltas.numel() * 1 + len(anchor_indices) * 4 # INT8 + scales
                    
                else:
                    # Shared Basis modes
                    rank = 16
                    if "Rank" in mode:
                        try:
                            # Extract rank from "Layer-Shared Rank8" -> 8
                            rank = int(mode.split("Rank")[-1])
                        except:
                            rank = 16
                    
                    sparse_ratio = 0.0
                    adaptive = False
                    
                    if mode == "Adaptive-Only":
                        adaptive = True
                    elif "Hybrid-S" in mode:
                        adaptive = True
                        sparse_ratio = float(mode.split("Hybrid-S")[-1].replace("%", "")) / 100.0
                    
                    # Audit: Verify configuration
                    if layer_idx == 0 and component_name == "K":
                        print(f"[Verification] Mode: {mode} | Base Rank: {rank} | Adaptive: {adaptive} | Sparse: {sparse_ratio:.1%}")

                    basis_id = f"L{layer_idx}_{component_name}_{mode}"
                    basis = self.sb_manager.create_basis(stacked_deltas, 64 if adaptive else rank, basis_id)
                    layer_comp_bytes += basis.nbytes()
                    
                    curr_pos = 0
                    for i in range(len(anchor_indices)):
                        start = anchor_indices[i]
                        end = anchor_indices[i+1] if i+1 < len(anchor_indices) else s
                        if start + 1 < end:
                            n = end - (start + 1)
                            d_block = stacked_deltas[curr_pos:curr_pos+n]
                            
                            target_rank = rank
                            if adaptive:
                                target_rank = self.rank_selector.select_rank(d_block)
                            
                            sbd = self.sb_manager.compress_block(
                                d_block, basis_id, rank=target_rank, sparse_ratio=sparse_ratio
                            )
                            
                            # Verification: check if repair tensors exist
                            if sbd.sparse_indices is not None and i == 0:
                                print(f"[Verification] Layer {layer_idx} block {i}: Sparse repair active ({sbd.sparse_indices.numel()} tokens)")
                            
                            layer_comp_bytes += sbd.nbytes()
                            
                            d_recon = self.sb_manager.decompress_block(sbd).to(self.device)
                            
                            # Check for NaNs or INF in reconstruction
                            if torch.isnan(d_recon).any():
                                print(f"[CRITICAL] NaN in reconstruction at Layer {layer_idx} {component_name}")
                                
                            recon_target[0, :, start+1:end, :] = (data_flat[start] + d_recon).reshape(n, h, d).transpose(0, 1)
                            curr_pos += n

            recon_kv_list.append((k_recon.half().to(self.device), v_recon.half().to(self.device)) + layer_data[2:])
            total_comp_bytes += layer_comp_bytes
            
        # Re-pack
        if is_cache_obj:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for i, layer_data in enumerate(recon_kv_list):
                # DynamicCache.update takes k, v
                new_cache.update(layer_data[0], layer_data[1], layer_idx=i)
            ratio = total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0
            return new_cache, {"ratio": ratio, "bytes": total_comp_bytes}
        else:
            ratio = total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0
            return tuple(recon_kv_list), {"ratio": ratio, "bytes": total_comp_bytes}

    @torch.no_grad()
    def evaluate_perplexity(self, samples, mode, interval=64):
        total_nll = 0
        total_tokens = 0
        
        pbar = tqdm(samples, desc=f"Eval {mode}")
        for input_ids in pbar:
            input_ids = input_ids.to(self.device)
            # 1. Prefill (first half)
            prefill_len = input_ids.size(1) // 2
            eval_len = input_ids.size(1) - prefill_len
            
            prefill_ids = input_ids[:, :prefill_len]
            eval_ids = input_ids[:, prefill_len:]
            
            outputs = self.model(prefill_ids, use_cache=True)
            past_kv = outputs.past_key_values
            
            # 2. Compress & Reconstruct
            past_kv_recon, stats = self.compress_kv(past_kv, mode, interval=interval)
            
            # 3. Evaluate on eval_ids
            outputs = self.model(eval_ids, past_key_values=past_kv_recon, use_cache=True)
            logits = outputs.logits # [batch, eval_len, vocab]
            
            # Calculate PPL
            # We shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = eval_ids[..., 1:].contiguous()
            
            # Flatten
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            
            total_nll += loss.item()
            total_tokens += shift_labels.numel()
            
            pbar.set_postfix({"ppl": math.exp(total_nll / total_tokens) if total_tokens > 0 else 0})

        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)
        return ppl, stats

def run_ppl_benchmark(model_id="Qwen/Qwen2-0.5B", n_samples=3):
    evaluator = Phase8PerplexityEvaluator(model_id=model_id)
    samples = evaluator.get_wiki_data(n_samples=n_samples)
    
    modes = [
        "FP16",
        "INT8-DKV",
        "Layer-Shared Rank16",
        "Adaptive-Only",
        "Hybrid-S1%",
        "Hybrid-S5%",
        "Hybrid-S10%"
    ]
    
    results = {}
    for mode in modes:
        ppl, stats = evaluator.evaluate_perplexity(samples, mode)
        results[mode] = {
            "perplexity": ppl,
            "compression_ratio": stats["ratio"],
            "mem_bytes": stats["bytes"]
        }
        print(f"Mode: {mode:20} | PPL: {ppl:8.4f} | Ratio: {stats['ratio']:6.2f}x")
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/phase8/perplexity.json")
    args = parser.parse_args()
    
    res = run_ppl_benchmark(model_id=args.model, n_samples=args.samples)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
