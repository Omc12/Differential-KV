import torch
import math
from typing import List, Dict, Optional, Set, Any

class FactEntry:
    def __init__(self, start_idx: int, end_idx: int, K: torch.Tensor, V: torch.Tensor, descriptor: torch.Tensor):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.K = K              # [layer, heads, len, D] (on CPU or GPU)
        self.V = V              # [layer, heads, len, D]
        self.descriptor = descriptor  # [DESC_DIM] (on CPU)

class FactualExactStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.entries: List[FactEntry] = []
        
    def build(self, prefill_kv: Dict[int, List[torch.Tensor]], token_ids: torch.Tensor, W_proj: torch.Tensor, stop_token_ids: Set[int]):
        """
        Identify rare content words and group them into factual spans.
        prefill_kv: Dict[layer_idx, [K_cpu, V_cpu]]
          K_cpu/V_cpu shape: [1, kv_heads, total_seq_len, head_dim]
        token_ids: [total_seq_len] cpu
        W_proj: [DESC_DIM, head_dim]
        stop_token_ids: set of common token IDs
        """
        if not prefill_kv or token_ids is None or token_ids.numel() == 0:
            return
            
        total_seq_len = token_ids.numel()
        layers = sorted(list(prefill_kv.keys()))
        
        # 1. Identify rare content words (not in stop words)
        rare_mask = torch.zeros(total_seq_len, dtype=torch.bool)
        for i in range(total_seq_len):
            tid = int(token_ids[i].item())
            if tid not in stop_token_ids and tid > 0:
                rare_mask[i] = True
                
        # Group contiguous rare content words into spans of max length 8
        spans = []
        in_span = False
        start = -1
        for i in range(total_seq_len):
            if rare_mask[i]:
                if not in_span:
                    start = i
                    in_span = True
            else:
                if in_span:
                    spans.append((start, i))
                    in_span = False
        if in_span:
            spans.append((start, total_seq_len))
            
        chunked_spans = []
        for s, e in spans:
            for sub_s in range(s, e, 8):
                sub_e = min(sub_s + 8, e)
                chunked_spans.append((sub_s, sub_e))
                
        # 2. Extract verbatim KV sequences across all layers for each span
        for s, e in chunked_spans:
            span_len = e - s
            if span_len <= 0:
                continue
                
            span_K_list = []
            span_V_list = []
            
            for layer in layers:
                K_layer, V_layer = prefill_kv[layer]
                # K_layer/V_layer: [1, kv_heads, total_seq_len, head_dim]
                span_K_list.append(K_layer[0, :, s:e, :].clone())
                span_V_list.append(V_layer[0, :, s:e, :].clone())
                
            span_K = torch.stack(span_K_list, dim=0) # [num_layers, kv_heads, span_len, head_dim]
            span_V = torch.stack(span_V_list, dim=0) # [num_layers, kv_heads, span_len, head_dim]
            
            # Compute descriptor for the span using layer 0 average key
            avg_k = span_K[0].mean(dim=(0, 1)).float() # [head_dim]
            if W_proj is not None:
                desc = avg_k.to(W_proj.device) @ W_proj.T # [DESC_DIM]
                desc = desc / (desc.norm() + 1e-8)
            else:
                desc = torch.zeros(64)
                
            self.entries.append(FactEntry(
                start_idx=s,
                end_idx=e,
                K=span_K,
                V=span_V,
                descriptor=desc.cpu()
            ))
            
    def query(self, Q: torch.Tensor, W_proj: torch.Tensor, threshold: float = 0.4) -> List[FactEntry]:
        """
        Query the factual store.
        Q: [H_q, D]
        """
        if not self.entries or W_proj is None:
            return []
            
        avg_q = Q.mean(dim=0).float()
        q_desc = avg_q @ W_proj.T # [DESC_DIM]
        q_desc = q_desc / (q_desc.norm() + 1e-8)
        q_desc_cpu = q_desc.cpu()
        
        matches = []
        for entry in self.entries:
            sim = torch.dot(q_desc_cpu, entry.descriptor).item()
            if sim >= threshold:
                matches.append((entry, sim))
                
        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in matches[:3]]
