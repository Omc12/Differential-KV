import os
import sys
import json
import torch
import numpy as np

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_runtime_dir = os.path.dirname(_script_dir)
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# FORCE DIFFKV TO ENGAGE FOR SHORT SEQUENCES (Defaults to 4096!)
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "1"

# Import PyTorchDiffKVHFWrapper directly to bypass the dynamic MLX redirect
from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper
from native_core.mac_utils import get_best_device

def serialize_srl_graph(wrapper, session_id):
    manager = wrapper.manager
    srl_state = manager.get_srl_state(session_id)
    
    # Print diagnostics
    print(f"\n--- serialize_srl_graph diagnostics for {session_id} ---")
    blocks = manager.get_streaming_blocks(session_id, 0)
    print(f"Number of blocks in layer 0: {len(blocks)}")
    for i, b in enumerate(blocks):
        print(f"  Block {i}: anchor_idx={b.anchor_idx}, state={getattr(b, 'state', 'NONE')}, pool_idx={getattr(b, 'pool_idx', 'NONE')}")
    print(f"Pending CPU blocks: {getattr(manager, '_pending_cpu_blocks', 0)}")
    print(f"Manager active session srl keys: {list(getattr(manager, '_session_srl', {}).keys())}")
    
    if srl_state is None:
        print("SRL state not found in manager. Attempting explicit finalize_srl_index...")
        try:
            manager.finalize_srl_index(session_id)
            srl_state = manager.get_srl_state(session_id)
            print(f"SRL state after explicit finalize: {srl_state is not None}")
        except Exception as e:
            print(f"Error in finalize_srl_index: {e}")
            
    if srl_state is None:
        return None
        
    tokenizer = wrapper.tokenizer
    
    # Get session token IDs safely
    token_ids_cpu = manager._session_token_ids.get(session_id)
    if token_ids_cpu is None and hasattr(wrapper, "_session_token_ids"):
        token_ids_cpu = wrapper._session_token_ids.get(session_id)
    
    if token_ids_cpu is None:
        print(f"Error: Token IDs not found for session {session_id}")
        return None
        
    if isinstance(token_ids_cpu, list):
        token_ids_cpu = torch.tensor(token_ids_cpu)
        
    mbs = manager.get_session_micro_block_size(session_id)
    index_block_size = mbs + 1
    
    # 1. Nodes
    nodes = []
    slot_ids = srl_state.ordered_slot_ids
    
    role_names = {0: "outer", 1: "around", 2: "center"}
    
    for i, slot_id in enumerate(slot_ids):
        # Decode text for this block
        start_idx = i * index_block_size
        end_idx = min(start_idx + index_block_size, len(token_ids_cpu))
        block_tokens = token_ids_cpu[start_idx:end_idx].tolist()
        block_text = tokenizer.decode(block_tokens)
        
        # Proximity/Cluster Role mapping
        role_val = -1
        if srl_state.chunk_graph.role_mapping_tensor is not None and slot_id < srl_state.chunk_graph.role_mapping_tensor.shape[0]:
            role_val = int(srl_state.chunk_graph.role_mapping_tensor[slot_id].item())
        role_name = role_names.get(role_val, "unknown")
        
        # Parent mapping
        parent_id = -1
        if srl_state.chunk_graph.slot_to_parent_tensor is not None and slot_id < srl_state.chunk_graph.slot_to_parent_tensor.shape[0]:
            parent_id = int(srl_state.chunk_graph.slot_to_parent_tensor[slot_id].item())
            
        # Center mapping
        center_id = -1
        if srl_state.chunk_graph.slot_to_center_tensor is not None and slot_id < srl_state.chunk_graph.slot_to_center_tensor.shape[0]:
            center_id = int(srl_state.chunk_graph.slot_to_center_tensor[slot_id].item())
            
        is_sink = slot_id in srl_state.sink_blocks
        is_dynamic_anchor = slot_id in srl_state.dynamic_anchors
        
        segment_id = srl_state.segment_ids.get(slot_id, 0)
        
        # Find corresponding block in manager streaming blocks to extract metadata
        block_obj = None
        for b in blocks:
            if getattr(b, 'pool_idx', None) == slot_id:
                block_obj = b
                break
                
        block_meta = {}
        kv_summary = {}
        if block_obj is not None:
            block_meta = {
                "scale": float(getattr(block_obj, "scale", 1.0)),
                "cosine_sim": float(getattr(block_obj, "cosine_sim", 1.0)),
                "norm_drift": float(getattr(block_obj, "norm_drift", 0.0)),
                "dynamic_rank": int(getattr(block_obj, "dynamic_rank", -1)),
                "is_outlier": bool(getattr(block_obj, "is_outlier", False)),
                "state": str(getattr(block_obj, "state", "NONE")),
                "anchor_idx": int(getattr(block_obj, "anchor_idx", -1)),
                "dirty": bool(getattr(block_obj, "dirty", True)),
                "skip_compression": bool(getattr(block_obj, "skip_compression", False))
            }
            if block_obj.state == "COMPRESSED" and block_obj.U_cpu is not None:
                kv_summary = {
                    "u_shape": list(block_obj.U_cpu.shape),
                    "v_shape": list(block_obj.V_cpu.shape),
                    "u_norm": float(block_obj.U_cpu.norm().item()),
                    "v_norm": float(block_obj.V_cpu.norm().item()),
                }
            elif block_obj.active_k_cpu is not None:
                kv_summary = {
                    "k_shape": list(block_obj.active_k_cpu.shape),
                    "v_shape": list(block_obj.active_v_cpu.shape),
                    "k_norm": float(block_obj.active_k_cpu.norm().item()),
                    "v_norm": float(block_obj.active_v_cpu.norm().item()),
                }
        
        # Fetch L2-normalized semantic descriptor vector
        descriptor = []
        if srl_state.semantic_index is not None and srl_state.semantic_index.desc_matrix is not None:
            try:
                row_idx = srl_state.semantic_index.slot_to_idx(slot_id)
                if row_idx != -1:
                    descriptor = srl_state.semantic_index.desc_matrix[row_idx].tolist()
            except Exception:
                pass
        
        nodes.append({
            "slot_id": int(slot_id),
            "block_index": i,
            "text": block_text,
            "tokens": block_tokens,
            "role": role_name,
            "parent_id": parent_id,
            "center_id": center_id,
            "is_sink": is_sink,
            "is_dynamic_anchor": is_dynamic_anchor,
            "segment_id": segment_id,
            "block_metadata": block_meta,
            "kv_summary": kv_summary,
            "descriptor": descriptor
        })
        
    # 2. Links
    links = []
    
    # Adjacency neighbors from chunk_graph
    neighbors_tensor = srl_state.chunk_graph.neighbors
    weights_tensor = srl_state.chunk_graph.weights
    
    if neighbors_tensor is not None:
        for i, slot_i in enumerate(slot_ids):
            row_neighbors = neighbors_tensor[i].tolist()
            row_weights = weights_tensor[i].tolist() if weights_tensor is not None else [1.0] * len(row_neighbors)
            for nb_idx, nb_row in enumerate(row_neighbors):
                if nb_row == -1 or nb_row >= len(slot_ids):
                    continue
                slot_j = slot_ids[nb_row]
                weight = row_weights[nb_idx]
                links.append({
                    "source": int(slot_i),
                    "target": int(slot_j),
                    "weight": float(weight),
                    "type": "neighborhood"
                })
                
    # Prime neighbors (inter-cluster)
    prime_neighbors = srl_state.chunk_graph.prime_neighbors
    prime_weights = srl_state.chunk_graph.prime_weights
    if prime_neighbors is not None:
        for slot in range(prime_neighbors.shape[0]):
            for k in range(prime_neighbors.shape[1]):
                target_slot = int(prime_neighbors[slot, k].item())
                if target_slot != -1:
                    weight = float(prime_weights[slot, k].item()) if prime_weights is not None else 1.0
                    links.append({
                        "source": int(slot),
                        "target": int(target_slot),
                        "weight": weight,
                        "type": "prime"
                    })
                    
    # Parent-child links
    if srl_state.chunk_graph.slot_to_parent_tensor is not None:
        for child_slot in range(srl_state.chunk_graph.slot_to_parent_tensor.shape[0]):
            parent_slot = int(srl_state.chunk_graph.slot_to_parent_tensor[child_slot].item())
            if parent_slot != -1:
                links.append({
                    "source": int(parent_slot),
                    "target": int(child_slot),
                    "weight": 1.0,
                    "type": "parent-child"
                })
                
    # 2b. Factual exact store entries
    factual_entries = []
    factual_store = manager._factual_stores.get(session_id)
    if factual_store is not None and hasattr(factual_store, "entries"):
        for entry in factual_store.entries:
            decoded_text = tokenizer.decode(entry.tokens) if entry.tokens else ""
            dist_tok_text = tokenizer.decode([entry.distinguishing_token]) if entry.distinguishing_token is not None else None
            prefix_text = tokenizer.decode(entry.prefix_tokens) if entry.prefix_tokens else ""
            triple_seqs_text = [tokenizer.decode(seq) for seq in entry.triple_sequences] if entry.triple_sequences else []
            
            k_norm = float(entry.K.norm().item()) if entry.K is not None else 0.0
            v_norm = float(entry.V.norm().item()) if entry.V is not None else 0.0
            k_shape = list(entry.K.shape) if entry.K is not None else []
            v_shape = list(entry.V.shape) if entry.V is not None else []
            
            factual_entries.append({
                "start_idx": entry.start_idx,
                "end_idx": entry.end_idx,
                "text": decoded_text,
                "tokens": entry.tokens,
                "is_prime": entry.is_prime,
                "entity_id": entry.entity_id,
                "distinguishing_token": entry.distinguishing_token,
                "distinguishing_token_text": dist_tok_text,
                "prefix_text": prefix_text,
                "prefix_tokens": entry.prefix_tokens if getattr(entry, "prefix_tokens", None) is not None else [],
                "triple_sequences_text": triple_seqs_text,
                "triple_sequences": entry.triple_sequences if getattr(entry, "triple_sequences", None) is not None else [],
                "is_definition": entry.is_definition,
                "slot_ids": entry.slot_ids,
                "neighbors": entry.neighbors,
                "weights": entry.weights,
                "entity_signature": entry.entity_sig.tolist() if getattr(entry, "entity_sig", None) is not None else [],
                "descriptor": entry.descriptor.tolist() if getattr(entry, "descriptor", None) is not None else [],
                "kv_summary": {
                    "k_shape": k_shape,
                    "v_shape": v_shape,
                    "k_norm": k_norm,
                    "v_norm": v_norm
                }
            })

    # 3. Metadata
    cluster_centers = []
    if srl_state.chunk_graph is not None and srl_state.chunk_graph.cluster_centers_tensor is not None:
        cluster_centers = srl_state.chunk_graph.cluster_centers_tensor.tolist()
        
    parent_landmarks = []
    if srl_state.chunk_graph is not None and srl_state.chunk_graph.parent_landmarks is not None:
        parent_landmarks = srl_state.chunk_graph.parent_landmarks.tolist()

    recent_decode_keys = []
    if srl_state.recent_decode_keys:
        for k in srl_state.recent_decode_keys:
            recent_decode_keys.append(k.tolist())

    prompt_eagle_scores = []
    if srl_state.prompt_eagle_scores is not None:
        prompt_eagle_scores = srl_state.prompt_eagle_scores.tolist()

    parent_to_children = {}
    if srl_state.chunk_graph is not None and hasattr(srl_state.chunk_graph, "parent_to_children"):
        try:
            for p, children in srl_state.chunk_graph.parent_to_children.items():
                parent_to_children[int(p)] = [int(c) for c in children]
        except Exception:
            pass

    metadata = {
        "session_id": session_id,
        "sink_blocks": srl_state.sink_blocks,
        "dynamic_anchors": srl_state.dynamic_anchors,
        "k_min": srl_state.k_min,
        "k_max": srl_state.k_max,
        "routing_threshold": srl_state.routing_threshold,
        "srl_age_penalty": srl_state.srl_age_penalty,
        
        # Session statistics & adaptive tuning fields
        "recent_miss_rate": float(getattr(srl_state, "recent_miss_rate", 0.0)),
        "k_multiplier": float(getattr(srl_state, "k_multiplier", 1.0)),
        "call_count": int(getattr(srl_state, "call_count", 0)),
        "recent_generated_tokens": getattr(srl_state, "recent_generated_tokens", []),
        
        # Structured Attention Segmenting (SAS) fields
        "segment_ids": srl_state.segment_ids,
        "current_query_segment_id": int(getattr(srl_state, "current_query_segment_id", 0)),
        "concept_tok_1": int(getattr(srl_state, "concept_tok_1", -1)),
        "concept_tok_1_text": tokenizer.decode([srl_state.concept_tok_1]) if getattr(srl_state, "concept_tok_1", -1) != -1 else None,
        "concept_tok_2": int(getattr(srl_state, "concept_tok_2", -1)),
        "concept_tok_2_text": tokenizer.decode([srl_state.concept_tok_2]) if getattr(srl_state, "concept_tok_2", -1) != -1 else None,
        
        # EQA-DR fields
        "prompt_anchors": getattr(srl_state, "prompt_anchors", []),
        "generated_token_slots": getattr(srl_state, "generated_token_slots", []),
        "prompt_eagle_scores": prompt_eagle_scores,
        "recent_decode_keys": recent_decode_keys,
        
        # ChunkGraph hierarchy fields
        "cluster_centers": cluster_centers,
        "parent_to_children": parent_to_children,
        "parent_landmarks": parent_landmarks
    }
    
    # 2c. Lexical Inverted Index
    inverted_index_dict = {}
    if srl_state.inverted_index is not None and getattr(srl_state.inverted_index, "index", None) is not None:
        for tid, slots in srl_state.inverted_index.index.items():
            text = tokenizer.decode([tid]) if tokenizer is not None else ""
            inverted_index_dict[int(tid)] = {
                "text": text,
                "slots": [int(s) for s in slots]
            }

    return {
        "metadata": metadata,
        "nodes": nodes,
        "links": links,
        "factual_entries": factual_entries,
        "inverted_index": inverted_index_dict
    }

def main():
    device = get_best_device()
    print(f"Device: {device}")
    
    # Initialize the tiny model
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading PyTorch wrapper for {MODEL}...")
    
    # We set rank=16 and micro_block_size=32 to get a highly detailed graph for the content.
    wrapper = PyTorchDiffKVHFWrapper(
        MODEL,
        config={
            "rank": 16,
            "micro_block_size": 32,
            "block_size": 32,
            "serving_mode": "balanced",
            "local_files_only": True
        },
        device=device
    )
    
    # Configure the streaming manager to compress blocks with a much smaller recency window,
    # so we get a detailed chunk graph even for short prompts.
    if wrapper.manager._streaming_mgr is not None:
        wrapper.manager._streaming_mgr.recency_window = 32
        wrapper.manager._streaming_mgr.short_context_threshold = 0
        wrapper.manager._streaming_mgr.protect_block_zero = True
        wrapper.manager._streaming_mgr._should_skip_compression = lambda *args, **kwargs: False
        
        from native_core.streaming_sparse_ingest import StreamingKVBlock
        StreamingKVBlock.short_context_threshold = 0
        StreamingKVBlock.protect_block_zero = True
        print("Configured Streaming Manager overrides: recency_window=32, protect_block_zero=True, skip_compression=False")
    
    session_id = "graph-tracking-session"
    wrapper.active_session = session_id
    
    content = """Degeneracies occur when two eigenvalues become equal, but not all degeneracies are alike. In Hermitian systems, a degeneracy does not destroy the independence of eigenvectors: although the eigenvalues coincide, the eigenvectors remain distinct and can still be chosen orthogonal. Such degeneracies are often represented geometrically by conical intersections, also called diabolical points, where two eigenvalue sheets touch without merging. The number of independent parameters that must be tuned to create a degeneracy is called its codimension. For real symmetric matrices this codimension is two, while for complex Hermitian matrices it is three, reflecting the additional constraint required to force the eigenvalues to coincide.

Non-Hermitian systems exhibit a qualitatively different phenomenon known as an exceptional point. Here the degeneracy is stronger: not only do the eigenvalues become equal, but the eigenvectors themselves coalesce into a single state, causing the matrix to become defective and lose a complete eigenbasis. Exceptional points typically have codimension two and possess a square-root branch-point topology. The eigenvalue surfaces therefore form two connected Riemann sheets rather than a simple double cone. Encircling an exceptional point once exchanges the eigenvalues and associated states, so two loops are required to return to the original branch. Because non-Hermitian operators are not generally equal to their adjoints, left and right eigenvectors must be treated separately; near an exceptional point they exhibit biorthogonal behavior and can become self-orthogonal."""

    question = """Using only the information above, answer the following:

1. Define codimension and state the codimensions of:

   * real symmetric degeneracies,
   * complex Hermitian degeneracies,
   * exceptional points.

2. Compare the topology of eigenvalue surfaces near:

   * a diabolical point,
   * an exceptional point.

3. Explain the difference between:

   * eigenvalue degeneracy,
   * eigenvector coalescence.

4. Why are exceptional points associated with Riemann sheets whereas diabolical points are associated with conical intersections?

5. What happens after one closed loop around:

   * a Hermitian degeneracy,
   * an exceptional point?

6. Why are left and right eigenvectors important in non-Hermitian systems but not in Hermitian ones?"""

    # Use tokenizer to apply template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    prompt1 = wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\n--- Running Prompt 1 (Content Ingestion) ---")
    response1 = wrapper.generate(prompt1, max_new_tokens=10, temperature=0.0)
    print(f"Response 1: {response1}")
    
    print("Extracting Graph 1...")
    graph1 = serialize_srl_graph(wrapper, session_id)
    
    if graph1:
        out_path1 = os.path.join(os.path.dirname(_runtime_dir), "graph_turn1.json")
        with open(out_path1, "w") as f:
            json.dump(graph1, f, indent=2)
        print(f"Graph 1 successfully written to: {out_path1}")
        print(f"Number of nodes: {len(graph1['nodes'])}")
        print(f"Number of links: {len(graph1['links'])}")
        print(f"Number of factual store entries: {len(graph1.get('factual_entries', []))}")
        
    # Construct second turn prompt incorporating response1 (extracting assistant reply only)
    assistant_reply = response1.split("assistant")[-1].strip()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": question}
    ]
    prompt2 = wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\n--- Running Prompt 2 (Question Answering) ---")
    response2 = wrapper.generate(prompt2, max_new_tokens=256, temperature=0.0)
    print(f"Response 2:\n{response2}")
    
    print("Extracting Graph 2...")
    graph2 = serialize_srl_graph(wrapper, session_id)
    
    if graph2:
        out_path2 = os.path.join(os.path.dirname(_runtime_dir), "graph_turn2.json")
        with open(out_path2, "w") as f:
            json.dump(graph2, f, indent=2)
        print(f"Graph 2 successfully written to: {out_path2}")
        print(f"Number of nodes: {len(graph2['nodes'])}")
        print(f"Number of links: {len(graph2['links'])}")
        print(f"Number of factual store entries: {len(graph2.get('factual_entries', []))}")

    wrapper.close()

if __name__ == "__main__":
    main()
