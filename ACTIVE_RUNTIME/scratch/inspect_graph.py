import json
import os

def inspect_graph(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, "r") as f:
        data = json.load(f)
        
    print(f"\n==========================================")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"==========================================")
    
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    factual_entries = data.get("factual_entries", [])
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Total links: {len(links)}")
    print(f"Total factual entries: {len(factual_entries)}")
    
    print("\n--- NODES ---")
    for idx, node in enumerate(nodes[:25]):
        print(f"Node {idx} (Slot {node.get('slot_id')}): text={repr(node.get('text'))} role={node.get('role_name')} is_sink={node.get('is_sink')} is_dyn={node.get('is_dynamic_anchor')}")
        
    print("\n--- LINKS ---")
    # Print links sorted by weight desc
    sorted_links = sorted(links, key=lambda x: x.get("weight", 0.0), reverse=True)
    for idx, link in enumerate(sorted_links[:30]):
        source = link.get("source")
        target = link.get("target")
        weight = link.get("weight")
        link_type = link.get("type")
        
        # Resolve source/target text
        src_text = "UNKNOWN"
        tgt_text = "UNKNOWN"
        for n in nodes:
            if n.get("slot_id") == source:
                src_text = n.get("text")
            if n.get("slot_id") == target:
                tgt_text = n.get("text")
                
        print(f"Link {idx}: {repr(src_text)} (Slot {source}) -> {repr(tgt_text)} (Slot {target}) weight={weight:.4f} type={link_type}")

if __name__ == "__main__":
    inspect_graph("/Users/omchimurkar1/Desktop/Differential-KV/graph_turn1.json")
    inspect_graph("/Users/omchimurkar1/Desktop/Differential-KV/graph_turn2.json")
