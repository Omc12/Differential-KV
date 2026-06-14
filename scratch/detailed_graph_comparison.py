import json
import os

def main():
    root = "/Users/omchimurkar1/Desktop/Differential-KV"
    with open(os.path.join(root, "graph_turn1.json"), "r") as f:
        g1 = json.load(f)
    with open(os.path.join(root, "graph_turn2.json"), "r") as f:
        g2 = json.load(f)
        
    g1_nodes = {node["slot_id"]: node for node in g1["nodes"]}
    g2_nodes = {node["slot_id"]: node for node in g2["nodes"]}
    
    print("=== TURN 1 NODES ===")
    for slot_id, node in g1_nodes.items():
        print(f"Slot {slot_id}: {repr(node['text'][:80])}")
        
    print("\n=== TURN 2 NODES ===")
    for slot_id, node in g2_nodes.items():
        print(f"Slot {slot_id}: {repr(node['text'][:80])}")
        
    # Analyze links that contain keywords
    print("\n=== TURN 1 KEYWORD LINKS ===")
    for link in g1["links"]:
        src_id = link["source"]
        tgt_id = link["target"]
        if src_id in g1_nodes and tgt_id in g1_nodes:
            src = g1_nodes[src_id]["text"]
            tgt = g1_nodes[tgt_id]["text"]
            if ("symmetric" in src or "codimension" in src) and ("exceptional" in tgt or "Riemann" in tgt):
                print(f"Link {src_id} -> {tgt_id} (weight={link['weight']:.3f}):")
                print(f"  Src: {repr(src[:80])}")
                print(f"  Tgt: {repr(tgt[:80])}")

    print("\n=== TURN 2 KEYWORD LINKS ===")
    for link in g2["links"]:
        src_id = link["source"]
        tgt_id = link["target"]
        if src_id in g2_nodes and tgt_id in g2_nodes:
            src = g2_nodes[src_id]["text"]
            tgt = g2_nodes[tgt_id]["text"]
            if ("symmetric" in src or "codimension" in src) and ("exceptional" in tgt or "Riemann" in tgt):
                print(f"Link {src_id} -> {tgt_id} (weight={link['weight']:.3f}):")
                print(f"  Src: {repr(src[:80])}")
                print(f"  Tgt: {repr(tgt[:80])}")

if __name__ == "__main__":
    main()
