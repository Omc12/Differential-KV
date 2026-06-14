import json
import os

def main():
    root = "/Users/omchimurkar1/Desktop/Differential-KV"
    with open(os.path.join(root, "graph_turn2.json"), "r") as f:
        g = json.load(f)
        
    nodes = {node["slot_id"]: node for node in g["nodes"]}
    links = g["links"]
    
    print("=== Nodes containing keywords ===")
    keywords = ["riemann", "diabolical", "exceptional", "unitary", "eigenvector", "loop", "cone"]
    kw_nodes = []
    for slot_id, node in nodes.items():
        text = node["text"].lower()
        matched = [kw for kw in keywords if kw in text]
        if matched:
            print(f"Slot {slot_id} (block {node['block_index']}, parent {node['parent_id']}, center {node['center_id']}):")
            print(f"  Matched: {matched}")
            print(f"  Role: {node['role']}")
            print(f"  Text: {repr(node['text'][:120])}")
            kw_nodes.append(slot_id)
            
    print("\n=== Parent-Child Mapping ===")
    parent_to_children = {}
    for slot_id, node in nodes.items():
        p = node["parent_id"]
        if p != -1:
            parent_to_children.setdefault(p, []).append(slot_id)
    for p, children in parent_to_children.items():
        p_text = repr(nodes[p]['text'][:80]) if p in nodes else "MISSING NODE"
        print(f"Parent Slot {p} ({p_text}) -> Children: {children}")

    print("\n=== Links connecting these slots ===")
    for link in links:
        src_id = link["source"]
        tgt_id = link["target"]
        if src_id in kw_nodes or tgt_id in kw_nodes:
            src = nodes.get(src_id)
            tgt = nodes.get(tgt_id)
            if src and tgt:
                if link["type"] == "parent-child" or link["weight"] > 0.1:
                    print(f"[{link['type']}] Slot {src_id} ({repr(src['text'][:50])}) -> Slot {tgt_id} ({repr(tgt['text'][:50])}) weight={link['weight']:.3f}")
            else:
                print(f"[{link['type']}] Slot {src_id} (exists={src_id in nodes}) -> Slot {tgt_id} (exists={tgt_id in nodes}) weight={link['weight']:.3f}")

if __name__ == "__main__":
    main()
