import json
import os

def print_graph(path, title):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, "r") as f:
        g = json.load(f)
    print(f"\n=================== {title} ===================")
    print("METADATA:")
    for k, v in g.get("metadata", {}).items():
        if k in ["sink_blocks", "dynamic_anchors", "segment_ids", "concept_tok_1_text", "concept_tok_2_text"]:
            print(f"  {k}: {v}")
    
    print("\nNODES:")
    for n in g.get("nodes", []):
        print(f"  Slot {n['slot_id']} (Block {n['block_index']}): Segment={n.get('segment_id', 'N/A')} | Role={n.get('role')} | is_sink={n.get('is_sink')} | Text={repr(n['text'][:120])}")
        
    print("\nLINKS:")
    for l in g.get("links", []):
        print(f"  {l['source']} -> {l['target']} (weight={l.get('weight', 0.0):.3f}, type={l.get('type')})")

def main():
    print_graph("graph_turn1.json", "Graph Turn 1 (Content Prefill)")
    print_graph("graph_turn2.json", "Graph Turn 2 (Question Answering)")

if __name__ == "__main__":
    main()
