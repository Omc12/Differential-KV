import json
import os

def load_graph(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def analyze_graph(name, graph):
    if not graph:
        return
    print(f"\n=================== ANALYZING {name} ===================")
    metadata = graph.get("metadata", {})
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    factual_entries = graph.get("factual_entries", [])
    
    print(f"Nodes: {len(nodes)}")
    print(f"Links: {len(links)}")
    print(f"Factual Entries: {len(factual_entries)}")
    
    print("\n--- Prime Nodes / Entities ---")
    primes = [n for n in nodes if n.get("is_sink") or n.get("is_dynamic_anchor")]
    print(f"Total sinks/anchors: {len(primes)}")
    for p in primes:
        print(f"  Slot {p.get('slot_id')}: Block {p.get('block_index')} | is_sink: {p.get('is_sink')} | is_anchor: {p.get('is_dynamic_anchor')} | Text: {repr(p.get('text')[:100])}")
        
    print("\n--- Factual Entries ---")
    for idx, fe in enumerate(factual_entries):
        print(f"  Entry {idx}: is_prime: {fe.get('is_prime')} | entity_id: {fe.get('entity_id')} | distinguishing_token_text: {fe.get('distinguishing_token_text')} | Text: {repr(fe.get('text')[:120])}")
        
    # Link distribution by type
    link_types = {}
    for l in links:
        t = l.get("type", "unknown")
        link_types[t] = link_types.get(t, 0) + 1
    print("\n--- Link types distribution ---")
    for t, count in link_types.items():
        print(f"  {t}: {count}")

def main():
    root = "/Users/omchimurkar1/Desktop/Differential-KV"
    g1_path = os.path.join(root, "graph_turn1.json")
    g2_path = os.path.join(root, "graph_turn2.json")
    
    g1 = load_graph(g1_path)
    g2 = load_graph(g2_path)
    
    analyze_graph("Graph Turn 1 (Ingestion)", g1)
    analyze_graph("Graph Turn 2 (Question Answering)", g2)

if __name__ == "__main__":
    main()
