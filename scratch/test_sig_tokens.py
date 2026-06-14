import json
import os

def main():
    root = "/Users/omchimurkar1/Desktop/Differential-KV"
    with open(os.path.join(root, "graph_turn1.json"), "r") as f:
        g = json.load(f)
        
    inv_index = g.get("inverted_index", {})
    nodes = g["nodes"]
    
    # Let us see what vocabs and sig_tokens would be computed
    for node in nodes:
        slot_id = node["slot_id"]
        # Find tokens in this slot from inverted_index
        vocab_tids = []
        for tid_str, index_entry in inv_index.items():
            if slot_id in index_entry["slots"]:
                vocab_tids.append(int(tid_str))
                
        # In chunk_graph.py, sig_token is the token with the highest IDF in the slot vocab
        # But wait, how idf is computed? Let us assume idf is from g["metadata"] or we can look it up.
        # In g["metadata"], do we have idfs?
        # Actually, let us just print the slot id, text, and its vocabulary.
        print(f"Slot {slot_id} | Text: {repr(node['text'][:60])}")
        print(f"  Vocab tokens: {[t for t in vocab_tids][:10]}")

if __name__ == "__main__":
    main()
