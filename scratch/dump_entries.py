import json
import os

def main():
    root = "/Users/omchimurkar1/Desktop/Differential-KV"
    with open(os.path.join(root, "graph_turn2.json"), "r") as f:
        g = json.load(f)
        
    for e in g["factual_entries"]:
        kind = "PRIME" if e["is_prime"] else "PROP "
        dist_tok = e.get("distinguishing_token_text", "None")
        print(f"{kind} | Entity ID: {e['entity_id']:4d} | Start: {e['start_idx']:4d} | Slot IDs: {e['slot_ids']} | DistTok: {repr(dist_tok)} | Text: {repr(e['text'][:80])}")

if __name__ == "__main__":
    main()
