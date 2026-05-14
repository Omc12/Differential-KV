import json
import os

class RetrievalSurvivalMapper:
    """
    PHASE 18.4B: Maps retrieval accuracy against context length and pruning density.
    """
    def __init__(self, export_dir="results/reconstruction_18_4/"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        self.results = []

    def record_test(self, context_len, test_type, success, response, ground_truth=None):
        entry = {
            "context_len": context_len,
            "test_type": test_type,
            "success": success,
            "response": response,
            "ground_truth": ground_truth,
            "timestamp": os.path.getmtime(__file__) # Real clock
        }
        self.results.append(entry)
        
        # Incremental export
        with open(os.path.join(self.export_dir, "raw_retrieval_accuracy.jsonl"), 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def generate_summary(self):
        summary = {}
        for r in self.results:
            key = (r['context_len'], r['test_type'])
            if key not in summary:
                summary[key] = {"total": 0, "passed": 0}
            summary[key]["total"] += 1
            if r['success']:
                summary[key]["passed"] += 1
        return summary
