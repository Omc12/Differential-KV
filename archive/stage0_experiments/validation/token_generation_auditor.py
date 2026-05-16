import torch
import time

class TokenGenerationAuditor:
    """
    PHASE 18.1C: Audits real token generation for scientific validity.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generation_traces = []

    def audit_generation(self, input_ids, output_ids, start_time, end_time):
        total_time = end_time - start_time
        num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
        
        tps = num_new_tokens / total_time if total_time > 0 else 0
        
        trace = {
            "num_new_tokens": num_new_tokens,
            "total_time": total_time,
            "tps": tps,
            "tokens": output_ids[0][-num_new_tokens:].tolist(),
            "text": self.tokenizer.decode(output_ids[0][-num_new_tokens:], skip_special_tokens=True)
        }
        self.generation_traces.append(trace)
        return trace

    def get_summary(self):
        if not self.generation_traces:
            return None
        
        all_tps = [t['tps'] for t in self.generation_traces]
        return {
            "avg_tps": sum(all_tps) / len(all_tps),
            "max_tps": max(all_tps),
            "min_tps": min(all_tps),
            "total_tokens": sum(t['num_new_tokens'] for t in self.generation_traces)
        }
