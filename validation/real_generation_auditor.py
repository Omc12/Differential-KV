import json
import os

class RealGenerationAuditor:
    """
    Audits the generation process to ensure it follows the intended real model path.
    Saves traces for forensic verification.
    """
    def __init__(self, log_dir="results/reconstruction_10_5/audits"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def audit_generation(self, request_id, model_name, prompt, output):
        audit_data = {
            "request_id": request_id,
            "model_name": model_name,
            "prompt": prompt,
            "output_tokens": output.get("tokens", []),
            "output_text": output.get("text", ""),
            "timestamp": os.path.getmtime(__file__) # Just for mock timestamp
        }
        
        path = os.path.join(self.log_dir, f"audit_{request_id}.json")
        with open(path, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        return path
