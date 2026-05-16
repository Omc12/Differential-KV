import time
import json
import os

class LifecycleAuditor:
    """
    Audits the full lifecycle of an inference request.
    Records fine-grained events to distinguish prompt processing from generation.
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.active_requests = {}
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def start_request(self, request_id, prompt_len):
        self.active_requests[request_id] = {
            "request_id": request_id,
            "prompt_length": prompt_len,
            "events": [{"event": "start", "timestamp": time.perf_counter()}],
            "status": "active"
        }

    def record_event(self, request_id, event_name):
        if request_id in self.active_requests:
            self.active_requests[request_id]["events"].append({
                "event": event_name,
                "timestamp": time.perf_counter()
            })

    def end_request(self, request_id, total_tokens):
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            req["events"].append({"event": "end", "timestamp": time.perf_counter()})
            req["total_generated_tokens"] = total_tokens
            req["status"] = "completed"
            
            self._flush_to_disk(req)
            del self.active_requests[request_id]

    def _flush_to_disk(self, request_data):
        filename = f"audit_{request_data['request_id']}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(request_data, f, indent=4)
