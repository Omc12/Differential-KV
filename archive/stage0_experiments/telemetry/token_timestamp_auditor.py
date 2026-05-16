import time
import json
import os

class TokenTimestampAuditor:
    """
    PHASE 18.1E: Records exact timestamps for every token generated.
    """
    def __init__(self, export_path: str = "results/reconstruction_18_1/raw_wallclock_trace.log"):
        self.export_path = export_path
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

    def record_token(self, token_index: int, timestamp: float):
        with open(self.export_path, 'a') as f:
            f.write(f"TOKEN_{token_index}: {timestamp:.6f}\n")
