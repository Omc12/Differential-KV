import os
import json
from .real_inference_harness import RealInferenceHarness

class RealPromptExecutor:
    """
    Executes real-world prompts against the inference harness.
    Supports loading prompt sets from disk.
    """
    def __init__(self, harness):
        self.harness = harness
        self.prompt_dir = "results/reconstruction_10/raw_prompt_runs"

    def load_prompts(self, prompt_file=None):
        if prompt_file is None or not os.path.exists(prompt_file):
            # Fallback to some default real-world prompts
            return [
                "Summarize the following technical documentation about KV caches...",
                "Write a Python script to optimize sparse attention kernels...",
                "Explain the difference between FlashAttention and Differential KV...",
                "Given a context of 128k tokens, find the needle in the haystack..."
            ]
        with open(prompt_file, 'r') as f:
            return json.load(f)

    def execute_suite(self, prompt_file=None):
        prompts = self.load_prompts(prompt_file)
        print(f"[Executor] Running suite with {len(prompts)} real prompts.")
        
        results = self.harness.run_benchmark_suite(prompts)
        
        # Save raw prompt execution logs
        for res in results:
            log_path = os.path.join(self.prompt_dir, f"run_{res['request_id']}.json")
            with open(log_path, 'w') as f:
                json.dump(res, f, indent=4)
                
        return results

if __name__ == "__main__":
    # Mock engine for testing
    class MockEngine:
        def process_prompt(self, p): pass
        def generate_next_token(self): return "token"

    harness = RealInferenceHarness(MockEngine())
    executor = RealPromptExecutor(harness)
    executor.execute_suite()
