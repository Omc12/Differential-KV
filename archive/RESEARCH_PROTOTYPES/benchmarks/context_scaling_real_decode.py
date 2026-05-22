class ContextScalingRealDecode:
    """
    Measures how real decode throughput scales with context length.
    Reveals the O(N^2) vs O(N) or O(log N) behavior of Differential KV.
    """
    def __init__(self, runner):
        self.runner = runner

    def run_scaling_test(self, context_lengths, tokens_to_generate=50):
        results = {}
        for ctx in context_lengths:
            # Construct dummy prompt of ctx tokens
            prompt = "token " * ctx
            res = self.runner.run_inference(prompt, tokens_to_generate)
            results[ctx] = res["tps"]
        return results
