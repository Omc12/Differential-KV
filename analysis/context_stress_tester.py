class ContextStressTester:
    """Stress tests the model under extreme context lengths and noise."""
    def __init__(self, runner):
        self.runner = runner

    def run_stress_test(self, mode, base_ctx=16384):
        print(f"Starting stress test for {mode} at {base_ctx} tokens...")
        # Increase noise intensity or add multiple needles
        results = []
        for intensity in [0.1, 0.2, 0.3]:
            res = self.runner.execute_single_run(mode, base_ctx, "activation_code", use_noise=True)
            results.append(res)
        return results
