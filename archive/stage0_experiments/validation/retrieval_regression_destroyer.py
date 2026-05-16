import torch

class RetrievalRegressionDestroyer:
    """
    Adversarially hunts for retrieval regressions caused by sparsity.
    Randomly perturbs sparse density to find failure boundaries.
    """
    def __init__(self, model_runner):
        self.model_runner = model_runner

    def run_adversarial_test(self, context: str, target: str):
        densities = [0.01, 0.05, 0.1, 0.2]
        results = {}
        
        for d in densities:
            success = self.model_runner.test_retrieval(context, target, density=d)
            results[d] = success
            if not success and d > 0.1:
                return f"CRITICAL REGRESSION: Failed at {d} density"
                
        return results
