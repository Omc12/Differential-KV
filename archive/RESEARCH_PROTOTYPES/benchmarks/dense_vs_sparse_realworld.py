class DenseVsSparseRealworld:
    """
    Direct comparison between Dense and Sparse Differential KV in real-world scenarios.
    Focuses on physically grounded metrics.
    """
    def __init__(self, dense_runner, sparse_runner, tps_calculator):
        self.dense_runner = dense_runner
        self.sparse_runner = sparse_runner
        self.tps_calculator = tps_calculator

    def compare(self, prompt, max_new_tokens=100):
        dense_res = self.dense_runner.run_inference(prompt, max_new_tokens)
        sparse_res = self.sparse_runner.generate(prompt, max_new_tokens=max_new_tokens)
        
        # Normalize sparse output if it's a string
        if isinstance(sparse_res, str):
            # We would need tokens here for real TPS calculation
            pass
            
        return {
            "dense_tps": dense_res["tps"],
            "sparse_tps": sparse_res.get("tps", 0) if isinstance(sparse_res, dict) else 0,
            "speedup": sparse_res.get("tps", 0) / dense_res["tps"] if dense_res["tps"] > 0 and isinstance(sparse_res, dict) else 0
        }
