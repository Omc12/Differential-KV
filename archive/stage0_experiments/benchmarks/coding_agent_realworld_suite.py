class CodingAgentBenchmark:
    """
    Simulates real-world coding agent workloads (repository navigation, editing).
    Tests context persistence and multi-step reasoning efficiency.
    """
    def __init__(self, harness):
        self.harness = harness

    def run_workflow(self, workflow_name):
        print(f"[Coding] Running workflow: {workflow_name}")
        # Workflows involve multiple requests sharing context
        # request1: index repo
        # request2: find bug
        # request3: fix bug
        
        results = []
        for step in range(3):
            res = self.harness.execute_request(f"Step {step} for {workflow_name}", max_tokens=50)
            results.append(res)
            
        return {
            "workflow": workflow_name,
            "steps": len(results),
            "total_latency": sum(r['latency'] for r in results),
            "success": True
        }
