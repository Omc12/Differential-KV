import threading
import time

class MultiUserServingSuite:
    """
    Benchmarks concurrent serving performance.
    Measures degradation as concurrency increases.
    """
    def __init__(self, harness):
        self.harness = harness
        self.results = []

    def run_concurrent_load(self, num_users, requests_per_user):
        print(f"[Serving] Running concurrent load: {num_users} users")
        threads = []
        
        def user_task():
            for _ in range(requests_per_user):
                res = self.harness.execute_request("Concurrent request", max_tokens=20)
                self.results.append(res)

        for _ in range(num_users):
            t = threading.Thread(target=user_task)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return self.results
