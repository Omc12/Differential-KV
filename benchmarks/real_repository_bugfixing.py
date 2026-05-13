class RealRepositoryBugfixing:
    def __init__(self, repo_path, bug_dataset):
        self.repo_path = repo_path
        self.bug_dataset = bug_dataset

    def run_benchmark(self, agent):
        print(f"Running Real Repository Bugfixing on {self.repo_path}")
        results = []
        for bug in self.bug_dataset:
            success = agent.attempt_fix(bug)
            results.append(success)
        
        accuracy = sum(results) / len(results) if results else 0
        print(f"Bugfix Accuracy: {accuracy:.2f}")
        return accuracy
