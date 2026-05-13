class RepositoryMemoryDrift:
    """
    Tracks how "memory" (KV cache) of a repo changes over time.
    Detects if the system is "forgetting" old parts of the repo.
    """
    def __init__(self):
        self.repo_map_history = []

    def log_repo_state(self, file_retention_map: dict):
        """
        file_retention_map: mapping of file_id to percentage of KV retained.
        """
        self.repo_map_history.append(file_retention_map)

    def calculate_forgetting_rate(self) -> float:
        if len(self.repo_map_history) < 2:
            return 0.0
        # Compare first state with last state
        start = sum(self.repo_map_history[0].values()) / len(self.repo_map_history[0])
        end = sum(self.repo_map_history[-1].values()) / len(self.repo_map_history[-1])
        return (start - end) / start
