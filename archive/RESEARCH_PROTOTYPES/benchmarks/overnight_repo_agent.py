import time
import random

class OvernightRepoAgent:
    """
    Simulates a coding agent editing a large repository over many hours.
    Tests if the system maintains context of the entire repo.
    """
    def __init__(self, repo_size_files: int = 1000):
        self.repo_size = repo_size_files
        self.current_file = 0

    def run_session(self, hours: float):
        end_time = time.time() + (hours * 3600)
        while time.time() < end_time:
            # Simulate editing a random file
            file_id = random.randint(0, self.repo_size - 1)
            print(f"Agent editing file_{file_id}...")
            # Simulate retrieval and modification
            time.sleep(random.uniform(0.1, 0.5)) 
            
if __name__ == "__main__":
    agent = OvernightRepoAgent()
    # Run a short 5s session for validation
    agent.run_session(5/3600)
