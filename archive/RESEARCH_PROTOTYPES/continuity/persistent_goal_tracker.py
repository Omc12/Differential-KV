import json
import os
from typing import List, Dict, Any

class PersistentGoalTracker:
    """
    Tracks long-term objectives and planning states across sessions.
    """
    def __init__(self, goal_file: str = "persistent_goals.json"):
        self.goal_file = goal_file
        self.goals = self.load_goals()

    def load_goals(self) -> List[Dict[str, Any]]:
        """
        Loads goals from persistent storage.
        """
        if os.path.exists(self.goal_file):
            with open(self.goal_file, "r") as f:
                return json.load(f)
        return []

    def save_goals(self):
        """
        Saves goals to persistent storage.
        """
        with open(self.goal_file, "w") as f:
            json.dump(self.goals, f, indent=4)

    def add_goal(self, title: str, description: str, priority: int = 1):
        """
        Adds a new persistent goal.
        """
        goal = {
            "id": len(self.goals),
            "title": title,
            "description": description,
            "priority": priority,
            "status": "pending",
            "progress": 0.0,
            "created_at": "now"
        }
        self.goals.append(goal)
        self.save_goals()

    def update_goal_progress(self, goal_id: int, progress: float, status: str = "active"):
        """
        Updates the progress of an existing goal.
        """
        for goal in self.goals:
            if goal["id"] == goal_id:
                goal["progress"] = progress
                goal["status"] = status
                break
        self.save_goals()

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """
        Returns all goals that are not yet completed.
        """
        return [g for g in self.goals if g["status"] != "completed"]
