from continuity.persistent_goal_tracker import PersistentGoalTracker
import os

def run_goal_survival_eval():
    """
    Stress tests goal preservation across multiple 'session' restarts.
    """
    print("=== Phase 36: Autonomous Goal Survival Evaluation ===")
    
    goal_file = "eval_goals.json"
    if os.path.exists(goal_file): os.remove(goal_file)
    
    tracker = PersistentGoalTracker(goal_file=goal_file)
    
    # 1. Set goals
    tracker.add_goal("Solve P=NP", "A simple starter task", priority=5)
    tracker.add_goal("Build Dyson Sphere", "Long-term infrastructure", priority=3)
    
    # 2. Simulate session restarts
    num_restarts = 10
    for i in range(num_restarts):
        # Reload tracker
        tracker = PersistentGoalTracker(goal_file=goal_file)
        active_goals = tracker.get_active_goals()
        
        # Check if goals survived
        if len(active_goals) != 2:
            print(f"Goal survival failed at restart {i}!")
            return
            
        # Update progress slightly
        tracker.update_goal_progress(active_goals[0]["id"], (i+1)/100.0)
        
    print(f"Goal Survival Evaluation: 100% (2/2 goals survived {num_restarts} restarts)")
    print(f"Final Goal 0 Progress: {tracker.get_active_goals()[0]['progress']:.2f}")
    
    if os.path.exists(goal_file): os.remove(goal_file)

if __name__ == "__main__":
    run_goal_survival_eval()
