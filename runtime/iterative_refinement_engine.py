from typing import Callable, Any, Dict, List
import time

class IterativeRefinementEngine:
    """
    Executes bounded reasoning refinement loops.
    Each iteration must produce an explicit rollup and check for convergence or depth limit.
    """
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def refine(self, initial_state: Any, refine_fn: Callable[[Any, int], Any], convergence_fn: Callable[[Any], bool]) -> Dict[str, Any]:
        """
        Runs the refinement loop.
        refine_fn: (current_state, iteration_index) -> next_state
        convergence_fn: (current_state) -> bool
        """
        current_state = initial_state
        history = []
        start_time = time.time()
        
        for i in range(self.max_depth):
            iteration_start = time.time()
            next_state = refine_fn(current_state, i)
            iteration_duration = time.time() - iteration_start
            
            history.append({
                "iteration": i,
                "duration": iteration_duration,
                "state_snapshot": str(next_state)[:200] # Representative summary
            })
            
            current_state = next_state
            
            if convergence_fn(current_state):
                break
                
        total_duration = time.time() - start_time
        return {
            "final_state": current_state,
            "iterations": len(history),
            "history": history,
            "total_duration": total_duration,
            "converged": convergence_fn(current_state)
        }
