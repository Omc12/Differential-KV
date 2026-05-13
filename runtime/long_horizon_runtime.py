import time
import logging
from typing import Dict, Any, List
try:
    from .persistent_sparse_executor import PersistentSparseExecutor
    from .runtime_fatigue_tracker import RuntimeFatigueTracker
    from .context_survival_manager import ContextSurvivalManager
except ImportError:
    try:
        from runtime.persistent_sparse_executor import PersistentSparseExecutor
        from runtime.runtime_fatigue_tracker import RuntimeFatigueTracker
        from runtime.context_survival_manager import ContextSurvivalManager
    except ImportError:
        from persistent_sparse_executor import PersistentSparseExecutor
        from runtime_fatigue_tracker import RuntimeFatigueTracker
        from context_survival_manager import ContextSurvivalManager

class LongHorizonRuntime:
    """
    Main entry point for continuous sparse execution validation.
    Designed for 1h - 8h+ runs.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executor = PersistentSparseExecutor(config)
        self.fatigue_tracker = RuntimeFatigueTracker()
        self.context_manager = ContextSurvivalManager()
        self.start_time = 0
        self.is_running = False

    def run(self, duration_hours: float):
        """
        Runs the long-horizon loop for the specified duration.
        """
        duration_seconds = duration_hours * 3600
        self.start_time = time.perf_counter()
        self.is_running = True
        
        logging.info(f"Starting Long-Horizon Runtime for {duration_hours}h...")
        
        try:
            while time.perf_counter() - self.start_time < duration_seconds:
                elapsed = time.perf_counter() - self.start_time
                
                # 1. Execute Sparse Step
                step_results = self.executor.execute_step()
                
                # 2. Track Fatigue
                self.fatigue_tracker.log_step(step_results)
                
                # 3. Manage Context Survival
                self.context_manager.verify_integrity(step_results)
                
                # 4. Heartbeat
                if int(elapsed) % 60 == 0:
                    logging.info(f"Heartbeat: {elapsed/3600:.2f}h elapsed. TPS: {step_results.get('tps', 0):.2f}")
                
                # Simulate some workload delay or processing
                time.sleep(0.1) 
                
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.is_running = False
        report = self.fatigue_tracker.generate_report()
        logging.info("Long-Horizon Runtime Shutdown.")
        return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runtime = LongHorizonRuntime({"model": "test-model"})
    # Short test run for 0.01h (36s)
    runtime.run(0.01)
