from orchestration.context_execution_router import ContextExecutionRouter
from orchestration.retrieval_priority_scheduler import RetrievalPriorityScheduler
from orchestration.hardware_pressure_controller import HardwarePressureController
from typing import Dict, Any, List

class InferenceRuntimeOrchestrator:
    """
    Main orchestrator for workload-aware routing and adaptive execution balancing.
    Manages hardware pressure and retrieval priorities.
    """
    def __init__(self):
        self.router = ContextExecutionRouter()
        self.scheduler = RetrievalPriorityScheduler()
        self.pressure_controller = HardwarePressureController()

    def process_request(self, request_id: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrates the processing of an inference request."""
        # 1. Check hardware pressure
        load_status = self.pressure_controller.get_load_status()
        
        # 2. Schedule based on priority
        priority = context_data.get("priority", "normal")
        self.scheduler.add_task(request_id, priority)
        
        # 3. Route to appropriate execution unit
        execution_unit = self.router.route_context(context_data)
        
        return {
            "request_id": request_id,
            "execution_unit": execution_unit,
            "load_status": load_status,
            "scheduled_priority": priority
        }

    def get_orchestration_metrics(self) -> Dict[str, Any]:
        return {
            "router_stats": self.router.get_stats(),
            "scheduler_stats": self.scheduler.get_stats(),
            "pressure_stats": self.pressure_controller.get_stats()
        }
