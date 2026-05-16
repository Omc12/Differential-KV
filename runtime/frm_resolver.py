import torch
from typing import Dict, Any, Optional

from persistent_weight_residency_controller import PersistentWeightResidencyController
from continuous_transformer_forward_materializer import ContinuousTransformerForwardMaterializer
from residency_truth_telemetry import residency_telemetry
from dense_path_residency_auditor import auditor
from residency_integrity_guard import guard
from runtime.hf_diffkv_wrapper import DiffKVHFWrapper

class FRMResolver:
    """
    Main resolver for FRM (Full Residency Materialization).
    Hard-binds sparse systems into the real continuously resident transformer path.
    """
    def __init__(self, wrapper: DiffKVHFWrapper):
        self.wrapper = wrapper
        self.weight_controller = PersistentWeightResidencyController(wrapper.model, device=wrapper.device)
        self.materializer = ContinuousTransformerForwardMaterializer(wrapper.model)
        
        # Enforce residency immediately
        self.weight_controller.enforce_residency()
        print("[FRM] Resolver initialized. Full model residency ENFORCED.")

    def execute_materialized_decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Executes a single decode step with residency auditing.
        """
        # Audit dense paths
        auditor.audit_step("embeddings")
        auditor.audit_step("mlp")
        auditor.audit_step("logits")
        auditor.audit_step("sampling")
        auditor.audit_step("tokenizer")
        
        # Execute materialized forward
        logits = self.materializer.execute_materialized_step(input_ids)
        
        return logits

    def get_frm_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive residency materialization report.
        """
        residency_metrics = self.weight_controller.get_residency_metrics()
        telemetry = residency_telemetry.get_residency_report()
        audit_report = auditor.get_audit_report()
        
        report = {
            **residency_metrics,
            **telemetry,
            **audit_report
        }
        
        guard.validate_residency(report)
        guard.check_integrity()
        
        return report
