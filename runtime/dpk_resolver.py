import os
import asyncio
from typing import Dict, Any, Optional

from deployment.environment_configuration_manager import EnvironmentConfigurationManager
from deployment.deployment_bundle_builder import DeploymentBundleBuilder
from deployment.docker_runtime_materializer import DockerRuntimeMaterializer
from deployment.deployment_integrity_guard import DeploymentIntegrityGuard
from serving.serving_observability_bridge import ServingObservabilityBridge

class DPKResolver:
    """
    Unified Deployment Packaging & Kubernetes Readiness (DPK) Orchestrator.
    Manages the lifecycle from configuration to containerization.
    """
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.config_manager = EnvironmentConfigurationManager()
        self.bundle_builder = DeploymentBundleBuilder(workspace_root)
        self.integrity_guard = DeploymentIntegrityGuard(workspace_root)
        
        # We'll initialize observability later when we have a gateway/runtime
        self.observability = None

    def prepare_deployment(self, profile: str = "production"):
        """
        Full deployment preparation pipeline.
        """
        print(f"--- DPK Deployment Pipeline: {profile} ---")
        
        # 1. Configuration
        self.config_manager.save_profile(profile)
        config = self.config_manager.get_config()
        
        # 2. Safety Validation
        safety = self.integrity_guard.validate_deployment_safety(config)
        if not safety["is_safe"]:
            print(f"[DPK] Safety check failed: {safety['issues']}")
        
        # 3. Bundling
        bundle_path = self.bundle_builder.build_bundle(f"diffkv-{profile}")
        
        # 4. Containerization
        materializer = DockerRuntimeMaterializer(bundle_dir=bundle_path.replace(".tar.gz", ""))
        materializer.generate_docker_files()
        
        # 5. Integrity Check
        checksum = self.integrity_guard.calculate_bundle_checksum()
        print(f"[DPK] Final Deployment Checksum: {checksum}")
        
        return {
            "bundle_path": bundle_path,
            "checksum": checksum,
            "config": config,
            "readiness": 1.0 if safety["is_safe"] else 0.8
        }

    def setup_observability(self, gateway):
        """
        Links observability bridge to the serving stack.
        """
        self.observability = ServingObservabilityBridge(
            gateway.scheduler,
            gateway.session_manager,
            gateway.recovery_engine
        )
        self.observability.attach_to_app(gateway.app)
        print("[DPK] Observability bridge attached to API gateway.")

if __name__ == "__main__":
    resolver = DPKResolver(os.getcwd())
    results = resolver.prepare_deployment("local_test")
    print(f"DPK Pipeline Complete. Readiness: {results['readiness']}")
