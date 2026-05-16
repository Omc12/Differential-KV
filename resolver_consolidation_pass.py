import logging
import os
from typing import Dict, Any, List

class ResolverConsolidationPass:
    """
    Audits and consolidates resolver systems.
    Reduces resolver fragmentation and ensures clean runtime flow.
    """
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.logger = logging.getLogger("ResolverConsolidationPass")
        self.resolvers_dir = os.path.join(root_dir, "runtime")

    def audit_resolvers(self) -> List[str]:
        """Identifies active and redundant resolvers."""
        active_resolvers = [
            "lgs_resolver.py", "pdm_resolver.py", "xvm_resolver.py"
        ]
        all_resolvers = [f for f in os.listdir(self.resolvers_dir) if f.endswith("_resolver.py")]
        
        redundant = [r for r in all_resolvers if r not in active_resolvers]
        self.logger.info(f"Identified {len(redundant)} redundant resolvers for archival.")
        return redundant

    def unify_orchestration_logic(self):
        """
        Ensures active resolvers use a unified pattern.
        (Refactoring active resolvers to inherit from a base class if needed)
        """
        # In this pass, we primarily focus on classifying and archiving old ones.
        # Structural refactoring would happen in Stage 2.
        pass

    def run_consolidation(self, archive_manager: Any):
        """
        Executes the consolidation by moving redundant resolvers to the archive.
        """
        redundant = self.audit_resolvers()
        for r in redundant:
            rel_path = os.path.join("runtime", r)
            archive_manager.archive_files({"SUPERSEDED": [rel_path]})
        self.logger.info("Resolver consolidation COMPLETED.")
