import os
import shutil
import logging
from typing import Dict, List

class HistoricalArchiveManager:
    """
    Manages structured archival of legacy and superseded systems.
    Ensures NO FILE DELETION, only reorganization.
    """
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.logger = logging.getLogger("HistoricalArchiveManager")
        self.archive_root = os.path.join(root_dir, "archive")
        
        self.structure = {
            "stage0_experiments": os.path.join(self.archive_root, "stage0_experiments"),
            "stage1_legacy": os.path.join(self.archive_root, "stage1_legacy"),
            "obsolete_benchmarks": os.path.join(self.archive_root, "obsolete_benchmarks"),
            "synthetic_era": os.path.join(self.archive_root, "synthetic_era"),
            "deprecated_integrity": os.path.join(self.archive_root, "deprecated_integrity"),
            "old_resolvers": os.path.join(self.archive_root, "old_resolvers")
        }

    def initialize_archive(self):
        """Creates the archival directory structure."""
        if not os.path.exists(self.archive_root):
            os.makedirs(self.archive_root)
            
        for name, path in self.structure.items():
            if not os.path.exists(path):
                os.makedirs(path)
                self.logger.info(f"Created archive directory: {name}")

    def archive_files(self, classification_manifest: Dict[str, List[str]]):
        """
        Moves files into the archive based on their classification.
        """
        self.logger.info("Archiving obsolete files...")
        
        # Define move mapping
        move_map = {
            "LEGACY": "stage1_legacy",
            "SUPERSEDED": "stage1_legacy",
            "STAGE1_HISTORICAL": "stage1_legacy",
            "EXPERIMENTAL": "stage0_experiments"
        }
        
        for cat, target_dir_name in move_map.items():
            target_dir = self.structure[target_dir_name]
            files_to_move = classification_manifest.get(cat, [])
            
            for rel_path in files_to_move:
                src = os.path.join(self.root_dir, rel_path)
                if not os.path.exists(src):
                    continue
                    
                # Handle nested files by recreating subdirs in archive
                dest = os.path.join(target_dir, rel_path)
                dest_dir = os.path.dirname(dest)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    
                try:
                    # Move file (overwrites if already in archive)
                    shutil.move(src, dest)
                    self.logger.debug(f"Archived: {rel_path} -> {target_dir_name}")
                except Exception as e:
                    self.logger.error(f"Failed to archive {rel_path}: {e}")

        self.logger.info("Archival process COMPLETED.")

    def restore_from_archive(self, rel_path: str):
        """Utility to recover a file from the archive if needed."""
        # Search for file in all archive structure
        for path in self.structure.values():
            archived_src = os.path.join(path, rel_path)
            if os.path.exists(archived_src):
                dest = os.path.join(self.root_dir, rel_path)
                dest_dir = os.path.dirname(dest)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                shutil.move(archived_src, dest)
                self.logger.info(f"Restored: {rel_path}")
                return True
        return False
