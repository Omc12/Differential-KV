"""
memory/migration_storm_detector.py

Detects and suppresses excessive shard migrations ('migration storms').
Prevents bandwidth collapse due to ping-ponging shards.
"""

import time
from typing import List, Dict
import logging

class MigrationStormDetector:
    """
    Traffic monitor for distributed shard migrations.
    """
    def __init__(self, storm_threshold: int = 10, window_sec: float = 5.0):
        self.storm_threshold = storm_threshold
        self.window = window_sec
        self.history: List[float] = []
        self.logger = logging.getLogger("MigrationStormDetector")

    def record_migration(self):
        """Records a migration event."""
        self.history.append(time.time())
        self._clean_history()
        
        if self.is_storming():
            self.logger.warning(f"MIGRATION STORM DETECTED: {len(self.history)} migrations in {self.window}s")

    def is_storming(self) -> bool:
        """Returns True if migration frequency exceeds threshold."""
        return len(self.history) >= self.storm_threshold

    def should_suppress(self) -> bool:
        """Advises whether to suppress further migrations."""
        return self.is_storming()

    def _clean_history(self):
        """Removes old events from history."""
        now = time.time()
        self.history = [t for t in self.history if now - t <= self.window]
