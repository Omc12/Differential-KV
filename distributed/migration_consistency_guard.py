"""
distributed/migration_consistency_guard.py

Ensures data consistency during high-pressure shard migrations.
Prevents 'double routing' or 'dropped requests' during transitions.
"""

from typing import Dict, Set, Any
import logging

class MigrationConsistencyGuard:
    """
    Consistency manager for shard migrations.
    """
    def __init__(self):
        self.in_flight_migrations: Set[int] = set()
        self.logger = logging.getLogger("MigrationConsistencyGuard")

    def begin_migration(self, shard_id: int):
        """Marks a shard as 'migrating'."""
        self.in_flight_migrations.add(shard_id)
        self.logger.info(f"Migration Consistency GUARD: Shard {shard_id} is moving...")

    def end_migration(self, shard_id: int):
        """Releases the lock on a shard."""
        if shard_id in self.in_flight_migrations:
            self.in_flight_migrations.remove(shard_id)
            self.logger.info(f"Migration Consistency GUARD: Shard {shard_id} settled.")

    def is_safe_to_route(self, shard_id: int) -> bool:
        """Checks if a shard is in a stable state for routing."""
        return shard_id not in self.in_flight_migrations
