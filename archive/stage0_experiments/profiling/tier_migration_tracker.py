class TierMigrationTracker:
    """
    Logs KV migration events between memory tiers (e.g., L1 Cache -> RAM -> SSD).
    """
    def __init__(self):
        self.migrations = [] # List of (timestamp, size, direction)

    def record_migration(self, size_bytes: int, source: str, destination: str):
        self.migrations.append({
            "size": size_bytes,
            "src": source,
            "dst": destination
        })

    def get_total_bandwidth_used(self) -> int:
        return sum(m['size'] for m in self.migrations)
