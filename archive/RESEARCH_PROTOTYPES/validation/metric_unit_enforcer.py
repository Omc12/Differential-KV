class MetricUnitEnforcer:
    """
    Enforces consistent units (e.g., seconds for latency, tokens/sec for throughput).
    Prevents mixing ms and seconds or characters and tokens.
    """
    def __init__(self):
        pass

    def enforce_seconds(self, value, unit="s"):
        if unit == "ms":
            return value / 1000.0
        return value

    def enforce_tps(self, count, duration_s):
        if duration_s == 0:
            return 0
        return count / duration_s
