import torch

class BandwidthPressureBalancer:
    """
    Balances distributed load based on real-time bandwidth pressure.
    Prevents node congestion by migrating KV shards dynamically.
    """
    def __init__(self, bandwidth_threshold: float = 0.8):
        self.bandwidth_threshold = bandwidth_threshold
        self.node_loads = {}

    def report_load(self, node_id: int, load: float):
        self.node_loads[node_id] = load

    def get_rebalance_plan(self):
        """
        Identifies overloaded nodes and suggests shard migrations.
        """
        overloaded = [node for node, load in self.node_loads.items() if load > self.bandwidth_threshold]
        underloaded = [node for node, load in self.node_loads.items() if load < 0.3]
        
        plan = []
        for src in overloaded:
            if underloaded:
                dst = underloaded.pop(0)
                plan.append((src, dst, "migrate_shard"))
                
        return plan
