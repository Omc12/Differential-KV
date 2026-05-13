class CodeRetrievalChain:
    """
    Validates deep file-dependency recall chains.
    Ensures that if File A imports B, and B imports C, all are retrievable.
    """
    def __init__(self):
        self.chains = []

    def verify_chain(self, chain: list, retrieved_ids: set):
        missing = [f for f in chain if f not in retrieved_ids]
        success = len(missing) == 0
        self.chains.append({"chain": chain, "success": success, "missing": missing})
        return success

    def get_chain_integrity(self):
        if not self.chains: return 1.0
        return sum(1 for c in self.chains if c['success']) / len(self.chains)
