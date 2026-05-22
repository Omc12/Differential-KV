class RetrievalEntropyScheduler:
    def __init__(self):
        pass

    def schedule(self, sequence_entropy):
        # High entropy tokens (surprisal) trigger deeper / broader retrieval
        # Low entropy tokens (predictable) use minimal cached anchors
        if sequence_entropy > 0.8:
            return "deep_search"
        return "local_cache"
