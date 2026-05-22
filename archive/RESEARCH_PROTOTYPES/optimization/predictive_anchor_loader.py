class PredictiveAnchorLoader:
    def __init__(self, history_length=10):
        self.history = []
        self.history_length = history_length

    def update_history(self, anchor_id):
        self.history.append(anchor_id)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def predict_next(self):
        # Simple n-gram or Markov based prediction for anchor locality
        if not self.history:
            return []
        # Mock prediction
        last_anchor = self.history[-1]
        return [last_anchor + 1, last_anchor + 2]
