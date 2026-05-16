class AsyncTimingGuard:
    """
    Prevents asynchronous execution overlap from inflating throughput metrics.
    Detects if background tasks are leaking into the primary timing window.
    """
    def __init__(self):
        self.active_tasks = 0

    def enter_async_scope(self):
        self.active_tasks += 1

    def exit_async_scope(self):
        self.active_tasks -= 1

    def is_contaminated(self):
        # If there are active background tasks during a synchronous timing measurement,
        # the result might be contaminated.
        return self.active_tasks > 0
