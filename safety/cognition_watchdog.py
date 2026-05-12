import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class CognitionWatchdog:
    """
    A daemon that runs alongside a cognition session. If the runtime hangs
    or entropy reaches fatal levels, it forcefully restarts the container from the last snapshot.
    """
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = time.time()
        self.is_running = False

    def start_monitoring(self, restart_callback: Callable):
        self.is_running = True
        self.last_heartbeat = time.time()
        logger.info("Cognition Watchdog started.")
        # In reality, this would run in a separate thread.
        # We simulate a check here.
        self._check_health(restart_callback)

    def ping(self):
        """Runtime calls this to indicate it is still alive."""
        self.last_heartbeat = time.time()

    def _check_health(self, restart_callback: Callable):
        if time.time() - self.last_heartbeat > self.timeout_seconds:
            logger.critical("WATCHDOG TIMEOUT! Runtime appears hung. Initiating restart.")
            self.is_running = False
            restart_callback()
