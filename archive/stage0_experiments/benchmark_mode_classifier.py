from enum import Enum, auto

class BenchmarkMode(Enum):
    SUBSYSTEM = auto()    # Isolated sparse runtime components
    INTEGRATED = auto()   # Sparse runtime + real transformer execution
    PRODUCTION = auto()   # True user-visible serving benchmark

class BenchmarkModeClassifier:
    """
    Ensures every benchmark run explicitly declares its mode.
    """
    def __init__(self, mode: str):
        mode = mode.upper()
        if mode == "SUBSYSTEM":
            self.mode = BenchmarkMode.SUBSYSTEM
        elif mode == "INTEGRATED":
            self.mode = BenchmarkMode.INTEGRATED
        elif mode == "PRODUCTION":
            self.mode = BenchmarkMode.PRODUCTION
        else:
            raise ValueError(f"Invalid Benchmark Mode: {mode}. Must be SUBSYSTEM, INTEGRATED, or PRODUCTION.")

    def get_mode_name(self) -> str:
        return self.mode.name
