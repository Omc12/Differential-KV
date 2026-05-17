import time
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    NVML_TEMPERATURE_GPU,
    NVML_CLOCK_SM,
)

class NativeNVMLTelemetryRuntime:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.initialized = False

        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
            self.initialized = True

            print(
                "[NVML] Native telemetry initialized successfully."
            )

        except Exception as e:
            raise RuntimeError(
                f"[FALLBACK_VIOLATION] "
                f"NVML initialization failed: {e}"
            )

    def sample(self):
        if not self.initialized:
            raise RuntimeError(
                "[FALLBACK_VIOLATION] "
                "NVML not initialized."
            )

        util = nvmlDeviceGetUtilizationRates(self.handle)
        mem = nvmlDeviceGetMemoryInfo(self.handle)

        return {
            "timestamp": time.time(),
            "temperature_c": nvmlDeviceGetTemperature(
                self.handle,
                NVML_TEMPERATURE_GPU
            ),
            "power_w": (
                nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            ),
            "sm_clock_mhz": nvmlDeviceGetClockInfo(
                self.handle,
                NVML_CLOCK_SM
            ),
            "gpu_util_percent": util.gpu,
            "memory_util_percent": util.memory,
            "vram_used_mb": (
                mem.used / (1024 * 1024)
            ),
            "vram_total_mb": (
                mem.total / (1024 * 1024)
            ),
        }

    def shutdown(self):
        if self.initialized:
            nvmlShutdown()
            self.initialized = False
