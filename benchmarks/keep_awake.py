"""Hold off idle sleep while a campaign runs: sleep kills the CUDA context and
the run dies without an error. Exits when the watched process does.
Usage: python benchmarks/keep_awake.py <pid>
"""
import ctypes
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x102

k32 = ctypes.windll.kernel32
h = k32.OpenProcess(SYNCHRONIZE, False, int(sys.argv[1]))
if not h:
    sys.exit("no such process")
k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
while k32.WaitForSingleObject(h, 60000) == WAIT_TIMEOUT:
    time.sleep(0)
k32.SetThreadExecutionState(ES_CONTINUOUS)
