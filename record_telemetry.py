import subprocess
import time

print("Starting nvidia-smi dmon...")
with open("live_serving_telemetry.log", "w") as f:
    p = subprocess.Popen(["nvidia-smi", "dmon"], stdout=f)
    time.sleep(30)
    p.terminate()
print("Telemetry recording complete.")
