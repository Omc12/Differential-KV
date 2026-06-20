import subprocess, time, sys, os, threading
import psutil
def sample(proc, stop, peaks):
    p = psutil.Process(proc.pid)
    while not stop.is_set():
        try:
            rss = p.memory_info().rss
            for c in p.children(recursive=True):
                try: rss += c.memory_info().rss
                except: pass
            peaks['cur']=rss; peaks['peak']=max(peaks['peak'],rss)
        except: break
        time.sleep(0.2)
proc = subprocess.Popen(["./diffkv_venv/bin/python3","scratch/mlx_ram_runner.py", sys.argv[1]],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
peaks={'peak':0,'cur':0}; stop=threading.Event()
th=threading.Thread(target=sample,args=(proc,stop,peaks)); th.start()
startup=None
for line in proc.stdout:
    if "__READY__" in line and startup is None: startup=peaks['cur']
    if "__FINISH__" in line: break
time.sleep(0.3); stop.set(); th.join(); proc.kill()
print(f"[MLX 4bit] startup_RSS={startup/1e9:.2f}GB  peak_RSS={peaks['peak']/1e9:.2f}GB")
