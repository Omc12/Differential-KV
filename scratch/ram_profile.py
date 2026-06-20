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
            peaks['cur'] = rss
            peaks['peak'] = max(peaks['peak'], rss)
            peaks['series'].append((time.time()-peaks['t0'], rss))
        except: break
        time.sleep(0.2)

def run_native(prompt, model):
    env = dict(os.environ, DIFFKV_PRESET="mid", DIFFKV_PREFILL_CHUNK_SIZE="512", DIFFKV_COMPRESSOR_THREADS="4",
               DIFFKV_MAX_TOKENS="40", DIFFKV_USE_GPU="1", DIFFKV_MICRO_BLOCK_SIZE="256", DIFFKV_TEMPERATURE="0.1",
               VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.Popen(["diffkv_native/build/diffkv_native", model],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    peaks = {'peak':0,'cur':0,'series':[],'t0':time.time()}
    stop = threading.Event()
    th = threading.Thread(target=sample, args=(proc, stop, peaks)); th.start()
    proc.stdin.write(prompt+"\n"); proc.stdin.flush(); proc.stdin.close()
    startup = None
    for line in proc.stdout:
        if "__READY__" in line and startup is None:
            startup = peaks['cur']
        if "__FINISH__" in line: break
    time.sleep(0.3)
    stop.set(); th.join(); proc.kill()
    return startup, peaks['peak']

prompt = open(sys.argv[1]).read()
model = "diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"
startup, peak = run_native(prompt, model)
print(f"[NATIVE q4] startup_RSS={startup/1e9:.2f}GB  peak_RSS={peak/1e9:.2f}GB")
