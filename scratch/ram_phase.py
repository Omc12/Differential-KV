import subprocess, time, sys, os, threading
import psutil
def sample(proc, stop, st):
    p = psutil.Process(proc.pid)
    while not stop.is_set():
        try:
            rss = p.memory_info().rss
            for c in p.children(recursive=True):
                try: rss += c.memory_info().rss
                except: pass
            st['cur']=rss; st['peak']=max(st['peak'],rss)
        except: break
        time.sleep(0.1)
env = dict(os.environ, DIFFKV_PRESET="mid", DIFFKV_PREFILL_CHUNK_SIZE="512", DIFFKV_COMPRESSOR_THREADS="4",
           DIFFKV_MAX_TOKENS="50", DIFFKV_USE_GPU="1", DIFFKV_MICRO_BLOCK_SIZE="256", DIFFKV_TEMPERATURE="0.1",
           VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
proc = subprocess.Popen(["diffkv_native/build/diffkv_native","diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
st={'peak':0,'cur':0}; stop=threading.Event()
th=threading.Thread(target=sample,args=(proc,stop,st)); th.start()
proc.stdin.write(open(sys.argv[1]).read()+"\n"); proc.stdin.flush(); proc.stdin.close()
ready=resp=None
for line in proc.stdout:
    if "__READY__" in line and ready is None: ready=st['cur']
    if "__RESPONSE__" in line and resp is None: resp=st['peak']  # peak so far = through prefill
    if "__FINISH__" in line: break
fin=st['peak']; stop.set(); th.join(); proc.kill()
print(f"startup={ready/1e9:.2f}GB  peak_through_prefill={resp/1e9:.2f}GB  peak_overall={fin/1e9:.2f}GB")
