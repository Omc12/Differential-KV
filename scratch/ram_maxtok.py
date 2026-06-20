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
            st['peak']=max(st['peak'],rss)
        except: break
        time.sleep(0.1)
env = dict(os.environ, DIFFKV_PRESET="mid", DIFFKV_PREFILL_CHUNK_SIZE="512", DIFFKV_COMPRESSOR_THREADS="4",
           DIFFKV_MAX_TOKENS=sys.argv[2], DIFFKV_USE_GPU="1", DIFFKV_MICRO_BLOCK_SIZE="256", DIFFKV_TEMPERATURE="0.1",
           VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
proc = subprocess.Popen(["diffkv_native/build/diffkv_native","diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
st={'peak':0}; stop=threading.Event()
th=threading.Thread(target=sample,args=(proc,stop,st)); th.start()
proc.stdin.write(open(sys.argv[1]).read()+"\n"); proc.stdin.flush(); proc.stdin.close()
for line in proc.stdout:
    if "__FINISH__" in line: break
stop.set(); th.join(); proc.kill()
print(f"[max_tokens={sys.argv[2]}] peak_RSS={st['peak']/1e9:.2f}GB")
