import json, time, threading, subprocess, urllib.request
import psutil
prompt = open("scratch/test_pp24k.txt").read()
payload = json.dumps({"model":"qwen2.5:1.5b-instruct","prompt":prompt,"stream":False,
                      "options":{"num_predict":30,"temperature":0.1,"num_ctx":25600}}).encode()
def ollama_procs():
    out=[]
    for p in psutil.process_iter(['name','cmdline']):
        cl=' '.join(p.info.get('cmdline') or [])
        if 'ollama' in (p.info.get('name') or '').lower() or 'ollama' in cl.lower():
            out.append(p)
    return out
peak={'rss':0,'pid':0}; stop=threading.Event()
def sample():
    while not stop.is_set():
        tot=0; hi=(0,0)
        for p in ollama_procs():
            try:
                r=p.memory_info().rss; tot+=r
                if r>hi[0]: hi=(r,p.pid)
            except: pass
        if tot>peak['rss']: peak['rss']=tot; peak['pid']=hi[1]
        time.sleep(0.12)
th=threading.Thread(target=sample); th.start()
t0=time.time()
req=urllib.request.Request("http://localhost:11434/api/generate",data=payload,headers={"Content-Type":"application/json"})
resp=json.loads(urllib.request.urlopen(req,timeout=900).read())
wall=time.time()-t0
# grab footprint of the heaviest runner right now (KV still allocated)
fp="?"
try:
    o=subprocess.run(["footprint","-p",str(peak['pid'])],capture_output=True,text=True,timeout=20).stdout
    for line in o.splitlines():
        if "phys_footprint" in line.lower(): fp=line.strip()
except Exception as e: fp=f"(err {e})"
stop.set(); th.join()
pe_n=resp.get("prompt_eval_count",0); pe_d=resp.get("prompt_eval_duration",1)/1e9
ev_n=resp.get("eval_count",0);        ev_d=resp.get("eval_duration",1)/1e9
print(f"[OLLAMA plain llama.cpp q4, num_ctx=25600] prompt_tokens={pe_n}")
print(f"  prefill: {pe_d:.1f}s  ({pe_n/max(pe_d,.01):.0f} tok/s)")
print(f"  decode : {ev_d:.2f}s  ({ev_n/max(ev_d,.01):.0f} tok/s)")
print(f"  peak_RSS={peak['rss']/1e9:.2f}GB  footprint[{peak['pid']}]: {fp}  wall={wall:.1f}s")
print(f"  OUTPUT: {resp.get('response','')[:220]!r}")
