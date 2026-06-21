import json, time, threading, subprocess, sys, urllib.request
import psutil

prompt = open("scratch/test_pp24k.txt").read()
payload = json.dumps({"model":"qwen2.5:1.5b-instruct","prompt":prompt,"stream":False,
                      "options":{"num_predict":30,"temperature":0.1}}).encode()

def ollama_procs():
    out=[]
    for p in psutil.process_iter(['name','cmdline']):
        cl=' '.join(p.info.get('cmdline') or [])
        if 'ollama' in (p.info.get('name') or '').lower() or 'ollama' in cl.lower():
            out.append(p)
    return out

peak={'rss':0}; stop=threading.Event()
def sample():
    while not stop.is_set():
        tot=0
        for p in ollama_procs():
            try: tot+=p.memory_info().rss
            except: pass
        peak['rss']=max(peak['rss'],tot)
        time.sleep(0.15)
th=threading.Thread(target=sample); th.start()

t0=time.time()
req=urllib.request.Request("http://localhost:11434/api/generate",data=payload,headers={"Content-Type":"application/json"})
resp=json.loads(urllib.request.urlopen(req,timeout=600).read())
wall=time.time()-t0
stop.set(); th.join()

pe_n=resp.get("prompt_eval_count",0); pe_d=resp.get("prompt_eval_duration",1)/1e9
ev_n=resp.get("eval_count",0);        ev_d=resp.get("eval_duration",1)/1e9
print(f"[OLLAMA plain llama.cpp q4] prompt_tokens={pe_n}")
print(f"  prefill: {pe_d:.1f}s  ({pe_n/pe_d:.0f} tok/s)")
print(f"  decode : {ev_d:.2f}s  ({ev_n/ev_d:.0f} tok/s)")
print(f"  peak_RSS(all ollama procs)={peak['rss']/1e9:.2f}GB  wall={wall:.1f}s")
print(f"  OUTPUT: {resp.get('response','')[:240]!r}")
