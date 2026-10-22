"""One GPU's worker for the cloud campaign: claim a job, run it, hand it back.

    python benchmarks/cloud/worker.py --gpu 0 --repo-url URL [--manage-code]

THE LOOP
    1. preflight once per code version: GPU facts, then logit fidelity of every
       campaign model against plain HF (colab/logit_fidelity.py; the readiness
       bar this repo learned the hard way). A model that fails is never run on
       this GPU, and the failure is visible in machines/<worker>.json.
    2. read control.json: pause, stop, a new code_ref (finish the job in hand,
       check the new code out, restart), tracks to skip, jobs to kill.
    3. claim the best job this GPU can take: lowest prio, then longest first,
       its smoke job passed, enough VRAM, not running anywhere else.
    4. run it as a child process; every 10 minutes push its output files and
       log tail and refresh the claim's heartbeat. If this machine dies, the
       claim goes stale after gitq.STALE_MIN minutes, another worker takes it,
       restores the pushed files and the harness RESUMES from them (every
       harness here is checkpointed), so a dead machine costs <= 10 minutes.
    5. on exit push the final files and done.json (status, rows, GPU, code).

SIGTERM (spot preemption, `systemctl stop`) stops the child, pushes what it has
and releases the claim so the job restarts elsewhere at once.

Only --manage-code lets the worker move its checkout to control.json's
code_ref. Without it (a developer's own tree) a code_ref mismatch is an error.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gitq  # noqa: E402
from gitq import GitQueue, classify, copy_into, now, _read, _write  # noqa: E402
import jobs as J  # noqa: E402

HEARTBEAT_S = 600
LOG_TAIL = 200_000
CARD12_GB = 12.878          # the paper's card, as torch reports it (bytes / 1e9)


def log(msg: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_ref() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def gpu_info(gpu: str) -> Dict[str, Any]:
    probe = ("import json, torch, platform;"
             "p = torch.cuda.get_device_properties(0);"
             "print(json.dumps({'gpu': p.name, 'total_gb': p.total_memory / 1e9,"
             "'torch': torch.__version__, 'cuda': torch.version.cuda,"
             "'python': platform.python_version(), 'os': platform.platform()}))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit("no usable CUDA device:\n" + p.stderr[-1500:])
    info = json.loads(p.stdout.strip().splitlines()[-1])
    try:
        q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=driver_version,uuid",
                            "--format=csv,noheader"], capture_output=True, text=True)
        info["driver"], info["uuid"] = [s.strip() for s in q.stdout.strip().split(",")][:2]
    except Exception:                                            # noqa: BLE001
        pass
    return info


# ----------------------------------------------------------------- jobs ---
def placeholders(info: Dict[str, Any], ruler_root: str) -> Dict[str, str]:
    total = info["total_gb"]
    return {
        # the elastic window's line where the 12.878 GB card had it (jobs.py)
        "card12_reserve_gb": f"{0.94 * total - (0.94 * CARD12_GB - 3.5):.3f}",
        "pool_gb": f"{0.8 * total / 1.073741824:.1f}",      # DKV_POOL_BUDGET_GB is GiB
        "ruler_data": ruler_root.replace("\\", "/"),
    }


def resolve(job: Dict[str, Any], ph: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    def sub(s: str) -> str:
        for k, v in ph.items():
            s = s.replace("{" + k + "}", v)
        return s
    argv = [sys.executable] + [sub(a) for a in job["argv"]]
    # A clean environment: nothing from the shell that started us may steer DKV.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DKV_", "RULER_", "PYTORCH_CUDA_ALLOC"))}
    env.update({k: sub(v) for k, v in job["env"].items()})
    return argv, env


def count_rows(path: str) -> Tuple[int, int]:
    keys, errs = set(), 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue            # a torn last line; checkpoint drops it too
                keys.add(r.get("key"))
                errs += bool(r.get("error"))
    except OSError:
        pass
    return len(keys), errs


def expected_rows(job: Dict[str, Any]) -> int:
    """RULER shards know their count exactly; others are checked for any rows."""
    a = job["argv"]
    if job["harness"] == "decode_fidelity.py":        # one .json per arm that ran
        i = a.index("--arms") + 1
        return sum(1 for x in a[i:] if not x.startswith("--"))
    if job["harness"] != "run_ruler_cuda.py" or "--per-task" not in a:
        return 1
    n = int(a[a.index("--per-task") + 1])
    i = a.index("--tasks") + 1
    k = i
    while k < len(a) and not a[k].startswith("--"):
        k += 1
    return n * (k - i)


def kill_tree(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
    except Exception:                                            # noqa: BLE001
        p.kill()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        pass


# ------------------------------------------------------------ preflight ---
FID_ROW = re.compile(r"^\s*(\S+)\s+(\d+)/(\d+)\s+(-?[\d.]+)")


def fidelity_gate(model: str, gpu: str, ctx: int = 8192) -> Tuple[bool, str]:
    """dense_true's own decode-step KL must read ~0 (positions aligned) and DKV
    must pick dense's top-1 on every prompt with KL < 0.05."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DKV_", "RULER_"))}
    env.update(J.BASE_ENV, CUDA_VISIBLE_DEVICES=gpu)
    p = subprocess.run([sys.executable, "colab/logit_fidelity.py", "--model", model,
                        "--arms", "dense_true", "baseline", "--quant", "nf4",
                        "--ctx", str(ctx)], cwd=REPO, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    rows = {}
    for line in p.stdout.splitlines():
        m = FID_ROW.match(line)
        if m:
            rows[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    self_chk, base = rows.get("dense_true/dec"), rows.get("baseline")
    detail = json.dumps({k: list(v) for k, v in rows.items()})
    if p.returncode != 0 or not self_chk or not base:
        return False, f"rc={p.returncode} rows={detail} tail={(p.stderr or p.stdout)[-800:]}"
    agree, n, kl = base
    ok = self_chk[2] < 1e-3 and agree == n and kl < 0.05
    return ok, detail


# --------------------------------------------------------------- worker ---
class Worker:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.gpu = a.gpu
        self.name = a.name or f"{socket.gethostname()}-g{a.gpu}"
        self.jobs = J.by_id()
        self.ref = code_ref()
        self.info = gpu_info(self.gpu)
        self.ph = placeholders(self.info, a.ruler_data)
        self.q = GitQueue(a.repo_url, a.branch, a.qdir or os.path.join(
            os.path.expanduser("~"), ".dkv_cloud", f"q_{self.name}"), who=self.name).open()
        self.gate: Dict[str, Dict[str, Any]] = {}
        self.stop_flag = False
        self.state = "starting"
        self.current: Optional[str] = None
        self.done_count = 0
        signal.signal(signal.SIGTERM, self._on_term)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._on_term)

    def _on_term(self, *_):
        log("SIGTERM: stopping after pushing what the current job has")
        self.stop_flag = True

    # -- machine file
    def _machine(self, d: str) -> None:
        _write(os.path.join(d, "machines", self.name + ".json"), {
            "name": self.name, "host": socket.gethostname(), "gpu_index": self.gpu,
            **self.info, "code_ref": self.ref, "state": self.state, "job": self.current,
            "last_seen": now(), "fidelity": self.gate, "jobs_done": self.done_count})

    def beat(self, state: Optional[str] = None) -> None:
        if state:
            self.state = state
        self.q.transact(self._machine, f"{self.name}: {self.state}")

    # -- preflight
    def preflight(self) -> None:
        cache = os.path.join(os.path.expanduser("~"), ".dkv_cloud",
                             f"fidelity_{self.ref[:12]}_{self.info.get('uuid', self.gpu)}.json")
        self.gate = _read(cache) or {}
        if self.a.no_preflight:
            return
        self.beat("preflight")
        for model in sorted({j["model"] for j in self.jobs.values()}):
            if model in self.gate:
                continue
            log(f"fidelity gate: {model}")
            t0 = time.time()
            ok, detail = fidelity_gate(model, self.gpu)
            self.gate[model] = {"ok": ok, "detail": detail[:1500],
                                "s": round(time.time() - t0)}
            log(f"  {model}: {'PASS' if ok else 'FAIL'} {detail[:300]}")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            _write(cache, self.gate)
        self.beat("ready")

    # -- choosing
    def eligible(self, jid: str, states: Dict[str, Dict], ctl: Dict) -> bool:
        j = self.jobs[jid]
        if jid in ctl.get("skip", []) or j["track"] in ctl.get("skip_tracks", []):
            return False
        only = ctl.get("only_tracks") or []
        if only and j["track"] not in only and j["track"] != "smoke":
            return False
        if self.a.only and not re.search(self.a.only, jid):
            return False
        if j["min_vram_gb"] > self.info["total_gb"] + 0.5:
            return False
        req = (ctl.get("require_gpu") or {}).get(j["track"])
        if req and req.lower() not in self.info["gpu"].lower():
            return False
        g = self.gate.get(j["model"])
        if g is not None and not g.get("ok"):
            return False
        if classify(states.get(jid, {})) not in ("pending", "failed", "stale"):
            return False
        return all(classify(states.get(n, {})) == "ok" for n in j["needs"])

    def claim_next(self) -> Optional[str]:
        token = f"{self.name}:{time.time():.0f}"

        def fn(d: str) -> Optional[str]:
            states = self.q.all_states()
            ctl = self.q.control()
            pick = sorted((jid for jid in self.jobs if self.eligible(jid, states, ctl)),
                          key=lambda jid: (self.jobs[jid]["prio"], -self.jobs[jid]["est_h"], jid))
            if not pick:
                return None
            # Every worker ranks alike, so all of them would race for pick[0]
            # and retry in lockstep; drawing from the head of the same priority
            # spreads the claims and keeps longest-first nearly intact.
            top = [p for p in pick[:4] if self.jobs[p]["prio"] == self.jobs[pick[0]]["prio"]]
            jid = random.choice(top)
            st = states.get(jid, {})
            old = st.get("claim") or {}
            attempt = old.get("attempt", 0) + (0 if old.get("released") else 1)
            jd = os.path.join(d, "jobs", jid)
            if os.path.exists(os.path.join(jd, "done.json")):
                os.remove(os.path.join(jd, "done.json"))
            _write(os.path.join(jd, "claim.json"), {
                "worker": self.name, "token": token, "gpu": self.info["gpu"],
                "started": now(), "heartbeat": now(), "attempt": attempt,
                "code_ref": self.ref, "previous": old.get("worker")})
            self.current, self.state = jid, "running"
            self._machine(d)
            return jid

        jid = self.q.transact(fn, f"{self.name}: claim")
        self.token = token
        return jid

    # -- running
    def _still_mine(self, d: str, jid: str) -> bool:
        c = _read(os.path.join(d, "jobs", jid, "claim.json")) or {}
        return c.get("token") == self.token

    def push(self, jid: str, logf: str, final: Optional[Dict] = None,
             release: bool = False) -> str:
        job = self.jobs[jid]

        def fn(d: str) -> str:
            if not self._still_mine(d, jid):
                return "lost"
            jd = os.path.join(d, "jobs", jid)
            copy_into(REPO, job["outputs"], os.path.join(jd, "out"))
            try:
                with open(logf, "rb") as f:
                    f.seek(max(0, os.path.getsize(logf) - LOG_TAIL))
                    tail = f.read().decode("utf-8", "replace")
                with open(os.path.join(jd, "log.txt"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(tail)
            except OSError:
                pass
            c = _read(os.path.join(jd, "claim.json"))
            c["heartbeat"] = now()
            if release:
                c["released"] = True
            _write(os.path.join(jd, "claim.json"), c)
            if final is not None:
                _write(os.path.join(jd, "done.json"), final)
            self._machine(d)
            return "ok"

        what = "done " + final["status"] if final else ("release" if release else "beat")
        return self.q.transact(fn, f"{self.name}: {jid} {what}")

    def restore(self, jid: str) -> int:
        """Bring back what a previous attempt pushed, so the harness resumes.
        A local copy that is larger (this machine's own earlier attempt, never
        pushed in full) wins."""
        job, n = self.jobs[jid], 0
        src = os.path.join(self.q.dir, "jobs", jid, "out")
        for rel in gitq.expand(src, job["outputs"]):
            s, d = os.path.join(src, rel), os.path.join(REPO, rel)
            if os.path.isfile(s) and (not os.path.isfile(d) or os.path.getsize(s) > os.path.getsize(d)):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(s, d)
                n += 1
            lock = d + ".lock"
            if os.path.exists(lock):
                os.remove(lock)     # ours, from a run that is certainly dead now
        return n

    def run(self, jid: str) -> None:
        job = self.jobs[jid]
        argv, env = resolve(job, self.ph)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        logd = os.path.join(REPO, ".cloud_logs")
        os.makedirs(logd, exist_ok=True)
        logf = os.path.join(logd, jid + ".log")
        t0 = time.time()
        data_check: Dict[str, Any] = {}
        status, rc = "ok", None
        with open(logf, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {now()} {self.name} {self.info['gpu']} code {self.ref[:12]}\n"
                     f"argv: {json.dumps(argv)}\nenv: {json.dumps(job['env'])}\n"
                     f"resolved: {json.dumps(self.ph)}\n")
            lf.flush()
            try:
                import ruler_data
                for model, L in job["data"]:
                    shas = ruler_data.ensure(model, L, self.a.ruler_data, verbose=False)
                    v = ruler_data.verify(model, L, shas)
                    data_check[f"{model}@{L}"] = {"sha": shas, "check": v}
                    if "MISMATCH" in v.values():
                        raise RuntimeError(f"RULER data for {model}@{L} differs from the "
                                           f"paper's items: {v}")
            except Exception as e:                               # noqa: BLE001
                lf.write("DATA FAILED: " + "".join(traceback.format_exception(e)))
                status = "failed"
            n = self.restore(jid)
            lf.write(f"restored {n} file(s) from earlier attempts\n")
            lf.flush()
            limit_s = (4 * job["est_h"] + 3) * 3600 * float(self.q.control().get("timeout_x", 1))
            if status == "ok":
                kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                      else {"start_new_session": True})
                p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT, **kw)
                last = time.time()
                while True:
                    try:
                        rc = p.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if self.stop_flag:
                        kill_tree(p)
                        lf.flush()
                        self.push(jid, logf, release=True)
                        log(f"{jid}: released after SIGTERM")
                        return
                    if time.time() - t0 > limit_s:
                        kill_tree(p)
                        status, rc = "timeout", p.returncode
                        break
                    if time.time() - last > HEARTBEAT_S:
                        last = time.time()
                        lf.flush()
                        if self.push(jid, logf) == "lost":
                            kill_tree(p)
                            log(f"{jid}: claim taken over elsewhere; stopped")
                            return
                        if jid in self.q.control().get("kill", []):
                            kill_tree(p)
                            status, rc = "killed", p.returncode
                            break
                if status == "ok" and rc != 0:
                    status = "failed"
        files = gitq.expand(REPO, job["outputs"])
        rows = {o: count_rows(os.path.join(REPO, o)) for o in files if o.endswith(".jsonl")}
        n_rows = sum(r[0] for r in rows.values())
        n_err = sum(r[1] for r in rows.values())
        if job["harness"] == "decode_fidelity.py":
            # one .json per arm that finished (the summary .jsonl is one line)
            n_rows = sum(1 for o in files if o.endswith(".json"))
        if status == "ok" and (n_rows < expected_rows(job) or (n_rows and n_err == n_rows)):
            status = "incomplete"
        final = {"status": status, "rc": rc, "wall_h": round((time.time() - t0) / 3600, 3),
                 "worker": self.name, "gpu": self.info["gpu"], "vram_gb": self.info["total_gb"],
                 "driver": self.info.get("driver"), "torch": self.info.get("torch"),
                 "code_ref": self.ref, "finished": now(), "rows": n_rows, "row_errors": n_err,
                 "expected_rows": expected_rows(job), "resolved": self.ph,
                 "data": data_check, "est_h": job["est_h"]}
        r = self.push(jid, logf, final=final)
        self.done_count += status == "ok"
        log(f"{jid}: {status} rc={rc} rows={n_rows} err={n_err} "
            f"{final['wall_h']:.2f} h{' (claim lost)' if r == 'lost' else ''}")

    # -- code updates
    def maybe_update_code(self, ctl: Dict) -> None:
        want = ctl.get("code_ref")
        if not want or self.ref.startswith(want) or want.startswith(self.ref):
            return
        if not self.a.manage_code:
            raise SystemExit(f"control.json wants code {want}, this checkout is "
                             f"{self.ref[:12]}; run with --manage-code or check it out")
        log(f"moving code {self.ref[:12]} -> {want}")
        self.beat("updating")
        subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin"], check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "--detach", want], check=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def loop(self) -> None:
        ctl = self.q.control()
        self.maybe_update_code(ctl)
        self.preflight()
        idle_since = None
        while not self.stop_flag:
            self.q.sync()
            ctl = self.q.control()
            self.maybe_update_code(ctl)
            if ctl.get("stop") or self.name in ctl.get("stop_workers", []):
                log("control says stop")
                break
            if ctl.get("paused"):
                self.current = None
                self.beat("paused")
                time.sleep(120)
                continue
            jid = self.claim_next()
            if not jid:
                states = self.q.all_states()
                left = [k for k in self.jobs if classify(states.get(k, {})) not in ("ok", "dead")]
                if not left or self.a.exit_when_idle:
                    log("nothing left for this GPU" if not left else "idle; exiting")
                    break
                self.current = None
                if idle_since is None:
                    idle_since = time.time()
                    log(f"no claimable job ({len(left)} not finished: running elsewhere, "
                        f"waiting on a smoke job, or too big for this GPU); polling")
                self.beat("idle")
                time.sleep(300)
                continue
            idle_since = None
            log(f"claimed {jid} (est {self.jobs[jid]['est_h']} h)")
            self.run(jid)
            self.current = None
            if self.a.max_jobs and self.done_count >= self.a.max_jobs:
                break
        self.current = None
        self.beat("exited")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default="")
    ap.add_argument("--repo-url", default=os.environ.get("DKV_RESULTS_URL", ""))
    ap.add_argument("--branch", default=os.environ.get("DKV_RESULTS_BRANCH", "cloud-results"))
    ap.add_argument("--qdir", default="")
    ap.add_argument("--ruler-data", default=os.environ.get("DKV_RULER_DATA") or
                    os.path.join(REPO, "paper", "results", "ruler_data"))
    ap.add_argument("--only", default="", help="regex: claim only matching job ids")
    ap.add_argument("--max-jobs", type=int, default=0)
    ap.add_argument("--exit-when-idle", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--manage-code", action="store_true")
    a = ap.parse_args()
    if not a.repo_url:
        raise SystemExit("--repo-url (or DKV_RESULTS_URL) is required")
    Worker(a).loop()


if __name__ == "__main__":
    main()
