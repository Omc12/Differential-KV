"""Operator's console for the cloud campaign. Runs anywhere with git and the token.

    python benchmarks/cloud/ctl.py init                 # create the results branch once
    python benchmarks/cloud/ctl.py status [-v]          # tracks, machines, failures, ETA
    python benchmarks/cloud/ctl.py plan --gpus 8        # ETA if N GPUs ran what is left
    python benchmarks/cloud/ctl.py log JOB_ID           # tail of a job's log
    python benchmarks/cloud/ctl.py pause | resume | stop
    python benchmarks/cloud/ctl.py code-ref SHA         # every worker moves to SHA between jobs
    python benchmarks/cloud/ctl.py skip-track q262  /  only-tracks q131 sys  /  only-tracks
    python benchmarks/cloud/ctl.py retry ID... | release ID... | kill ID...
    python benchmarks/cloud/ctl.py merge [--partial]     # results -> paper/results/*/Y1_*

Commands write control.json on the results branch; workers read it before
every claim and every heartbeat, so a command takes effect within ~10 minutes.
`retry` / `release` / `kill` take regexes as well as ids.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import jobs as J  # noqa: E402
from gitq import GitQueue, age_min, classify, expand, now, _read, _write  # noqa: E402

DEFAULT_URL = os.environ.get("DKV_RESULTS_URL", "https://github.com/Omc12/Differential-KV.git")
DEFAULT_BRANCH = os.environ.get("DKV_RESULTS_BRANCH", "cloud-results")


def queue(a) -> GitQueue:
    return GitQueue(a.url, a.branch, a.qdir or os.path.join(
        os.path.expanduser("~"), ".dkv_cloud", "q_ctl"), who="ctl").open()


# ----------------------------------------------------------------- init ---
def cmd_init(a) -> None:
    """An orphan branch holding only control.json: workers clone it in seconds."""
    ref = a.code_ref or subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    tmp = tempfile.mkdtemp(prefix="dkv-init-")
    g = lambda *x: subprocess.run(["git", "-C", tmp] + list(x), check=True,  # noqa: E731
                                  capture_output=True, text=True)
    g("init", "-q")
    g("checkout", "-q", "--orphan", a.branch)
    _write(os.path.join(tmp, "control.json"),
           {"campaign": J.CAMPAIGN, "code_ref": ref, "paused": False, "stop": False,
            "skip": [], "skip_tracks": [], "only_tracks": [], "kill": [],
            "require_gpu": {}, "timeout_x": 1, "created": now()})
    with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("Cloud campaign state (written by benchmarks/cloud/worker.py). "
                "Do not edit by hand; use benchmarks/cloud/ctl.py.\n")
    g("add", "-A")
    g("-c", "user.name=dkv-cloud ctl", "-c", "user.email=dkv-cloud@users.noreply.github.com",
      "commit", "-q", "-m", f"cloud campaign {J.CAMPAIGN}: control")
    p = subprocess.run(["git", "-C", tmp, "push", a.url, f"HEAD:refs/heads/{a.branch}"],
                       capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if p.returncode != 0:
        raise SystemExit("push failed (does the branch exist already?):\n" + p.stderr[-800:])
    print(f"created {a.branch} with code_ref {ref[:12]}")


# -------------------------------------------------------------- control ---
def set_control(a, mutate, msg: str) -> None:
    q = queue(a)

    def fn(d):
        c = _read(os.path.join(d, "control.json")) or {}
        mutate(c)
        _write(os.path.join(d, "control.json"), c)
    q.transact(fn, "ctl: " + msg)
    print("control:", json.dumps(q.control(), sort_keys=True))


def match_ids(pats: List[str]) -> List[str]:
    ids = list(J.by_id())
    out = [i for i in ids if any(i == p or re.fullmatch(p, i) for p in pats)]
    if not out:
        raise SystemExit("no job matches " + " ".join(pats))
    return out


def cmd_jobs_op(a) -> None:
    q, ids = queue(a), match_ids(a.ids)

    def fn(d):
        for jid in ids:
            jd = os.path.join(d, "jobs", jid)
            c = _read(os.path.join(jd, "claim.json"))
            if a.cmd == "retry":
                if os.path.exists(os.path.join(jd, "done.json")):
                    os.remove(os.path.join(jd, "done.json"))
                if c:
                    c.update(attempt=0, released=True)
                    _write(os.path.join(jd, "claim.json"), c)
            elif a.cmd == "release" and c:
                c["released"] = True
                _write(os.path.join(jd, "claim.json"), c)
            elif a.cmd == "reset":
                shutil.rmtree(jd, ignore_errors=True)
        if a.cmd == "kill":
            ctl = _read(os.path.join(d, "control.json")) or {}
            ctl["kill"] = sorted(set(ctl.get("kill", [])) | set(ids))
            _write(os.path.join(d, "control.json"), ctl)
    q.transact(fn, f"ctl: {a.cmd} {len(ids)} job(s)")
    print(f"{a.cmd}: {len(ids)} job(s)")


# --------------------------------------------------------------- status ---
def ratios(jobs, states) -> Dict[str, float]:
    """Measured wall / estimated, per harness, from finished non-smoke jobs."""
    r = defaultdict(list)
    for jid, st in states.items():
        d, j = st.get("done") or {}, jobs.get(jid)
        if j and d.get("status") == "ok" and j["track"] != "smoke" and j["est_h"] > 0:
            r[j["harness"]].append(d["wall_h"] / j["est_h"])
    return {h: statistics.median(v) for h, v in r.items()}


def remaining_h(j, st, rat) -> float:
    est = j["est_h"] * rat.get(j["harness"], 1.0)
    c = st.get("claim") or {}
    if classify(st) == "running" and c.get("started"):
        return max(0.1, est - age_min(c["started"]) / 60)
    return est


def simulate(jobs, states, workers: List[float], rat) -> float:
    """List scheduling as the workers do it (prio, then longest first); returns
    hours until the last job ends. workers = VRAM of each GPU."""
    dead = {jid for jid in jobs if classify(states.get(jid, {})) == "dead"}
    # Jobs nothing here can ever run (blocked by a dead smoke, or bigger than
    # every GPU) are left out: they would make every ETA infinite.
    left = {jid: j for jid, j in jobs.items()
            if classify(states.get(jid, {})) not in ("ok", "dead")
            and not dead & set(j["needs"]) and j["min_vram_gb"] <= max(workers) + 0.5}
    doneset = {jid for jid in jobs if classify(states.get(jid, {})) == "ok"}
    free = [0.0] * len(workers)
    finish: Dict[str, float] = {}
    order = sorted(left, key=lambda i: (left[i]["prio"], -left[i]["est_h"], i))
    t_end = 0.0
    while order:
        w = min(range(len(free)), key=lambda k: free[k])
        t = free[w]
        pick = None
        for jid in order:
            j = left[jid]
            if j["min_vram_gb"] > workers[w] + 0.5:
                continue
            if all(n in doneset or finish.get(n, 1e18) <= t for n in j["needs"]):
                pick = jid
                break
        if pick is None:
            if t == float("inf"):
                return float("inf")         # every GPU is out of work it can take
            # wait for the next dependency to finish, or retire this GPU
            nxt = [f for f in finish.values() if f > t]
            free[w] = min(nxt) if nxt else float("inf")
            continue
        order.remove(pick)
        free[w] = t + remaining_h(left[pick], states.get(pick, {}), rat)
        finish[pick] = free[w]
        t_end = max(t_end, free[w])
    return t_end


def cmd_status(a) -> None:
    q = queue(a)
    jobs, states, ctl = J.by_id(), q.all_states(), q.control()
    rat = ratios(jobs, states)
    print(f"campaign {ctl.get('campaign')}  code {str(ctl.get('code_ref'))[:12]}  "
          f"{'PAUSED ' if ctl.get('paused') else ''}{'STOP ' if ctl.get('stop') else ''}"
          f"skip_tracks={ctl.get('skip_tracks')} only={ctl.get('only_tracks')}")
    by = defaultdict(Counter)
    gpuh = defaultdict(float)
    for jid, j in jobs.items():
        s = classify(states.get(jid, {}))
        if s == "pending" and not all(classify(states.get(n, {})) == "ok" for n in j["needs"]):
            s = "blocked" if any(classify(states.get(n, {})) in ("dead", "failed")
                                 for n in j["needs"]) else "waiting"
        by[j["track"]][s] += 1
        if s == "ok":
            gpuh[j["track"]] += (states[jid]["done"] or {}).get("wall_h", 0)
    cols = ["ok", "running", "stale", "pending", "waiting", "failed", "dead", "blocked"]
    print(f"\n{'track':<8}" + "".join(f"{c:>9}" for c in cols) + f"{'GPU-h':>8}")
    for t in sorted(by, key=lambda t: min(j["prio"] for j in jobs.values() if j["track"] == t)):
        print(f"{t:<8}" + "".join(f"{by[t][c]:>9}" for c in cols) + f"{gpuh[t]:>8.1f}")

    print("\nmachines:")
    alive = []
    for f in sorted(glob.glob(os.path.join(q.dir, "machines", "*.json"))):
        m = _read(f) or {}
        age = age_min(m.get("last_seen"))
        gate = ",".join(f"{k.split('/')[-1]}:{'ok' if v.get('ok') else 'FAIL'}"
                        for k, v in (m.get("fidelity") or {}).items())
        flag = "" if age < 25 else "  (SILENT)"
        if age < 25 and m.get("state") not in ("exited",):
            alive.append(m.get("total_gb", 80))
        print(f"  {m.get('name', '?'):<28} {m.get('gpu', '?'):<26} {m.get('state', '?'):<9} "
              f"{str(m.get('job') or ''):<44} {age:5.0f} min ago  jobs {m.get('jobs_done', 0)}"
              f"  [{gate}]{flag}")

    bad = [(jid, st) for jid, st in states.items()
           if jid in jobs and classify(st) in ("failed", "dead", "stale")]
    if bad:
        print("\nproblems:")
        for jid, st in sorted(bad):
            d = st.get("done") or {}
            print(f"  {classify(st):<6} {jid:<48} {d.get('status', '')} rc={d.get('rc')} "
                  f"{(st.get('claim') or {}).get('worker', '')}")
    # Same (model, length, task) data on every machine, or the pairs are not pairs.
    seen = defaultdict(set)
    for st in states.values():
        for key, v in ((st.get("done") or {}).get("data") or {}).items():
            for task, sha in (v.get("sha") or {}).items():
                seen[(key, task)].add(sha)
    split = [k for k, v in seen.items() if len(v) > 1]
    if split:
        print("\nDATA DISAGREES ACROSS MACHINES (paired results invalid): " + str(split[:10]))
    gpus = defaultdict(set)
    for jid, st in states.items():
        d = st.get("done") or {}
        if d.get("status") == "ok" and jid in jobs and jobs[jid]["track"] != "smoke":
            gpus[jobs[jid]["track"]].add(d.get("gpu"))
    mixed = {t: sorted(g) for t, g in gpus.items() if len(g) > 1}
    if mixed:
        print("\nnote: tracks measured on more than one GPU model: " + json.dumps(mixed))
    if rat:
        print("\nmeasured/estimated time: " + ", ".join(f"{h} x{r:.2f}" for h, r in rat.items()))
    if alive:
        eta = simulate(jobs, states, alive, rat)
        print(f"ETA with the {len(alive)} live GPU(s): "
              + ("never (a job fits no live GPU, or is blocked)" if eta == float("inf")
                 else f"{eta:.1f} h"))
    if a.verbose:
        for jid in sorted(states):
            st = states[jid]
            print(f"  {classify(st):<8} {jid}  {json.dumps(st.get('done') or st.get('claim'))[:160]}")


def cmd_plan(a) -> None:
    q = queue(a)
    jobs, states = J.by_id(), q.all_states()
    rat = ratios(jobs, states)
    left = sum(remaining_h(j, states.get(i, {}), rat) for i, j in jobs.items()
               if classify(states.get(i, {})) not in ("ok", "dead"))
    print(f"remaining work ~{left:.0f} GPU-h (estimates x measured ratios {rat or '-'})")
    for n in (a.gpus or [4, 8, 16, 32]):
        t = simulate(jobs, states, [a.vram] * n, rat)
        print(f"  {n:>3} x {a.vram:.0f} GB: {t:6.1f} h")


def cmd_log(a) -> None:
    q = queue(a)
    p = os.path.join(q.dir, "jobs", a.ids[0], "log.txt")
    if not os.path.exists(p):
        raise SystemExit("no log pushed yet for " + a.ids[0])
    with open(p, encoding="utf-8") as f:
        print(f.read()[-a.tail:])


# ---------------------------------------------------------------- merge ---
# Fields a shard's config records that differ between shards of one store.
SHARD_KEYS = {"datasets"}
SUB = {"ruler", "longbench", "systems", "ladder", "concurrency"}


def _rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def cmd_merge(a) -> None:
    """Shards -> one store per stem, under paper/results/<sub>/Y1_<stem>.jsonl.
    Configs must agree apart from the shard axis; a key twice is fatal."""
    q = queue(a)
    jobs, states = J.by_id(), q.all_states()
    groups = defaultdict(list)            # target store -> [shard path]
    for jid, j in jobs.items():
        if j["track"] == "smoke":
            continue
        s = classify(states.get(jid, {}))
        if s != "ok" and not (a.partial and s in ("running", "stale", "failed")):
            continue
        out_root = os.path.join(q.dir, "jobs", jid, "out")
        for rel in expand(out_root, j["outputs"]):
            src = os.path.join(out_root, rel)
            if not os.path.exists(src):
                continue
            if rel.endswith(".json") and not rel.endswith(".meta.json"):
                # per-arm files of a directory output (decode fidelity): copied as-is
                sub, d, f = rel[len(J.OUT_ROOT) + 1:].split("/")
                dst = os.path.join(REPO, "paper", "results", sub, f"{J.CAMPAIGN}_{d}", f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                print(f"  {os.path.relpath(dst, REPO)}")
                continue
            if not rel.endswith(".jsonl"):
                continue
            parts = rel[len(J.OUT_ROOT) + 1:].split("/")       # sub/stem/shard or sub/file
            sub = parts[0]
            stem = parts[1] if len(parts) == 3 else parts[1][:-len(".jsonl")]
            groups[os.path.join(REPO, "paper", "results", sub, f"{J.CAMPAIGN}_{stem}.jsonl")].append(src)
    for target, shards in sorted(groups.items()):
        metas = [_read(s + ".meta.json") for s in shards]
        base = {k: v for k, v in (metas[0] or {}).items() if k not in SHARD_KEYS}
        for s, m in zip(shards, metas):
            mm = {k: v for k, v in (m or {}).items() if k not in SHARD_KEYS}
            if mm != base:
                diff = {k for k in set(mm) | set(base) if mm.get(k) != base.get(k)}
                raise SystemExit(f"config differs between shards of {target}: {s} ({diff})")
        rows, keys = [], {}
        for s in shards:
            for r in _rows(s):
                k = r.get("key")
                if k in keys:
                    if keys[k] != r:
                        print(f"  note: {k} recorded twice in {os.path.basename(target)}; "
                              f"keeping the later row")
                keys[k] = r
                rows.append(r)
        meta = dict(base)
        ds = sorted({d for m in metas for d in ((m or {}).get("datasets") or [])})
        if ds:
            meta["datasets"] = ds
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        _write(target + ".meta.json", meta)
        print(f"  {os.path.relpath(target, REPO)}: {len(keys)} rows from {len(shards)} shard(s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "status", "plan", "log", "pause", "resume", "stop",
                                    "unstop", "code-ref", "skip-track", "unskip-track",
                                    "only-tracks", "require-gpu", "timeout-x",
                                    "retry", "release", "kill", "reset", "merge"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--qdir", default="")
    ap.add_argument("--code-ref", default="")
    ap.add_argument("--gpus", type=int, nargs="*", default=None)
    ap.add_argument("--vram", type=float, default=80)
    ap.add_argument("--tail", type=int, default=6000)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    c = a.cmd
    if c == "init":
        return cmd_init(a)
    if c == "status":
        return cmd_status(a)
    if c == "plan":
        return cmd_plan(a)
    if c == "log":
        return cmd_log(a)
    if c == "merge":
        return cmd_merge(a)
    if c in ("retry", "release", "kill", "reset"):
        return cmd_jobs_op(a)
    ops = {
        "pause": lambda x: x.update(paused=True),
        "resume": lambda x: x.update(paused=False),
        "stop": lambda x: x.update(stop=True),
        "unstop": lambda x: x.update(stop=False),
        "code-ref": lambda x: x.update(code_ref=a.ids[0]),
        "skip-track": lambda x: x.update(skip_tracks=sorted(set(x.get("skip_tracks", [])) | set(a.ids))),
        "unskip-track": lambda x: x.update(skip_tracks=sorted(set(x.get("skip_tracks", [])) - set(a.ids))),
        "only-tracks": lambda x: x.update(only_tracks=list(a.ids)),
        "require-gpu": lambda x: x.setdefault("require_gpu", {}).update({a.ids[0]: a.ids[1]}),
        "timeout-x": lambda x: x.update(timeout_x=float(a.ids[0])),
    }
    set_control(a, ops[c], c + " " + " ".join(a.ids))


if __name__ == "__main__":
    main()
