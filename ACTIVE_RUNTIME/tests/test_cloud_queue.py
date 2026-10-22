"""benchmarks/cloud: the git-branch queue, the worker's claim/resume/release
cycle and the shard merge. CPU-only; a local bare repository stands in for
GitHub, so pushes race exactly as they would there.

The failure modes these pin are the expensive ones on rented GPUs: two
machines running one job, a dead machine's job never restarting, a restarted
job not resuming from what was pushed, and shards of one store merging with a
different configuration.
"""
import json
import os
import subprocess
import sys
import threading
import types

import pytest

CLOUD = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "cloud")
sys.path.insert(0, os.path.abspath(CLOUD))

import ctl  # noqa: E402
import gitq  # noqa: E402
import jobs as J  # noqa: E402
import worker as W  # noqa: E402


def _git_ok():
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _git_ok(), reason="git not on PATH")


@pytest.fixture
def remote(tmp_path):
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    a = types.SimpleNamespace(url=str(bare), branch="cloud-results", code_ref="abc123", qdir="")
    ctl.cmd_init(a)
    return str(bare)


def _q(remote, tmp_path, name):
    return gitq.GitQueue(remote, "cloud-results", str(tmp_path / ("q_" + name)), who=name).open()


def test_only_one_of_two_racing_claims_lands(remote, tmp_path):
    qa, qb = _q(remote, tmp_path, "a"), _q(remote, tmp_path, "b")
    won = []

    def claim(q, who):
        def fn(d):
            p = os.path.join(d, "jobs", "j1", "claim.json")
            if os.path.exists(p):
                return None
            gitq._write(p, {"worker": who, "heartbeat": gitq.now()})
            return who
        r = q.transact(fn, who + " claim")
        if r:
            won.append(r)

    ts = [threading.Thread(target=claim, args=(q, n)) for q, n in ((qa, "a"), (qb, "b"))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(won) == 1
    qa.sync()
    assert gitq._read(qa.path("jobs", "j1", "claim.json"))["worker"] == won[0]


def test_classify():
    old = "2000-01-01T00:00:00Z"
    assert gitq.classify({}) == "pending"
    assert gitq.classify({"claim": {"heartbeat": gitq.now()}}) == "running"
    assert gitq.classify({"claim": {"heartbeat": old}}) == "stale"
    assert gitq.classify({"claim": {"heartbeat": old, "released": True}}) == "pending"
    assert gitq.classify({"done": {"status": "ok"}}) == "ok"
    assert gitq.classify({"claim": {"attempt": 1}, "done": {"status": "failed"}}) == "failed"
    assert gitq.classify({"claim": {"attempt": 2}, "done": {"status": "failed"}}) == "dead"


# A job that appends one row per step to a checkpoint-style store and resumes
# from whatever rows are already there -- what every real harness does.
STEP = r"""
import json, os, sys, time
out, n, crash_at = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(os.path.dirname(out), exist_ok=True)
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out) if l.strip()}
with open(out + ".meta.json", "w") as f:
    json.dump({"cfg": 1, "datasets": [os.path.basename(out)]}, f)
for i in range(n):
    if str(i) in done:
        continue
    if i == crash_at:
        sys.exit(3)
    with open(out, "a") as f:
        f.write(json.dumps({"key": str(i), "env": os.environ.get("X_SEEN")}) + "\n")
print("resumed from", len(done))
"""


def _fake_jobs(out_a, out_b, crash_at=-1):
    def job(jid, out, needs=()):
        return J.Job(id=jid, track="t", prio=1, est_h=0.01, min_vram_gb=1, needs=list(needs),
                     data=[], timing=False, sig="s", model="m", arm="x",
                     harness="fake.py", argv=["-c", STEP, out, "5", str(crash_at)],
                     env={"X_SEEN": "{ruler_data}"},
                     outputs=[out, out + ".meta.json"])
    return {"smoke.s": job("smoke.s", "paper/results/cloud/Y1/smoke/s.jsonl"),
            "t.a": job("t.a", out_a, ["smoke.s"]), "t.b": job("t.b", out_b, ["smoke.s"])}


def _worker(monkeypatch, remote, tmp_path, name, jobs, repo):
    monkeypatch.setattr(W, "REPO", str(repo))
    monkeypatch.setattr(W.J, "by_id", lambda: jobs)
    monkeypatch.setattr(W, "gpu_info", lambda g: {"gpu": "FakeGPU", "total_gb": 80.0,
                                                   "uuid": name})
    monkeypatch.setattr(W, "code_ref", lambda: "abc123")
    a = types.SimpleNamespace(gpu="0", name=name, repo_url=remote, branch="cloud-results",
                              qdir=str(tmp_path / ("wq_" + name)), ruler_data="RD",
                              only="", max_jobs=0, exit_when_idle=True, no_preflight=True,
                              manage_code=False)
    return W.Worker(a)


def test_worker_runs_dependencies_in_order_and_pushes(monkeypatch, remote, tmp_path):
    repo = tmp_path / "repo"
    jobs = _fake_jobs("paper/results/cloud/Y1/ruler/S/a.jsonl",
                      "paper/results/cloud/Y1/ruler/S/b.jsonl")
    w = _worker(monkeypatch, remote, tmp_path, "w1", jobs, repo)
    w.loop()
    q = _q(remote, tmp_path, "check")
    st = q.all_states()
    assert {k: gitq.classify(v) for k, v in st.items()} == {"smoke.s": "ok", "t.a": "ok", "t.b": "ok"}
    out = q.path("jobs", "t.a", "out", "paper/results/cloud/Y1/ruler/S/a.jsonl")
    rows = [json.loads(l) for l in open(out)]
    assert [r["key"] for r in rows] == ["0", "1", "2", "3", "4"]
    assert rows[0]["env"] == "RD"                # placeholders resolved into the env
    assert gitq._read(q.path("machines", "w1.json"))["state"] == "exited"


def test_failed_job_resumes_from_pushed_rows_on_another_machine(monkeypatch, remote, tmp_path):
    jobs = _fake_jobs("paper/results/cloud/Y1/ruler/S/a.jsonl",
                      "paper/results/cloud/Y1/ruler/S/b.jsonl", crash_at=3)
    w1 = _worker(monkeypatch, remote, tmp_path, "w1", jobs, tmp_path / "r1")
    w1.loop()                                    # smoke dies at row 3 twice -> dead
    q = _q(remote, tmp_path, "check")
    assert gitq.classify(q.all_states()["smoke.s"]) == "dead"
    assert "t.a" not in q.all_states()           # never claimed: its smoke failed
    pushed = q.path("jobs", "smoke.s", "out", "paper/results/cloud/Y1/smoke/s.jsonl")
    assert len(open(pushed).readlines()) == 3

    # The bug is fixed; the operator retries. A different machine, with an
    # empty checkout, must resume at row 3, not start over.
    ctl.cmd_jobs_op(types.SimpleNamespace(cmd="retry", ids=["smoke.s"], url=remote,
                                          branch="cloud-results", qdir=str(tmp_path / "qc")))
    fixed = _fake_jobs("paper/results/cloud/Y1/ruler/S/a.jsonl",
                       "paper/results/cloud/Y1/ruler/S/b.jsonl")
    w2 = _worker(monkeypatch, remote, tmp_path, "w2", fixed, tmp_path / "r2")
    w2.loop()
    q.sync()
    st = {k: gitq.classify(v) for k, v in q.all_states().items()}
    assert st == {"smoke.s": "ok", "t.a": "ok", "t.b": "ok"}
    assert "resumed from 3" in open(q.path("jobs", "smoke.s", "log.txt")).read()


def test_stale_claim_is_taken_over_and_the_old_owner_stands_down(monkeypatch, remote, tmp_path):
    jobs = _fake_jobs("paper/results/cloud/Y1/ruler/S/a.jsonl",
                      "paper/results/cloud/Y1/ruler/S/b.jsonl")
    q = _q(remote, tmp_path, "ghost")

    def ghost(d):                     # a machine that claimed and then vanished
        gitq._write(os.path.join(d, "jobs", "smoke.s", "claim.json"),
                    {"worker": "ghost", "token": "ghost:1", "attempt": 1,
                     "heartbeat": "2000-01-01T00:00:00Z"})
    q.transact(ghost, "ghost claim")
    w = _worker(monkeypatch, remote, tmp_path, "w1", jobs, tmp_path / "r1")
    w.loop()
    q.sync()
    assert gitq.classify(q.all_states()["smoke.s"]) == "ok"
    assert gitq._read(q.path("jobs", "smoke.s", "claim.json"))["previous"] == "ghost"
    # the ghost comes back and tries to push: it must see the claim is not its own
    w.token = "ghost:1"
    assert not w._still_mine(q.dir, "smoke.s")


def test_sigterm_releases_the_claim_for_immediate_restart(monkeypatch, remote, tmp_path):
    jobs = _fake_jobs("paper/results/cloud/Y1/ruler/S/a.jsonl",
                      "paper/results/cloud/Y1/ruler/S/b.jsonl")
    jobs["smoke.s"]["argv"] = ["-c", "import time; time.sleep(60)"]
    w = _worker(monkeypatch, remote, tmp_path, "w1", jobs, tmp_path / "r1")
    threading.Timer(4.0, lambda: setattr(w, "stop_flag", True)).start()
    jid = w.claim_next()
    w.run(jid)
    q = _q(remote, tmp_path, "check")
    st = q.all_states()["smoke.s"]
    assert st["claim"]["released"] and gitq.classify(st) == "pending"


def test_merge_joins_shards_and_refuses_mixed_configs(monkeypatch, remote, tmp_path):
    jobs = _fake_jobs("paper/results/cloud/Y1/ruler/S/a.jsonl",
                      "paper/results/cloud/Y1/ruler/S/b.jsonl")
    _worker(monkeypatch, remote, tmp_path, "w1", jobs, tmp_path / "r1").loop()
    monkeypatch.setattr(ctl, "REPO", str(tmp_path / "local"))
    monkeypatch.setattr(ctl.J, "by_id", lambda: jobs)
    a = types.SimpleNamespace(url=remote, branch="cloud-results", qdir=str(tmp_path / "qm"),
                              partial=False)
    ctl.cmd_merge(a)
    merged = tmp_path / "local" / "paper" / "results" / "ruler" / "Y1_S.jsonl"
    assert len(open(merged).readlines()) == 10       # 5 rows x 2 shards
    meta = json.load(open(str(merged) + ".meta.json"))
    assert meta["datasets"] == ["a.jsonl", "b.jsonl"] and meta["cfg"] == 1

    q = ctl.queue(a)

    def poison(d):
        p = os.path.join(d, "jobs", "t.b", "out", "paper/results/cloud/Y1/ruler/S/b.jsonl.meta.json")
        gitq._write(p, {"cfg": 2})
    q.transact(poison, "poison")
    with pytest.raises(SystemExit, match="config differs"):
        ctl.cmd_merge(a)


def test_many_workers_claiming_at_once_never_share_a_job(monkeypatch, remote, tmp_path):
    base = _fake_jobs("a.jsonl", "b.jsonl")["smoke.s"]
    jobs = {f"t.{i}": J.Job({**base, "id": f"t.{i}", "needs": [],
                             "outputs": [f"o{i}.jsonl"]}) for i in range(10)}
    ws = [_worker(monkeypatch, remote, tmp_path, f"w{i}", jobs, tmp_path / f"r{i}")
          for i in range(6)]
    got = {}
    ts = [threading.Thread(target=lambda w=w: got.__setitem__(w.name, w.claim_next()))
          for w in ws]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(got.values())) == 6 and None not in got.values()
    q = _q(remote, tmp_path, "check")
    owners = {jid: st["claim"]["worker"] for jid, st in q.all_states().items()}
    assert owners == {jid: w for w, jid in got.items()}


def test_manifest_is_consistent():
    js = J.all_jobs()                               # check() asserts uniqueness, deps
    real = [j for j in js if j["track"] != "smoke"]
    assert all(j["needs"] and j["needs"][0].startswith("smoke.") for j in real)
    # every dkv job that compresses on the 12.9 GB card's line, or says it does not
    for j in real:
        if j["arm"].startswith("dkv") and j["env"].get("DKV_STREAM_ELASTIC"):
            card12 = "DKV_STREAM_ELASTIC_RESERVE_GB" in j["env"]
            assert card12 != (j["arm"] in ("dkv_native", "dkv_exact")), j["id"]


def test_card12_reserve_reproduces_the_papers_line():
    for total in (12.878, 24.0, 79.6, 85.0, 141.0):
        r = float(W.placeholders({"total_gb": total}, "x")["card12_reserve_gb"])
        assert abs((0.94 * total - r) - (0.94 * 12.878 - 3.5)) < 1e-3
