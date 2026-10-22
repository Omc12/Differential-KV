# Cloud campaign Y1 (sponsored GPUs, MLSys 2027)

Many GPU machines run the paper's remaining experiments in parallel. **GitHub
is the only coordinator.** The code comes from `main`. The work queue is the
orphan branch `cloud-results`. A machine claims a job by pushing a claim file.
GitHub accepts only one push on top of a given tip, so two machines can never
claim the same job. Results, logs and heartbeats go back to the same branch.
Nothing else has to stay up.

```
 laptop (you)                       GitHub                         GPU machines
 ctl.py status/merge   <---->   main (code, pinned sha)   <----   bootstrap.sh clones it
 ctl.py pause/code-ref --->   cloud-results (queue)      <---->  worker.py per GPU:
                                control.json                      claim -> run -> push
                                jobs/<id>/{claim,done,out,log}    every 10 min + at end
                                machines/<worker>.json
```

| file | role |
|---|---|
| `benchmarks/cloud/jobs.py` | the manifest: every job, its harness command, env, outputs, cost |
| `benchmarks/cloud/worker.py` | one per GPU: preflight, claim, run, heartbeat, resume, release |
| `benchmarks/cloud/gitq.py` | the queue on a git branch (compare-and-swap by push) |
| `benchmarks/cloud/ctl.py` | operator console: init, status, plan, commands, merge |
| `benchmarks/cloud/ruler_data.py` | RULER data generated on each machine, hash-checked against the paper's items |
| `benchmarks/cloud/bootstrap.sh` | fresh Linux GPU machine to running workers, one command |
| `ACTIVE_RUNTIME/tests/test_cloud_queue.py` | claim race, takeover, resume, release, merge (CPU, local bare repo) |

## What it measures

`python benchmarks/cloud/jobs.py summary` prints the current numbers. Hours
are estimates from the 12.9 GB card divided by 2.5. `ctl.py plan` rescales
them from finished jobs.

1,123 jobs, ~734 GPU-h estimated: P1 262, P2 378, P3 80, smoke 14.

| track | prio | what | why |
|---|---|---|---|
| `smoke` | 0 | each of the 140 distinct (harness, model, arm, env) configs at a tiny size | a broken config fails in minutes, and its long jobs are never claimed |
| `q131` | 1 | Qwen3.5-4B RULER at 131k, 16 arms, 20 items/task (was 5), all 13 tasks with the answer prefix, **dense included** | limitation "small samples"; first exact reference past 65k |
| `sys` | 1 | TTFT, decode tok/s, peak alloc/reserved, KV bytes vs context, 14 arms × 2 models, Linux, Inductor compiled | MLSys systems table on a datacenter GPU (the old ones were Windows/WDDM) |
| `conc` | 1 | sessions held resident per arm at 16k, 32k and 131k | the serving question: users per GPU |
| `qshort` | 2 | Qwen3.5-4B 4k–64k, the paper's main RULER table: 10 arms, 20 items | on the fixed harness (answer prefix) |
| `q32x` | 2 | Qwen3.5-4B 32k: forced streaming (elastic and fixed window), routing off | the paper's tiling and routing ablations |
| `gshort`, `g24`, `g32`, `g131` | 2 | granite: dense row to 16k; 24k and 32k with 10 arms (20 items/task); 131k with 8 arms (10/task) | the dense-GQA model, past its 12.9 GB ceiling (the paper's 114k becomes the model's 131k) |
| `lb` | 2 | LongBench, all 18 arms of the paper's table, 200 items/dataset (was 20) | resolves the LongBench deltas |
| `lad` | 2 | memory ladders, all 19 ladder series × 2 models, to each model's max | peak memory vs context on the big card |
| `diag` | 2 | the four allocator traces; decode fidelity (routing check on granite 24k–32k, exact/forced/hybrid on Qwen 32k) | the paper's instrument and fidelity claims |
| `q262` | 3 | RULER at 262,144 (Qwen3.5-4B's max), 5 arms | the paper scored only to 131k |
| `q9b` | 3 | Qwen3.5-9B RULER 131k + systems | a third model |

**Coverage of what already ran on the 12.9 GB box.** Every series that
`paper/mlsys2027/figs/paperdata.py` reads has a cloud twin. Each twin's
environment and parameters were copied from the store's own `.meta.json`.

| paper series | cloud arm (jobs.py `ARMS`) |
|---|---|
| RULER dense, dkv_high, dkv_mid, snapkv, sl/h2o_chunk, dkv_qk, kivi4/2_tiled (4k–64k) | `qshort.*` |
| dkv_stream, dkv_stream_fixed, sl_eq, snapkv_q, dkv_noquery, dkv_codes4(_all), kivi*, dkv_qk (131k) | `q131.*` (`dkv_fixed`, `dkv_codes4_all`, `dkv_attendall` included) |
| dkv_fstream, dkv_fstream_fixed, dkv_allblocks (32k) | `q32x.*` |
| dense_prefix / dkv_mid_prefix (the prefix check) | unnecessary: every cloud RULER job runs with the prefix |
| granite dense ≤16k; beyond-limit 24k–32k arms; 114k | `gshort`, `g24`, `g32`, `g131` |
| LongBench, all 18 rows (incl. streamlow, hybrid, post-hoc H2O/StreamingLLM, KIVI-2, chunked KIVI-4) | `lb.*` |
| ladders, all 19 series (dense_shipped=`dense_plain`, `dense_offload`, `dkv_fixed`=TILED, `dkv_streamlow`=untiled recipe, `dkv_mid`=exact_old recipe) | `lad.*` |
| systems, all 12 arms + offload | `sys.*` |
| allocator traces (`granite_peak_snapshot*`) | `diag.snapshot.*` |
| concurrency (16k) | `conc.*.16k.*` |

Two cannot be literal re-runs. `dkv_exact_old` and `dkv_stream_untiled`
measured code from before history tiling, and their recipes now run on the
tiled code. The prefix check was a workaround for the missing prefix, which
the cloud harness no longer has.

**DKV on an 80 GB card.** The elastic window keeps history exact while
allocated memory + `DKV_STREAM_ELASTIC_RESERVE_GB` stays under 94% of the
card. On 80 GB that would keep almost everything exact. The quality arms
(`dkv_stream`, `dkv_noquery`, `dkv_codes4`, `dkv_hybrid`, …) therefore set the
reserve so that the line falls where it was on the 12.878 GB card (8.605 GB
allocated). That reproduces the paper's method exactly, with no code change.
`dkv_native` leaves the card's own line in place: what DKV does when it has
memory. It is reported as its own series.

**All arms of a store run on the same items.** Every machine generates RULER
data locally. Generation is deterministic: a fresh generation on the 12.9 GB box
was byte-identical to the paper's items. Every file is checked against
`ruler_data_sha256.json`, and a mismatch fails the job. For sets the paper never
generated (granite 131k, 262k, Qwen3.5-9B), `ctl.py status` flags any machines
whose files disagree.

## 0. Before the first machine (laptop, once)

1. **GitHub token.** github.com → Settings → Developer settings →
   Fine-grained tokens → *Generate*. Repository access: only
   `Omc12/Differential-KV`. Permissions: **Contents: Read and write**.
   Expiry: 30 days. Give the token to the machines only as an environment
   variable. bootstrap.sh reads it through a credential helper and never
   writes it to disk.
2. **Commit and push the code to run.** Workers run the commit that
   `control.json` names, not whatever is on `main`.
3. **Create the queue:**
   ```bash
   python benchmarks/cloud/ctl.py init
   ```
   This pushes `cloud-results` with `code_ref` = your current HEAD.
4. **What to ask Yotta for:** ~16 GPUs of **80 GB (H100/A100-80/H200), one
   model if possible**, from 25 Oct for ~2 days. Each machine needs Ubuntu
   22.04/24.04 with an NVIDIA driver ≥ 560, root or sudo, outbound HTTPS to
   github.com, huggingface.co, pypi.org and download.pytorch.org, ~150 GB
   disk at `/workspace` (or set `DKV_ROOT`), and SSH or a web terminal. No
   inbound ports and no shared filesystem are needed. Spot/preemptible is
   fine: a preempted job resumes elsewhere.
5. **GPU size.** Use one GPU model for everything if you can. Tracks `sys`, `conc`, `lad`, `g131` and `q262` need ≥ 70 GB.
   `q131` and `g32` need ≥ 40 GB. LongBench needs ≥ 24 GB. On mixed fleets, pin a
   track to one GPU model: `ctl.py require-gpu sys H100`.

## 1. Each machine

Needs Ubuntu-like Linux, an NVIDIA driver, outbound HTTPS, and ~150 GB disk
under `/workspace`.

```bash
export GH_TOKEN=github_pat_...
export DKV_MACHINE=yotta-01
curl -fsSL https://raw.githubusercontent.com/Omc12/Differential-KV/main/benchmarks/cloud/bootstrap.sh | bash
```

It installs git/gcc, clones the code at `code_ref` and creates one worktree per GPU. It then builds a
Python 3.13 venv with torch 2.11 for the driver's CUDA line and the pinned
`requirements.txt`, downloads the three models and LongBench, and generates
RULER data in the background. Last, it starts `run_worker.sh <gpu>` for every GPU
(restarted on crash; exit 0 = finished). Running it again is safe.

Each worker first runs the **fidelity preflight**: `colab/logit_fidelity.py
--arms dense_true baseline` per model. dense_true's self-check must read KL ≈ 0,
and DKV must match dense's top-1 on every prompt with KL < 0.05. A model that
fails is never run on that GPU, and `ctl.py status` shows it. Smoke jobs then go
before anything else.

`DKV_ROOT` must be the same on every machine (default `/workspace/dkv-cloud`).
RULER stores record their data path, and a resumed job refuses a store whose
path differs.

Logs: `tail -f /workspace/dkv-cloud/logs/worker_g*.log`. The per-job logs are
in `w<g>/.cloud_logs/`, and their tails are on the branch too.

## 2. Watching and steering (laptop)

```bash
python benchmarks/cloud/ctl.py status        # per track: ok/running/stale/pending/failed/dead/blocked; machines; ETA
python benchmarks/cloud/ctl.py plan --gpus 8 16
python benchmarks/cloud/ctl.py log q131.dense.vt
```

| command | effect (within ~10 min) |
|---|---|
| `pause` / `resume` | workers stop claiming; running jobs continue |
| `stop` | each worker exits after its current job |
| `code-ref <sha>` | each worker finishes its job, checks out `<sha>`, restarts, redoes preflight |
| `skip-track q262` / `unskip-track` / `only-tracks q131 sys` | change what is claimed (smoke always allowed) |
| `retry <id or regex>` | a dead/failed job back to pending; the next attempt resumes from its pushed rows |
| `release <id>` | free a claim now (a machine you killed by hand) |
| `kill <id>` | stop a running job at its next heartbeat |
| `reset <id>` | delete everything pushed for the job (start over clean) |
| `timeout-x 2` | double every job's time limit (4×estimate + 3 h by default) |

**Fault model.** A worker pushes the job's output files every 10 minutes.
After a crash, preemption or network loss, the claim goes stale after 40
minutes. Another worker then takes the job, restores the pushed rows, and the
harness resumes, so at most ~10 minutes are lost. SIGTERM (spot preemption)
releases the claim at once. A job that fails twice becomes `dead` and its
dependents stay blocked until you `retry` it. A worker whose claim was taken
over stops its copy.

**Changing code mid-campaign.** DKV stores carry a fingerprint of the
decode-critical sources. A `code-ref` that changes them makes resumed DKV
shards refuse their old rows: the harness exits on the config mismatch and
the job fails. Do this on purpose with `reset` on the affected jobs, never by
accident.

## 3. Results into the paper

```bash
python benchmarks/cloud/ctl.py merge          # finished jobs only; --partial to include running ones
```

Shards (RULER per task, LongBench per dataset) are joined into one store per
stem: `paper/results/<ruler|longbench|systems|ladder|concurrency>/Y1_<stem>.jsonl`.
Configs must agree except along the shard axis, and a mismatch is fatal. Then
point `paper/mlsys2027/figs/paperdata.py` at the `Y1_*` stems (new series;
do not mix them into the 12.9 GB series), rebuild, and run `verify_paper.py`.
Report cloud numbers as their own hardware row: datacenter GPU, Linux,
Inductor fused.

## 4. Schedule to the deadline

The deadline is **30 Oct 2026, 20:00 UTC** (check mlsys.org). Work back from it:
results merged by **27 Oct** leave three days to write. On the default
estimates (`ctl.py plan`), 80 GB GPUs finish everything in:

| GPUs | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| hours | 91 | 46 | 31 | 24 |

So about **16 GPUs started by 25 Oct** fit. With fewer, the priorities decide what lands:
P1 (`q131`, `sys`, `conc`, 262 GPU-h) finishes before P2 starts on any machine
that can take P1. If `ctl.py status` shows an ETA past 27 Oct, `skip-track q262
q9b` first, then `lad`, then `lb`. The estimates assume a datacenter GPU is
2.5× the 12.9 GB card. After the first few hours `status` rescales them from
measured times, so re-read the ETA then.

## 5. Known limits

- Timing jobs (`sys`) share a host with other workers on multi-GPU machines.
  GPU-bound numbers are unaffected, host-side overheads slightly. For the
  paper's table, `require-gpu sys <model>` and run them on one machine with
  `DKV_GPUS="0"`, or accept it and say so.
- Smoke tests run on this repo's 12.9 GB card exercise every config on
  Windows. The cloud smoke track re-runs them on the real hardware first, so
  Linux-only failures surface before the long jobs.
- The repository is public, and so is `cloud-results`. Results are fine to
  publish. The MLSys draft is not, so keep it out of the branch.
