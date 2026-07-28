# sam3d-api

FastAPI service that runs on a RunPod GPU pod:

1. **SAM 2** (Hugging Face `facebook/sam2.1-hiera-large`) — point-based image segmentation
2. **Sam-3d-objects** (Meta, `facebook/sam-3d-objects`) — turns an image + mask into a textured 3D mesh (GLB)

This README covers **setup and operating the pod**. For the API contract (endpoints, request/response shapes) see [API Reference](#api-reference) below.

Everything here targets a RunPod pod with a **Network Volume mounted at `/workspace`** — that's where the conda env, the repo, and the model checkpoints live so they survive a pod stop/restart. Scripts assume the repo is checked out at `/workspace/sam3d-api`.

---

## Prerequisites

- RunPod GPU pod (tested on RTX PRO 4500 Blackwell / sm_120; `setup.sh` detects the GPU's compute capability automatically, so other architectures — e.g. A40/A100 — should need little to no manual patching, likely less than sm_120 needed since prebuilt wheels for older architectures are more widely available)
- Network Volume mounted at `/workspace`
- Hugging Face account with access to the gated `facebook/sam-3d-objects` model, and a token (`hf auth login`)
- `tmux` available (`apt-get install -y tmux`) — `setup.sh` runs 15-30 min and auto-relaunches itself inside a `tmux` session so an SSH drop doesn't kill it mid-build

---

## Bringing up a fresh pod — two paths

| Situation | Path | Time |
|---|---|---|
| You have an env snapshot from a pod of the **same GPU architecture** | [Restore from snapshot](#option-a-restore-from-an-env-snapshot-fast) | ~5 min + checkpoint download |
| No snapshot, or a different GPU architecture | [Build from scratch](#option-b-build-from-scratch-with-setupsh) | 15-30 min + all the debugging in `docs/setup-fixes.md` if anything drifted |

Both assume a Network Volume mounted at `/workspace` and the repo at `/workspace/sam3d-api`.

---

## Option A: restore from an env snapshot (fast)

Skips `setup.sh` entirely. Requires a snapshot built on the **same compute capability** — see [Backup](#backup) for how one is created and where it's stored.

The env is an *editable* install: `sam3d_objects` resolves to `/workspace/sam3d-api/sam-3d-objects`, and the worker loads `notebook/inference.py` from there at runtime. Those source trees must exist at exactly those paths or the restored env is useless — so clone first, restore second.

```bash
# 1. Repos
cd /workspace
git clone <this-repo-url> sam3d-api
cd sam3d-api
git clone https://github.com/facebookresearch/sam-3d-objects.git

# 2. HF CLI in the container's system python (throwaway, just to pull the artifacts)
pip install 'huggingface-hub[cli]<1.0' hf_transfer
export HF_HOME=/workspace/hf-home HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login

# 3. Env snapshot -> /workspace/env-snapshot.tzst (~4.6 GB)
huggingface-cli download --repo-type=model --local-dir /workspace <user>/sam3d-env-sm120 env-snapshot.tzst

# 4. Checkpoints (~13 GB)
cd /workspace/sam3d-api/sam-3d-objects
huggingface-cli download --repo-type=model --local-dir checkpoints/hf-download --max-workers 4 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download
```

Step 3 puts the file at `/workspace/env-snapshot.tzst` — exactly where `install_conda_start_env_host_api.sh` looks for it. If your copy lives on your own machine instead, `runpodctl send` it from there and move it to that path.

`mv` in step 4 moves *into* the target when `checkpoints/hf` already exists, producing a nested duplicate — verify with `du -sh checkpoints/hf` (expect ~13 GB, not 25).

Then the normal start:

```bash
cd /workspace/sam3d-api
bash install_conda_start_env_host_api.sh
```

It finds the snapshot, restores the env to local disk in ~23s, and starts the API.

The conda env under `/workspace/envs/sam3d-objects` is **not** recreated by this path — only the local mirror exists. The start script detects that and runs straight from the mirror (`NOTE: kein Env auf dem Volume`). That is enough to serve requests, but `resume.sh` — which activates the volume env for repair work — needs it. To get it back without re-running `setup.sh`:

```bash
mkdir -p /workspace/envs/sam3d-objects
zstd -dc /workspace/env-snapshot.tzst | tar -C /workspace/envs/sam3d-objects -xf -
```

Slow (~150k small files onto MooseFS, tens of minutes) — only worth it if you expect to do repair work on this pod.

---

## Option B: build from scratch with `setup.sh`

```bash
cd /workspace
git clone <this-repo-url> sam3d-api
cd sam3d-api

pip install 'huggingface-hub[cli]<1.0'
hf auth login

source setup.sh
```

`setup.sh` is idempotent — safe to re-run, it skips steps that already succeeded (existing conda env, existing checkpoints). It handles, in order: conda env creation, PyTorch 2.7.0+cu128, the `sam3d-objects` package + its native-extension dependencies (pytorch3d, flash-attn, xformers, kaolin, spconv, nvdiffrast, gsplat), and the ~13.3 GB checkpoint download.

Runs inside `tmux` automatically. If it drops: `tmux attach -t setup`.

Once it's done and a `/generate-3d` call has succeeded, **create an env snapshot** — that is what turns every future pod start into Option A and cuts the env mirror from 74m42s to ~23s. See [Backup](#backup).

---

## Daily start (after a pod stop/restart)

```bash
cd /workspace/sam3d-api
bash install_conda_start_env_host_api.sh
```

This script:
1. Reinstalls miniconda if missing (it lives on the container disk, not the volume — gone after "Terminate", kept after "Stop")
2. Restores the conda env to local container disk from `/workspace/env-snapshot.tzst` and bind-mounts it over the original `/workspace/envs/sam3d-objects` path (once per pod) — see below
3. Activates the `sam3d-objects` conda env
4. Exports the runtime env vars the env needs (see [Known runtime quirks](#known-runtime-quirks) below)
5. Points `SAM3D_CHECKPOINT_DIR` at the checkpoints on the volume — they are **not** copied to local disk (measured: pipeline loads in ~81s straight off the volume; a copy costs more than it saves, see [`docs/env-snapshot-plan.md`](docs/env-snapshot-plan.md))
6. Starts the API via `$PYTHON_BIN -m uvicorn api:app --host 0.0.0.0 --port 8000` in the foreground (see below for what `$PYTHON_BIN` is), with a background monitor that logs progress (memory growth + `/health` polling) every 20s until the API responds

**Why the env gets mirrored:** `torch`/`kaolin`/`pytorch3d`/`nvdiffrast`/`flash-attn`/`xformers`/`sam3d_objects` are read off `/workspace` — a network volume (MooseFS/FUSE) — on every single import, every pod start. That's what makes the startup imports take 10-20+ min. The script puts the whole env on local NVMe once per pod (`/root/sam3d-env-local`).

**Why from a snapshot and not a plain copy:** the env has ~150k files and `cp -a` pays a FUSE round-trip per file — measured at **74m42s**. Extracting one sequential 4.6 GB archive instead takes **22.5s**. The script falls back to the file-by-file copy when no snapshot exists.

⚠️ **The snapshot must be rebuilt after every change to the env** (repair session, new package), otherwise the next pod start silently restores the old state:

```bash
printf '*.a\n./include\n./share/doc\n./share/man\n./pkgs\n' > /root/snap-exclude.txt
tar -C /root/sam3d-env-local -X /root/snap-exclude.txt -cf - . | zstd -T0 -3 -o /workspace/env-snapshot.tzst
```

Keep it on one line — a paste that breaks mid-command leaves `tar` without a source and writes a **13-byte archive** that the next start would accept as a valid snapshot.

**Two ways the mirror actually gets used, tried in this order:**
1. **`mount --bind`** the local mirror over the original `/workspace/envs/sam3d-objects` path — conda metadata and shebangs (which hardcode that path) keep working completely unchanged, only the storage backing it switches from network to local. Needs `CAP_SYS_ADMIN`, which RunPod containers have refused so far in practice (same class of restriction that blocks `py-spy`'s ptrace).
2. **If the mount is refused:** fall back to invoking the mirror's own python binary directly (`$LOCAL_ENV_MIRROR/bin/python -m uvicorn ...`) instead of going through `/workspace/envs/sam3d-objects/bin/python`. Confirmed working in practice — the interpreter resolves its own site-packages fine without needing the original path. This is what `$PYTHON_BIN` in step 6 refers to. Since `api.py` spawns the 3D worker via `sys.executable`, the worker subprocess automatically inherits whichever interpreter actually started `uvicorn` — so the speedup covers the worker's (heavier) import chain too, not just the main API process.

⚠️ **Only `bin/python` resolves its own prefix.** Every other console script in the mirror's `bin/` (`huggingface-cli`, `uvicorn`, `pip`, …) carries a hardcoded `#!/workspace/envs/sam3d-objects/bin/python` shebang, so running it silently executes off the network volume — and fails outright on a restore-only pod that has no volume env at all. Always go through the module:

```bash
/root/sam3d-env-local/bin/python -m huggingface_hub.commands.huggingface_cli whoami   # not bin/huggingface-cli
/root/sam3d-env-local/bin/python -m pip install ...                                   # not bin/pip
```

Either way, once the mirror exists the imports run off local disk — no case left where the copy goes to waste.

Run it inside `tmux` if you want to detach and keep the API running after disconnecting:
```bash
tmux new -s sam3d
bash install_conda_start_env_host_api.sh
# detach: Ctrl-b d   |   reattach: tmux attach -t sam3d
```

Verify it's up:
```bash
curl http://localhost:8000/health
```

### Progress bars

The env-mirror **fallback** copy (step 2 above, only when no snapshot exists) prints a live percent bar while it runs:
```
Mirroring conda env to local disk (/root/sam3d-env-local) for faster imports on future starts — one-time, can take a while...
0%..........10%..........20%..........30%..........40%..........50%..........60%..........70%..........80%..........90%..........100% (7m42s)
```
It's produced by polling the destination's size against the source's total size every 10s (`copy_with_progress` in the script) — both directories live on the same slow `/workspace` network volume, so plain `cp` gave zero output otherwise. The elapsed time printed at the end is kept on screen so you can see how long the copy actually took.

### Diagnosing a stuck copy or a stuck startup

If a progress bar (or the `[startup-monitor]` output once `uvicorn` starts) stops advancing for several minutes, check whether it's actually stuck or just slow — `/workspace` is a FUSE-backed network volume (MooseFS), so near-zero CPU and no visible output are *normal* while it's reading, not proof of a hang.

**Find the process** (matches multiple keywords via basic regex alternation):
```bash
pgrep -fa "cp -a|cp -r|uvicorn|worker_3d"
```

**Live CPU/mem for a specific PID:**
```bash
top -p <PID>
```

**Sleeping vs. genuinely stuck** — check kernel wait-channel and memory:
```bash
cat /proc/<PID>/status | grep -E "State|VmRSS"
cat /proc/<PID>/wchan; echo
```
`wchan: request_wait_answer` means it's blocked on a FUSE request (i.e. `/workspace` I/O) — expected while copying/importing, not a deadlock by itself.

**For a copy specifically** — exact byte count, since `du -sh` rounds and can look unchanged between checks on a multi-GB copy:
```bash
du -sb <destination-dir>
# wait ~30-60s
du -sb <destination-dir>
```
Growing → still copying, just slow. Identical twice in a row → check which file it's stuck on:
```bash
ls -l /proc/<PID>/fd
```
Run that twice a bit apart — same source file open both times points at one specific file being the bottleneck (or a genuine hang), not general slowness.

---

## Repairing a broken env (without a full `setup.sh` rebuild)

If a `pip install` mid-session pulled in an incompatible version (e.g. an unrelated package upgrading `numpy` or `torch` as a side effect), you don't need to rerun all of `setup.sh`. Reactivate the env and diagnose:

```bash
source resume.sh
```

This activates the env, restores the base env vars, and prints the torch version. From there, common fixes:

```bash
# torch/numpy got dragged to the wrong version by an unrelated pip install
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12==0.6.3 --force-reinstall --no-deps

# re-register sam3d_objects after any dependency reinstall
cd /workspace/sam3d-api/sam-3d-objects
pip install -e '.' --no-deps
python -c "import sam3d_objects; print('sam3d OK')"

# '--no-deps' above skips omegaconf and hydra too — sam-3d-objects/notebook/inference.py
# imports both directly, so the worker crashes on the first job otherwise (ModuleNotFoundError,
# not caught until you actually try a /generate-3d call)
pip install omegaconf hydra-core
```

**Verify the full env in one shot** (separate `python -c` calls each pay the full torch-import cost again on network storage — combine into one process):

```bash
python -c "
import torch; print('torch', torch.__version__, torch.cuda.is_available())
import numpy; print('numpy', numpy.__version__)
import kaolin, pytorch3d, nvdiffrast; print('native OK')
import flash_attn; print('flash-attn', flash_attn.__version__)
import xformers.ops; print('xformers OK')
import sam3d_objects; print('sam3d OK')
"
```

---

## Known runtime quirks

These are already handled by `install_conda_start_env_host_api.sh` (and `LIDRA_SKIP_INIT` is also set in-code by `api.py`/`worker_3d.py`) — listed here so manual `python -c` checks against the env don't hit surprise errors:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sam3d_objects.init'` | `init.py` isn't part of this distribution — only needed for heavyweight training paths, guarded by an env var | `export LIDRA_SKIP_INIT=true` |
| `ImportError: Requires Flash-Attention version >=2.7.1,<=2.7.4 but got 2.8.3` on `import xformers.ops` | flash-attn 2.8.3 is the only prebuilt wheel available for sm_120/Blackwell; xformers 0.0.30 only tests against 2.7.x | `export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1` — bypasses the version *check* only, the actual flash-attn bindings still load. Verified numerically compatible (see `runpod/docs/resume-schekclist-24_07.md` in the `MixedRealityInteriorArrangement` repo for the correctness test) |
| `CUDA error (.../xformers/third_party/flash-attention/hopper/...): no kernel image is available for execution on the device` — worker dies mid-`pipe.run()` on the first `/generate-3d` | DINOv2's attention dispatches into xformers' FlashAttention-3 kernels, which are built for sm_90/Hopper only | `export XFORMERS_DISABLED=1` — dinov2 falls back to its plain PyTorch attention/SwiGLU. See `docs/setup-fixes.md` point 20 |
| `ModuleNotFoundError: No module named 'appdirs'` while building `nvidia-pyindex` | its `setup.py` needs `appdirs` even under `--no-build-isolation` | `pip install appdirs` first |
| `numpy` silently downgraded/upgraded after installing `moge`/`utils3d` | those packages were installed without `--no-deps`, pulling their own numpy pin | re-pin per the repair snippet above |
| `[API] 3D worker exited` right after startup, worker log shows `ModuleNotFoundError: No module named 'omegaconf'` (or `'hydra'`) | `sam-3d-objects/notebook/inference.py` imports both `omegaconf` and `hydra` directly; re-registering `sam3d_objects` via `-e '.' --no-deps` (repair path) skips them | `pip install omegaconf hydra-core` — `setup.sh` now installs both explicitly too, so a fresh setup shouldn't hit this |

Also worth knowing: **`miniconda` (`/root/miniconda3`) lives on the container disk, not the network volume.** It survives a pod **Stop**, but not a **Terminate** — a Terminate needs `resume.sh`/`install_conda_start_env_host_api.sh` to reinstall it (automatic, no data loss — the conda env itself is on `/workspace` and survives).

---

## Backup

Only one artifact is genuinely irreplaceable:

| Artifact | How it comes back |
|---|---|
| this repo | `git clone` |
| `sam-3d-objects` source tree | `git clone` (done by `setup.sh` step 1) |
| checkpoints, ~13 GB | `huggingface-cli download` |
| **the conda env** | `setup.sh` — 25 min *plus* re-deriving every pin and workaround in `docs/setup-fixes.md` |

So the thing to back up is `/workspace/env-snapshot.tzst` (~4.6 GB) — the same snapshot the daily start already restores from.

**Create it** once the env is verified working (a `/generate-3d` call returned `completed`), from the local mirror rather than the volume — reading 150k files off MooseFS is what made the old copy take 74 minutes:

```bash
printf '*.a\n./include\n./share/doc\n./share/man\n./pkgs\n' > /root/snap-exclude.txt
tar -C /root/sam3d-env-local -X /root/snap-exclude.txt -cf - . | zstd -T0 -3 -o /workspace/env-snapshot.tzst
```

Takes ~17s (12.7 GiB → 4.6 GiB). Verify before relying on it:

```bash
zstd -dc /workspace/env-snapshot.tzst | tar -tf - | grep -c "site-packages/torch/"   # expect ~13500
```

It lives on the network volume, which survives Stop and Terminate but not a lost or replaced volume, so put a copy elsewhere.

**Private HF repo** (fast to push and pull). Needs a **write** token — the one used for the checkpoint download is read-only and `create_repo` returns `403 Forbidden`. `<user>` is the Hugging Face account name, not the shell user (`root` is not a valid namespace):

```bash
export HF_HOME=/workspace/hf-home HF_HUB_ENABLE_HF_TRANSFER=1
/root/sam3d-env-local/bin/python -m huggingface_hub.commands.huggingface_cli whoami   # -> <user>
/root/sam3d-env-local/bin/python -m huggingface_hub.commands.huggingface_cli upload --private --repo-type=model <user>/sam3d-env-sm120 /workspace/env-snapshot.tzst env-snapshot.tzst
```

**Plus a copy on your own machine** (RunPod egress is free) — protects against losing the HF account too:

```bash
runpodctl send /workspace/env-snapshot.tzst
```
then `runpodctl receive <code>` locally.

⚠️ **Architecture-bound.** The snapshot contains extensions compiled for one compute capability (`TORCH_CUDA_ARCH_LIST` comes from the detected GPU, see `setup.sh` step 4). sm_120 code does not run on Ampere or Ada — PTX JIT only works forward, never backward. Name the backup after the arch and keep one per GPU type. On a different architecture, run `setup.sh` there and make a second snapshot; the repo itself needs no changes, arch detection is automatic.

---

## Calling the API from your own machine

The API listens on `0.0.0.0:8000` inside the pod. Two ways to reach it from outside:

1. **RunPod HTTP proxy** — add `8000` to the pod's *Expose HTTP Ports*, then use
   `https://<POD_ID>-8000.proxy.runpod.net`. No SSH tunnel, works from anywhere.
2. **TCP port mapping** — expose 8000 as a TCP port, RunPod maps it to `<ip>:<external-port>`.

Check it's reachable:

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

⚠️ **The API has no authentication.** Anyone who knows the proxy URL can submit generation
jobs and use the GPU. Only expose it while you need it, and stop the pod when you're done.

### `client/generate3d.py`

Standard library only — no `pip install` needed. Image in, GLB out:

```bash
python client/generate3d.py --url https://<POD_ID>-8000.proxy.runpod.net \
                            --image chair.jpg --x 400 --y 300 --out chair.glb
```

`--x/--y` is the click point that SAM 2 segments around (pixel coordinates in the source
image). The script picks the highest-scoring of the returned masks, submits the 3D job,
polls until it finishes, and downloads the mesh.

Already have a mask? Skip segmentation:

```bash
python client/generate3d.py --url ... --image chair.jpg --mask chair_mask.png
```

It uses `/segment`, not `/segment-binary` — the latter returns the masked RGB image rather
than a binary mask, and dark pixels inside the object would turn into holes.

---

## API Reference

Base URL: `http://<pod-ip>:8000` (or via RunPod's proxy URL).

### `GET /health`
```json
{"status": "healthy", "model_loaded": true, "device": "cuda", "model": "facebook/sam2.1-hiera-large", "worker_ready": true}
```
`worker_ready: false` means the persistent 3D-generation worker (see below) is still loading its pipeline — segmentation (`/segment*`) works regardless, `/generate-3d` will wait for it.

### `POST /segment` — single-point segmentation
Body:
```json
{"image": "<base64 PNG/JPEG>", "x": 200, "y": 150, "multimask_output": true, "mask_threshold": 0.0}
```
Returns `masks` (array of `{mask: base64 PNG, mask_shape, score}`), `input_point`, `image_shape`.

### `POST /segment-binary` — multi-point, returns masked image
Body:
```json
{"image": "<base64 image>", "points": [{"x": 200, "y": 150}, {"x": 220, "y": 170}], "previous_mask": "<optional base64 PNG>", "mask_threshold": 0.0}
```
Returns `{"mask": "<base64 PNG>", "score": 0.95}` — the source image with everything outside the mask blacked out.

### `POST /generate-3d` — async 3D generation
Body:
```json
{"image": "<base64 image>", "mask": "<base64 binary mask PNG>", "seed": 42}
```
Returns immediately: `{"task_id": "<uuid>", "status": "queued"}`. Generation runs in a **persistent worker process** (loads the Sam-3d-objects pipeline once at API startup, not per-request — see `worker_3d.py`), serialized one job at a time (single GPU).

### `GET /generate-3d-status/{task_id}` — poll for the result
```json
{
  "task_id": "...",
  "status": "queued | processing | completed | failed",
  "progress": 0,
  "mesh_url": "/assets/mesh_<id>.glb",
  "mesh_format": "glb",
  "mesh_size_bytes": 1234567,
  "inference_seconds": 12.3
}
```
`mesh_url`/`mesh_format`/`mesh_size_bytes`/`inference_seconds` only present when `status == "completed"`; `error` present when `status == "failed"`. Download the mesh directly from `mesh_url` (served as a static file under `/assets`) — the API no longer embeds mesh bytes as base64 in the JSON response.

### `GET /assets-list`
```json
{"files": [{"name": "mesh_x.glb", "size_bytes": 123, "url": "/assets/mesh_x.glb", "created_at": "..."}], "total_files": 1, "total_size_bytes": 123}
```

---

## Development notes

- `api.py` — FastAPI app: SAM 2 segmentation endpoints, task queue for `/generate-3d`, spawns/manages the persistent worker.
- `worker_3d.py` — long-lived subprocess, loads the Sam-3d-objects pipeline once and processes jobs from stdin as line-delimited JSON. Replaces the old per-request subprocess (`generate_3d_subprocess.py`, removed) that reloaded all checkpoints on every request.
- `setup.sh` — from-scratch env bootstrap, heavily commented with the *why* behind each pin/workaround; read it before changing dependency versions.
- `resume.sh` — reactivate the env for manual repair work; does **not** start the API (that's `install_conda_start_env_host_api.sh`).

## License

MIT
