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

## First-time setup (fresh pod / empty volume)

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

---

## Daily start (after a pod stop/restart)

```bash
cd /workspace/sam3d-api
bash install_conda_start_env_host_api.sh
```

This script:
1. Reinstalls miniconda if missing (it lives on the container disk, not the volume — gone after "Terminate", kept after "Stop")
2. Mirrors the conda env itself to local container disk and bind-mounts it over the original `/workspace/envs/sam3d-objects` path (once per pod) — see below
3. Activates the `sam3d-objects` conda env
4. Exports the runtime env vars the env needs (see [Known runtime quirks](#known-runtime-quirks) below)
5. Copies checkpoints from the network volume to local container disk (`/root/sam3d-checkpoints`) once per pod — the worker then loads from fast local NVMe instead of the slower network volume
6. Starts the API via `$PYTHON_BIN -m uvicorn api:app --host 0.0.0.0 --port 8000` in the foreground (see below for what `$PYTHON_BIN` is), with a background monitor that logs progress (memory growth + `/health` polling) every 20s until the API responds — the import chain below can take 10-20+ min on first read, this makes that visible instead of a silent wait

**Why the env gets mirrored:** `torch`/`kaolin`/`pytorch3d`/`nvdiffrast`/`flash-attn`/`xformers`/`sam3d_objects` are read off `/workspace` — a network volume (MooseFS/FUSE) — on every single import, every pod start. That's what makes the startup imports take 10-20+ min. The script copies the whole env to local NVMe once (`/root/sam3d-env-local`).

**Two ways the mirror actually gets used, tried in this order:**
1. **`mount --bind`** the local mirror over the original `/workspace/envs/sam3d-objects` path — conda metadata and shebangs (which hardcode that path) keep working completely unchanged, only the storage backing it switches from network to local. Needs `CAP_SYS_ADMIN`, which RunPod containers have refused so far in practice (same class of restriction that blocks `py-spy`'s ptrace).
2. **If the mount is refused:** fall back to invoking the mirror's own python binary directly (`$LOCAL_ENV_MIRROR/bin/python -m uvicorn ...`) instead of going through `/workspace/envs/sam3d-objects/bin/python`. Confirmed working in practice — the interpreter resolves its own site-packages fine without needing the original path. This is what `$PYTHON_BIN` in step 6 refers to. Since `api.py` spawns the 3D worker via `sys.executable`, the worker subprocess automatically inherits whichever interpreter actually started `uvicorn` — so the speedup covers the worker's (heavier) import chain too, not just the main API process.

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

Both the checkpoint copy and the env mirror (step 2 above) print a live percent bar while they run:
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

# '--no-deps' above skips omegaconf too — sam-3d-objects/notebook/inference.py imports it
# directly, so the worker crashes on the first job otherwise (ModuleNotFoundError, not caught
# until you actually try a /generate-3d call)
pip install omegaconf
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
| `ModuleNotFoundError: No module named 'appdirs'` while building `nvidia-pyindex` | its `setup.py` needs `appdirs` even under `--no-build-isolation` | `pip install appdirs` first |
| `numpy` silently downgraded/upgraded after installing `moge`/`utils3d` | those packages were installed without `--no-deps`, pulling their own numpy pin | re-pin per the repair snippet above |
| `[API] 3D worker exited` right after startup, worker log shows `ModuleNotFoundError: No module named 'omegaconf'` | `sam-3d-objects/notebook/inference.py` imports `omegaconf` directly; re-registering `sam3d_objects` via `-e '.' --no-deps` (repair path) skips it | `pip install omegaconf` — `setup.sh` now installs it explicitly too, so a fresh setup shouldn't hit this |

Also worth knowing: **`miniconda` (`/root/miniconda3`) lives on the container disk, not the network volume.** It survives a pod **Stop**, but not a **Terminate** — a Terminate needs `resume.sh`/`install_conda_start_env_host_api.sh` to reinstall it (automatic, no data loss — the conda env itself is on `/workspace` and survives).

---

## Backup

Once the env is verified working, back it up so a future repair session doesn't need to repeat any of the above:

```bash
tar czvf /workspace/sam3d-backup-$(date +%d_%m).tar.gz \
    -C / workspace/envs/sam3d-objects workspace/sam3d-api
```

**What's in the archive:**

- `workspace/envs/sam3d-objects` — the complete conda env: Python interpreter + every installed package (torch, kaolin, pytorch3d, nvdiffrast, flash-attn, xformers, sam3d_objects and all transitive deps) exactly as verified working
- `workspace/sam3d-api` — this repo as checked out on the pod, including `sam-3d-objects/checkpoints/hf` (the ~13.3 GB model checkpoints)

**Not included:** `/root/miniconda3` (the conda installer itself — container disk, cheap to reinstall via `resume.sh`) and `/workspace/pip-cache` (throwaway download cache).

**What it's for:** restoring a working env in minutes instead of re-running `setup.sh` (which means redoing every version pin/workaround in this README) — extract the two paths back under `/workspace` on a fresh pod of the **same GPU architecture** and skip straight to [Daily start](#daily-start-after-a-pod-stoprestart).

Then copy it **off the volume** (RunPod egress is free) — a backup that lives only on the same volume doesn't protect against a volume-level problem.

⚠️ The backup is **architecture-bound**: the native extensions (kaolin, pytorch3d, nvdiffrast, flash-attn, xformers) are compiled for the specific GPU's compute capability. Restorable only on the same architecture (e.g. sm_120/Blackwell) — on a different GPU, rebuild via `setup.sh` instead.

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
