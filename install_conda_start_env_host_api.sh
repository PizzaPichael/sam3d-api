#!/bin/bash
set -e

ENV_PATH=/workspace/envs/sam3d-objects

# Miniconda neu installieren falls nicht vorhanden.
# Auf das Binary pruefen, nicht auf das Verzeichnis: ein leeres /root/miniconda3
# (z.B. nach abgebrochener Installation) wuerde einen -d Test bestehen.
if [ ! -x "/root/miniconda3/bin/conda" ]; then
    curl -fsSL -o /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/Miniconda3.sh -b -p /root/miniconda3
    rm /tmp/Miniconda3.sh
fi

source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs
conda activate "$ENV_PATH"

# Guard: 'conda activate' setzt CONDA_PREFIX auch dann, wenn das Env nur ein leeres
# Verzeichnis ist — 'python' faellt dann still auf /usr/local/bin/python (System 3.12)
# durch und die API laeuft im falschen Interpreter. Hier hart abbrechen.
if [ "$(command -v python)" != "$ENV_PATH/bin/python" ]; then
    echo "FATAL: Conda-Env nicht aktiv."
    echo "  erwartet: $ENV_PATH/bin/python"
    echo "  ist:      $(command -v python)"
    echo "  -> Env neu anlegen, siehe runpod/docs/setup-fixes.md Punkt 12"
    exit 1
fi

# sam3d_objects/__init__.py sonst ImportError (init.py fehlt in dieser Distribution, gewollt)
export LIDRA_SKIP_INIT=true
# xformers 0.0.30 verlangt flash-attn 2.7.1-2.7.4, installiert ist 2.8.3 (vorgebautes Wheel fuer sm_120/Blackwell)
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

# System CUDA (12.8 mit nvcc) fuer etwaige Laufzeit-JIT-Builds (z.B. nvdiffrast) — conda's nvcc ist zu alt fuer sm_120
CUDA_HOME_CANDIDATE=$(find /usr/local -maxdepth 1 -name "cuda-1*" -type d | sort -V | tail -1)
if [ -n "$CUDA_HOME_CANDIDATE" ] && [ -f "$CUDA_HOME_CANDIDATE/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA_HOME_CANDIDATE"
    export PATH=$CUDA_HOME/bin:$PATH
fi

# Checkpoints vom Network-Volume auf lokale Container-Disk kopieren (einmalig pro Pod).
# Der Worker laedt sie dann von lokaler NVMe statt vom langsamen /workspace-Volume.
LOCAL_CKPT=/root/sam3d-checkpoints
if [ ! -f "$LOCAL_CKPT/hf/pipeline.yaml" ]; then
    echo "Copying checkpoints to local disk ($LOCAL_CKPT)..."
    mkdir -p "$LOCAL_CKPT"
    cp -r /workspace/sam3d-api/sam-3d-objects/checkpoints/. "$LOCAL_CKPT/"
    echo "Checkpoint copy done."
fi
export SAM3D_CHECKPOINT_DIR="$LOCAL_CKPT"

cd /workspace/sam3d-api

# api.py's module-level imports (torch, kaolin, pytorch3d, nvdiffrast, xformers,
# sam3d_objects) read their .so/.py files off the /workspace network volume (MooseFS/FUSE)
# and print nothing while doing so — this can silently take 10-20+ min with the process
# sitting at 0% CPU (blocked on FUSE I/O, not hung). Background monitor below reports
# progress via RSS growth and polls /health, so this isn't a blind wait. It exits on its
# own once the API responds, or is killed via the trap when uvicorn exits/is interrupted.
(
    start=$(date +%s)
    while true; do
        sleep 20
        pid=$(pgrep -f "uvicorn api:app" | head -1)
        [ -z "$pid" ] && break
        elapsed=$(( $(date +%s) - start ))
        if curl -sf -o /dev/null http://localhost:8000/health; then
            echo "[startup-monitor] API responding after ${elapsed}s."
            break
        fi
        rss_kb=$(awk '/VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null)
        echo "[startup-monitor] still loading... ${elapsed}s elapsed, uvicorn RSS=${rss_kb:-?} kB (growing = alive, stuck for minutes = actually stalled)"
    done
) &
monitor_pid=$!
trap 'kill $monitor_pid 2>/dev/null' EXIT

uvicorn api:app --host 0.0.0.0 --port 8000
