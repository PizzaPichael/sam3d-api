#!/bin/bash
set -e

ENV_PATH=/workspace/envs/sam3d-objects

# Copies src -> dst in the background while printing a percent bar (0%....10%....20%...)
# by polling the destination's size against the source's total size. Plain 'cp' gives zero
# output — on the slow /workspace FUSE volume that's indistinguishable from a hang (see
# README "Diagnosing a stuck copy"). Used for both the checkpoint copy and the env mirror
# below, since both read from the same slow network volume.
copy_with_progress() {
    local src="$1" dst="$2"
    local total_bytes current_bytes percent last_percent step=2 pid status start_ts elapsed

    start_ts=$(date +%s)
    mkdir -p "$dst"
    total_bytes=$(du -sb "$src" 2>/dev/null | cut -f1)
    if [ -z "$total_bytes" ] || [ "$total_bytes" -eq 0 ]; then
        total_bytes=1
    fi

    cp -a "$src/." "$dst/" &
    pid=$!

    printf "0%%"
    last_percent=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        current_bytes=$(du -sb "$dst" 2>/dev/null | cut -f1)
        [ -z "$current_bytes" ] && current_bytes=0
        percent=$(( current_bytes * 100 / total_bytes ))
        [ "$percent" -gt 99 ] && percent=99   # cp still running -> never show 100 early
        while [ "$last_percent" -lt "$percent" ]; do
            last_percent=$(( last_percent + step ))
            if [ $(( last_percent % 10 )) -eq 0 ]; then
                printf "%d%%" "$last_percent"
            else
                printf "."
            fi
        done
    done
    if wait "$pid"; then
        status=0
    else
        status=$?
    fi
    while [ "$last_percent" -lt 100 ]; do
        last_percent=$(( last_percent + step ))
        if [ "$last_percent" -ge 100 ]; then
            last_percent=100
            printf "100%%"
        elif [ $(( last_percent % 10 )) -eq 0 ]; then
            printf "%d%%" "$last_percent"
        else
            printf "."
        fi
    done
    elapsed=$(( $(date +%s) - start_ts ))
    printf " (%dm%02ds)\n" "$(( elapsed / 60 ))" "$(( elapsed % 60 ))"
    return "$status"
}

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

# Mirror the conda env to local container disk and bind-mount it OVER the original
# network-volume path, so every path baked into the env (conda metadata, shebangs like
# '#!/workspace/envs/sam3d-objects/bin/python3.11') keeps working unchanged — only the
# storage backing that path switches from network (MooseFS/FUSE) to local NVMe. This is
# what makes 'import torch/kaolin/pytorch3d/nvdiffrast/xformers/sam3d_objects' take
# 10-20+ min on every pod start (see README "Known runtime quirks") — env files get read
# fresh from the network volume every time otherwise.
#
# Requires CAP_SYS_ADMIN for 'mount --bind', which RunPod containers have refused so far
# (same class of restriction that blocks py-spy's ptrace). If the mount isn't permitted,
# fall back to invoking the local mirror's python directly (confirmed working — its own
# prefix resolution doesn't depend on the bind-mount trick) instead of via the network path.
LOCAL_ENV_MIRROR=/root/sam3d-env-local
BIND_MOUNTED=0
if mountpoint -q "$ENV_PATH" 2>/dev/null; then
    BIND_MOUNTED=1
else
    if [ ! -x "$LOCAL_ENV_MIRROR/bin/python" ]; then
        echo "Mirroring conda env to local disk ($LOCAL_ENV_MIRROR) for faster imports on future starts — one-time, can take a while..."
        copy_with_progress "$ENV_PATH" "$LOCAL_ENV_MIRROR"
        echo "Env mirror done."
    fi
    if mount --bind "$LOCAL_ENV_MIRROR" "$ENV_PATH" 2>/dev/null; then
        echo "Conda env now served from local disk (bind-mounted over $ENV_PATH)."
        BIND_MOUNTED=1
    else
        echo "NOTE: bind-mount not permitted in this container — will invoke the local env mirror directly instead."
    fi
fi

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
    copy_with_progress /workspace/sam3d-api/sam-3d-objects/checkpoints "$LOCAL_CKPT"
    echo "Checkpoint copy done."
fi
export SAM3D_CHECKPOINT_DIR="$LOCAL_CKPT"

# Which python actually runs uvicorn (and, via sys.executable, the worker_3d.py subprocess
# api.py spawns): the network path if it's bind-mounted to local disk (or genuinely on
# network storage), otherwise the local mirror directly — bypassing the failed bind-mount
# still gets the speedup, since the mirror's own python resolves its prefix independently.
if [ "$BIND_MOUNTED" -eq 1 ] || [ ! -x "$LOCAL_ENV_MIRROR/bin/python" ]; then
    PYTHON_BIN="$ENV_PATH/bin/python"
else
    PYTHON_BIN="$LOCAL_ENV_MIRROR/bin/python"
    echo "Using local env mirror directly (no bind-mount) for faster imports: $PYTHON_BIN"
fi

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

"$PYTHON_BIN" -m uvicorn api:app --host 0.0.0.0 --port 8000
