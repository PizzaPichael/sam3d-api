#!/bin/bash
# Resume helper: reactivate the sam3d-objects env after a pod restart to CONTINUE work
# (repair or manual installs) — without copying checkpoints or starting the API.
#
# MUST be sourced so the activated env and exports persist in your shell:
#     source resume.sh
#
# For the normal daily start of the finished env (checkpoint copy + uvicorn), use
# install_conda_start_env_host_api.sh instead.

ENV_PATH=/workspace/envs/sam3d-objects

# --- 1. Ensure the env still exists on the volume ---
if [ ! -x "$ENV_PATH/bin/python" ]; then
    echo "FATAL: env not found at $ENV_PATH/bin/python"
    echo "  Is the /workspace volume mounted?  ->  df -h /workspace"
    echo "  If the volume is empty, the env must be rebuilt:  source setup.sh"
    return 1 2>/dev/null || exit 1
fi

# --- 2. Reinstall miniconda if missing (it lives on the container disk, not the volume) ---
# Check the binary, not the directory: an empty /root/miniconda3 would pass a -d test.
if [ ! -x "/root/miniconda3/bin/conda" ]; then
    echo "miniconda missing on container disk — reinstalling..."
    curl -fsSL -o /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/Miniconda3.sh -b -p /root/miniconda3
    rm /tmp/Miniconda3.sh
fi

# --- 3. Activate the env ---
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs
conda activate "$ENV_PATH"

# Guard: 'conda activate' sets CONDA_PREFIX and the prompt even when the env has no
# bin/python — python then falls through to the system interpreter and any pip install
# lands in the container instead of the volume. Abort hard.
if [ "$(command -v python)" != "$ENV_PATH/bin/python" ]; then
    echo "FATAL: conda env is not active."
    echo "  expected: $ENV_PATH/bin/python"
    echo "  actual:   $(command -v python)"
    echo "  See runpod/docs/setup-fixes.md point 12."
    return 1 2>/dev/null || exit 1
fi

# --- 4. Restore the env vars the setup/repair relies on ---
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
export CONDA_PKGS_DIRS=/root/conda-pkgs
mkdir -p /workspace/tmp /workspace/pip-cache /root/conda-pkgs
# cu121 kept so pip can resolve sam3d-objects' pinned torchaudio==2.5.1+cu121 during a repair
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu121"

# System CUDA (12.8 with nvcc) for any native builds — conda's nvcc is too old for sm_120
CUDA_HOME_CANDIDATE=$(find /usr/local -maxdepth 1 -name "cuda-1*" -type d | sort -V | tail -1)
if [ -n "$CUDA_HOME_CANDIDATE" ] && [ -f "$CUDA_HOME_CANDIDATE/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA_HOME_CANDIDATE"
    export PATH=$CUDA_HOME/bin:$PATH
fi

echo "Env active: $(python --version) at $(command -v python)"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null \
    || echo "  (torch not importable — check the env)"
echo "Ready. CUDA_HOME=${CUDA_HOME:-<none>}  |  continue with the repair steps."
