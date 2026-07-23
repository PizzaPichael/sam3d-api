#!/bin/bash

# 1. Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "#######################################################################"
    echo "WARNING: You are running this script as a subshell (./setup.sh)."
    echo "To remain inside the conda environment after the script finishes,"
    echo "please run: source setup.sh"
    echo "#######################################################################"
    sleep 2
fi

# Exit on error (disabled: sam3d-objects has pinned cu121 deps that cause non-fatal resolution warnings)
# set -e

echo "--- 1. Handling Repository ---"
REPO_DIR="sam-3d-objects"
if [ -d "$REPO_DIR" ]; then
    echo "Directory '$REPO_DIR' exists. Updating..."
    cd "$REPO_DIR"
    git pull
else
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    cd "$REPO_DIR"
fi

echo "--- 2. Checking Miniconda ---"
CONDA_ROOT="$HOME/miniconda3"
if [ -d "$CONDA_ROOT" ]; then
    echo "Miniconda already installed at $CONDA_ROOT."
else
    echo "Installing Miniconda..."
    curl -fsSL -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3.sh -b -p "$CONDA_ROOT"
    rm Miniconda3.sh
fi

# Initialize conda for the current shell session
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda init bash

# Route installs to network storage. The conda package cache is deliberately NOT on
# /workspace: that volume is MooseFS, and extracting tens of thousands of small files
# there produced CondaVerificationError ("package appears to be corrupted"). The cache is
# throwaway — only the env itself needs to persist.
export CONDA_ENVS_PATH=/workspace/envs
export CONDA_PKGS_DIRS=/root/conda-pkgs
export PIP_CACHE_DIR=/workspace/pip-cache
export TMPDIR=/workspace/tmp
mkdir -p /root/conda-pkgs /workspace/pip-cache /workspace/tmp

echo "--- 3. Setting up Conda Environment ---"
ENV_PATH="/workspace/envs/sam3d-objects"
# environments/default.yml is deliberately NOT used. It is a full conda export pinning the
# entire CUDA 12.1 toolkit plus a Qt/X11/audio stack (the cuda-nvvp and cuda-nsight
# profiler GUIs depend on qt-main). setup.sh overrides all of it anyway: CUDA_HOME points
# at system CUDA 12.8 because conda's nvcc 12.1 has no sm_120 support, and the CUDA
# runtime libs arrive as pip nvidia-* wheels with torch. On a fresh volume those unused
# packages only produced ClobberError (seven cuda-* tools sharing 'LICENSE') and
# CondaVerificationError (qt-main's translation files over MooseFS).
# What we need from the env is python 3.11 and a C/C++ toolchain for the step 9 builds.
#
# gcc is pinned to 12.4 to match what default.yml specified — nvcc 12.8 also accepts
# gcc 14, but the native builds (pytorch3d, gsplat, flash-attn) are only tested against 12.
#
# The test below checks for bin/python, not the directory: an empty $ENV_PATH left behind
# by an aborted create passes -d, and conda does not bootstrap python into a non-env.
# The result is a directory that activates cleanly but has no interpreter.
if [ -x "$ENV_PATH/bin/python" ]; then
    echo "Environment '$ENV_PATH' exists. Skipping create."
else
    [ -e "$ENV_PATH" ] && { echo "Removing incomplete env at '$ENV_PATH'..."; rm -rf "$ENV_PATH"; }
    conda create -p "$ENV_PATH" -c conda-forge -y \
        python=3.11 pip setuptools wheel \
        gcc_linux-64=12.4 gxx_linux-64=12.4
fi

conda activate "$ENV_PATH"

# Guard: 'conda activate' sets CONDA_PREFIX and the prompt even when the env has no
# bin/python — every python/pip call then falls through to /usr/local/bin/python (system
# 3.12) and the entire setup installs into the container instead of the volume.
# set -e is off above, so this must abort explicitly.
if [ "$(command -v python)" != "$ENV_PATH/bin/python" ]; then
    echo "#######################################################################"
    echo "FATAL: conda env is not active."
    echo "  expected: $ENV_PATH/bin/python"
    echo "  actual:   $(command -v python)"
    echo "  See runpod/docs/setup-fixes.md point 12."
    echo "#######################################################################"
    return 1 2>/dev/null || exit 1
fi
echo "Env active: $(python --version) at $(command -v python)"

# Use system CUDA for all native builds — conda env nvcc is too old and doesn't support sm_120
# Pick the highest available CUDA version under /usr/local
CUDA_HOME_CANDIDATE=$(find /usr/local -maxdepth 1 -name "cuda-1*" -type d | sort -V | tail -1)
if [ -n "$CUDA_HOME_CANDIDATE" ] && [ -f "$CUDA_HOME_CANDIDATE/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA_HOME_CANDIDATE"
    export PATH=$CUDA_HOME/bin:$PATH
    echo "Using system CUDA: $CUDA_HOME ($(nvcc --version | grep release))"
else
    echo "WARNING: No system nvcc found — native builds may fail for sm_120"
fi

echo "--- 4. Installing PyTorch 2.7.0 (Blackwell/cu128) ---"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# cu121 included so pip can resolve sam3d-objects' pinned torchaudio==2.5.1+cu121 — overridden in step 5b
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu121"

# auto-gptq is an sdist whose setup.py imports torch at build time. pip's build isolation
# hides the torch installed above, so '.[dev]' below aborts with "No module named 'torch'"
# and sam3d_objects never gets installed. Pre-install it here so pip sees it as satisfied.
# BUILD_CUDA_EXT=0 skips the CUDA kernel build — GPTQ quantization is not used in the
# mesh pipeline, only the import needs to resolve.
BUILD_CUDA_EXT=0 pip install auto-gptq==0.7.1 --no-build-isolation

pip install -e '.[dev]'
pip install -e '.[p3d]'

# '.[dev]' silently continues on failure (set -e is off) — verify the package actually landed
python -c "import sam3d_objects" 2>/dev/null || {
    echo "WARNING: sam3d_objects not importable after '.[dev]' — retrying without build isolation"
    pip install -e '.[dev]' --no-build-isolation
}

echo "--- 5. Installing Inference deps (manual, bypassing sam3d cu121 pins) ---"
# Install requirements.inference.txt manually to avoid sam3d-objects resolving torchaudio==2.5.1+cu121
pip install seaborn==0.13.2 gradio==5.49.0
pip install "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html

echo "--- 5b. cu128 overrides ---"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
# spconv has no cu128 package — cu121 build runs on cu128 at runtime
pip install spconv-cu121==2.3.8 --extra-index-url https://download.pytorch.org/whl/cu121
pip install xformers --index-url https://download.pytorch.org/whl/cu128 --no-deps

if [ -f "./patching/hydra" ]; then
    chmod +x ./patching/hydra
    ./patching/hydra
fi

echo "--- 6. Downloading Model Checkpoints ---"
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
if [ ! -d "checkpoints/${TAG}" ]; then
    mkdir -p checkpoints
    huggingface-cli download \
      --repo-type model \
      --local-dir checkpoints/${TAG}-download \
      --max-workers 1 \
      facebook/sam-3d-objects

    mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
    rm -rf checkpoints/${TAG}-download
else
    echo "Checkpoints already present. Skipping download."
fi

echo "--- 7. Final Requirements (Root) ---"
cd ..
if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt from $(pwd)..."
    pip install -r requirements.txt
    pip install hf_transfer
    pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
else
    echo "No requirements.txt found in $(pwd)."
fi

echo "--- 8. Final cu128 pin (must run last to override any cu121 reinstalls) ---"
pip install xformers --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html --force-reinstall --no-deps

echo "--- 9. Rebuilding native extensions against final torch 2.7.0+cu128 ---"
# These must be built AFTER the torch pin — they have CUDA kernels and are ABI-sensitive
# Must use system CUDA nvcc — conda env nvcc is too old and doesn't support newer architectures

# Detect GPU compute capability so builds work on any GPU (Blackwell sm_120, A100 sm_80, etc.)
GPU_ARCH=$(python -c "import torch; cap=torch.cuda.get_device_capability(); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "8.0")
echo "Detected GPU compute capability: sm_${GPU_ARCH/./} (TORCH_CUDA_ARCH_LIST=$GPU_ARCH)"

TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" --force-reinstall
# Re-pin torch before remaining builds: pytorch3d/gsplat may have pulled a newer torch,
# and ABI-sensitive packages must be compiled against exactly 2.7.0
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation --force-reinstall --no-deps
# diff_gaussian_rasterization and flash_attn are ABI-sensitive — must be rebuilt here
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization" --no-build-isolation --force-reinstall --no-deps
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install flash-attn --no-build-isolation --force-reinstall --no-deps

echo "--- 10. Absolute final version pins (overrides anything Step 9 may have pulled) ---"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12==0.6.3 --force-reinstall --no-deps

echo "--- Setup Complete! ---"
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "SUCCESS: You are now active in the '$ENV_PATH' environment."
else
    echo "To activate the environment now, run: conda activate $ENV_PATH"
fi