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

# Route all installs to network storage
export CONDA_ENVS_PATH=/workspace/envs
export CONDA_PKGS_DIRS=/workspace/conda-pkgs
export PIP_CACHE_DIR=/workspace/pip-cache
export TMPDIR=/workspace/tmp
mkdir -p /workspace/conda-pkgs /workspace/pip-cache /workspace/tmp

echo "--- 3. Setting up Conda Environment ---"
ENV_PATH="/workspace/envs/sam3d-objects"
if [ -d "$ENV_PATH" ]; then
    echo "Environment '$ENV_PATH' exists. Updating..."
    conda env update -p "$ENV_PATH" -f environments/default.yml --prune
else
    conda env create -p "$ENV_PATH" -f environments/default.yml
fi

conda activate "$ENV_PATH"

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
pip install -e '.[dev]'
pip install -e '.[p3d]'

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
# Must use system CUDA 12.8 nvcc — conda env nvcc is 12.1 and doesn't support sm_120
TORCH_CUDA_ARCH_LIST="12.0" pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
TORCH_CUDA_ARCH_LIST="12.0" pip install "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" --force-reinstall
# Re-pin torch before nvdiffrast: pytorch3d/gsplat may have pulled a newer torch,
# and nvdiffrast must be compiled against exactly 2.7.0 (ABI-sensitive)
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
TORCH_CUDA_ARCH_LIST="12.0" pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation --force-reinstall --no-deps

echo "--- 10. Absolute final version pins (overrides anything Step 9 may have pulled) ---"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12==0.6.3 --force-reinstall --no-deps

echo "--- Setup Complete! ---"
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "SUCCESS: You are now active in the '$ENV_NAME' environment."
else
    echo "To activate the environment now, run: conda activate $ENV_NAME"
fi