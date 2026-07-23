"""
Persistent 3D generation worker.

Loads the Sam-3d-objects pipeline ONCE, then processes jobs in a loop.
Replaces the per-request subprocess (generate_3d_subprocess.py) which
reloaded all checkpoints on every request.

Protocol (line-based JSON over stdin/stdout):
    stdin:  {"job_id": "...", "image_path": "...", "mask_path": "...", "seed": 42}
    stdout: {"job_id": "...", "status": "completed", "mesh_url": "/assets/...",
             "mesh_size_bytes": 123, "inference_seconds": 12.3}
        or: {"job_id": "...", "status": "failed", "error": "..."}

On startup, prints "READY" on stdout once the pipeline is loaded.
All diagnostic logging goes to stderr so stdout stays a clean JSON channel.
"""

import sys
import os
import json

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# ============================================================================
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME") or os.environ.get(
    "CONDA_PREFIX", ""
)
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"

# Prevent thread explosion - limit OpenMP threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import time
import uuid
from datetime import datetime

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from PIL import Image


def log(msg):
    """Diagnostics go to stderr - stdout is reserved for the JSON protocol."""
    print(f"[Worker] {msg}", file=sys.stderr, flush=True)


def emit(obj):
    """Write one JSON result line to stdout."""
    print(json.dumps(obj), flush=True)


def find_config_path():
    """Resolve pipeline.yaml: SAM3D_CHECKPOINT_DIR first, then fixed path."""
    ckpt_dir = os.environ.get("SAM3D_CHECKPOINT_DIR")
    candidates = []
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, "hf", "pipeline.yaml"))
    candidates.append("./sam-3d-objects/checkpoints/hf/pipeline.yaml")

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"pipeline.yaml not found. Tried: {candidates}. "
        "Set SAM3D_CHECKPOINT_DIR or run from the sam3d-api directory."
    )


def find_notebook_path():
    """Locate the sam-3d-objects notebook folder (contains inference.py)."""
    candidates = [
        "./sam-3d-objects/notebook",
        "../notebook",
        "/workspace/sam-3d-objects/notebook",
        "/workspace/notebook",
        os.path.expanduser("~/sam-3d-objects/notebook"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise ImportError(
        "Sam-3d-objects notebook path not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def make_synthetic_pointmap(image: np.ndarray, z: float = 1.0, f: float = None):
    """
    Create a simple pinhole-camera pointmap:
      X = (u - cx) / f * Z
      Y = (v - cy) / f * Z
      Z = constant depth

    This is non-degenerate (unlike all-zeros XY) and stays finite.
    Used to avoid intrinsics recovery failures from MoGe or dummy pointmaps.
    """
    H, W = image.shape[:2]
    if f is None:
        f = 0.9 * max(H, W)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    Z = np.full((H, W), z, dtype=np.float32)
    X = (uu - cx) / f * Z
    Y = (vv - cy) / f * Z

    pm = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return torch.from_numpy(pm)


def load_pipeline():
    """Load the Sam-3d-objects pipeline once. Returns (pipe, to_glb)."""
    notebook_path = find_notebook_path()
    sys.path.insert(0, notebook_path)
    from inference import Inference

    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import (
        to_glb,
    )

    config_path = find_config_path()
    log(f"Loading pipeline from {config_path}...")
    load_start = time.time()

    try:
        sam3d_inference = Inference(config_path, compile=False, device="cuda")
    except TypeError:
        sam3d_inference = Inference(config_path, compile=False)

    # Force all models to GPU
    moved_count = 0
    if hasattr(sam3d_inference, "_pipeline") and hasattr(
        sam3d_inference._pipeline, "models"
    ):
        for model_name, model in sam3d_inference._pipeline.models.items():
            if hasattr(model, "cuda"):
                model.cuda()
                moved_count += 1
            if hasattr(model, "eval"):
                model.eval()
    log(f"Moved {moved_count} models to GPU")

    torch.set_grad_enabled(False)

    pipe = getattr(sam3d_inference, "_pipeline", None)
    if pipe is None:
        raise RuntimeError("Inference object has no _pipeline")

    # Reduce inference steps for faster generation (default is 25)
    INFERENCE_STEPS = 12
    pipe.override_ss_generator_cfg_config(
        pipe.models["ss_generator"],
        inference_steps=INFERENCE_STEPS,
        cfg_strength=7,
        cfg_interval=[0, 500],
        rescale_t=3,
        cfg_strength_pm=0.0,
    )
    pipe.override_slat_generator_cfg_config(
        pipe.models["slat_generator"],
        inference_steps=INFERENCE_STEPS,
        cfg_strength=1,
        cfg_interval=[0, 500],
        rescale_t=1,
    )
    log(f"Inference steps set to {INFERENCE_STEPS} for both generators")

    load_time = time.time() - load_start
    log(f"Pipeline loaded in {load_time:.1f}s")
    return pipe, to_glb


def load_inputs(image_path: str, mask_path: str):
    """Load and validate image + mask. Returns (image, mask) as uint8 arrays."""
    image = np.array(Image.open(image_path).convert("RGB"))
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    mask = np.array(Image.open(mask_path).convert("L"))
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    mask_pixel_count = int(np.sum(mask_u8 > 0))
    log(
        f"Image {image.shape}, mask {mask_u8.shape}, "
        f"masked pixels: {mask_pixel_count} "
        f"({100.0 * mask_pixel_count / mask_u8.size:.2f}%)"
    )

    if mask_pixel_count == 0:
        raise ValueError(
            "Mask is empty (no valid pixels). The mask must contain at least some "
            "pixels with value > 0 to indicate the object region."
        )
    if mask_pixel_count < 100:
        log(f"WARNING: Mask has very few pixels ({mask_pixel_count})")

    if image.shape[:2] != mask_u8.shape:
        raise ValueError("Image/mask dimensions mismatch")

    return image, mask_u8


def select_mesh(output):
    """Pick a mesh-like element from the pipeline output for to_glb()."""
    mesh_data = output.get("mesh") or output.get("slat_mesh")
    if mesh_data is None:
        return None
    if isinstance(mesh_data, (list, tuple)):
        for m in mesh_data:
            if hasattr(m, "vertices") or hasattr(m, "triangles"):
                return m
            if isinstance(m, dict) and (
                "vertices" in m or "faces" in m or "triangles" in m
            ):
                return m
        return None
    return mesh_data


def run_job(pipe, to_glb, job, assets_dir):
    """Run one generation job. Returns the result dict to emit."""
    job_id = job["job_id"]
    seed = int(job.get("seed", 42))

    image, mask = load_inputs(job["image_path"], job["mask_path"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    pointmap = make_synthetic_pointmap(image, z=1.0)

    log(f"Job {job_id}: running pipe.run()...")
    start = time.time()
    with torch.no_grad():
        output = pipe.run(
            image=image,
            mask=mask,
            seed=seed,
            pointmap=pointmap,
            decode_formats=["gaussian", "mesh"],
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=True,
            use_vertex_color=True,
        )
    torch.cuda.synchronize()
    inference_seconds = time.time() - start
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
    log(f"Job {job_id}: inference done in {inference_seconds:.1f}s (peak {gpu_mem:.1f} GB)")

    if not isinstance(output, dict) or "gaussian" not in output:
        raise RuntimeError(
            f"Pipeline output has no 'gaussian' "
            f"(keys: {list(output.keys()) if isinstance(output, dict) else type(output).__name__})"
        )

    gs_list = output["gaussian"]
    gs = gs_list[0] if isinstance(gs_list, (list, tuple)) else gs_list

    mesh_data = select_mesh(output)
    if mesh_data is None:
        raise RuntimeError(
            f"No mesh data in pipeline output (keys: {list(output.keys())})"
        )

    log(f"Job {job_id}: baking GLB via to_glb()...")
    glb_start = time.time()
    glb_obj = to_glb(
        app_rep=gs,
        mesh=mesh_data,
        simplify=0.95,
        fill_holes=True,
        fill_holes_max_size=0.04,
        texture_size=1024,
        debug=False,
        verbose=True,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        rendering_engine="nvdiffrast",
    )
    if glb_obj is None or not hasattr(glb_obj, "export"):
        raise RuntimeError(f"to_glb() returned unusable object: {type(glb_obj).__name__}")

    mesh_filename = f"mesh_{uuid.uuid4().hex[:8]}.glb"
    mesh_path = os.path.join(assets_dir, mesh_filename)
    os.makedirs(assets_dir, exist_ok=True)
    glb_obj.export(mesh_path)

    if not os.path.exists(mesh_path):
        raise RuntimeError("GLB export reported success but file not found")

    mesh_size = os.path.getsize(mesh_path)
    log(f"Job {job_id}: GLB saved ({mesh_size} bytes, bake took {time.time() - glb_start:.1f}s)")

    metadata_path = os.path.join(assets_dir, f"{mesh_filename}.metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "filename": mesh_filename,
                "created_at": datetime.now().isoformat(),
                "size_bytes": mesh_size,
                "format": "glb",
                "has_textures": True,
                "method": "SAM3D_to_glb_native",
            },
            f,
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "mesh_url": f"/assets/{mesh_filename}",
        "mesh_format": "glb",
        "mesh_size_bytes": mesh_size,
        "inference_seconds": round(inference_seconds, 1),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python worker_3d.py <assets_dir>", file=sys.stderr)
        sys.exit(1)
    assets_dir = sys.argv[1]

    log(f"GPU available: {torch.cuda.is_available()}")
    pipe, to_glb = load_pipeline()

    print("READY", flush=True)
    log("Worker ready, waiting for jobs on stdin...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job_id = None
        try:
            job = json.loads(line)
            job_id = job["job_id"]
            result = run_job(pipe, to_glb, job, assets_dir)
            emit(result)
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            emit({"job_id": job_id, "status": "failed", "error": str(e)})

    log("stdin closed, exiting")


if __name__ == "__main__":
    main()
