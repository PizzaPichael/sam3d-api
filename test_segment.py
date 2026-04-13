"""
Quick test script for /segment and /segment-binary endpoints.

Usage:
    python test_segment.py --image path/to/image.jpg --points 200,150 300,200
    python test_segment.py --image path/to/image.jpg --points 200,150 --endpoint segment --pick 2
    python test_segment.py --image path/to/image.jpg --points 200,150 --url https://<pod-id>-8000.proxy.runpod.net

Parameters:
    --image     Path to input image (PNG or JPEG)
    --points    One or more click points as x,y pairs (e.g. 200,150 300,200)
    --endpoint  segment | segment-binary (default: segment-binary)
    --pick      Which mask to copy as final output (1-3, only for --endpoint segment)
                If omitted, all masks are saved individually but none is copied to --out
    --url       API base URL (default: http://localhost:8000)
    --out       Output file path for the final mask (default: mask_preview.png)

/segment        — single point only, returns up to 3 mask candidates as PNGs
/segment-binary — multi-point, returns one merged mask PNG (recommended)
"""

import argparse
import base64
import shutil
import sys
from pathlib import Path

import requests


def call_segment(url, image_b64, points, out, pick=None):
    # /segment only supports a single point
    if len(points) > 1:
        print("Warning: /segment only uses the first point, ignoring the rest.")
    point = points[0]
    payload = {"image": image_b64, "x": point["x"], "y": point["y"], "multimask_output": True}

    print(f"Sending to {url}/segment ...")
    resp = requests.post(f"{url}/segment", json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    masks = data.get("masks", [])
    scores = data.get("scores", [])
    print(f"Received {len(masks)} mask(s)")

    out_path = Path(out)
    saved_paths = []
    for i, mask_entry in enumerate(masks):
        mask_b64 = mask_entry["mask"]
        score = mask_entry.get("score", 0.0)
        path = out_path.with_stem(f"{out_path.stem}_{i+1}")
        Path(path).write_bytes(base64.b64decode(mask_b64))
        saved_paths.append(path)
        print(f"  Mask {i+1} (score {score:.4f}) saved to: {path}")

    if pick is not None:
        if 1 <= pick <= len(saved_paths):
            shutil.copy(saved_paths[pick - 1], out_path)
            print(f"Mask {pick} copied to: {out_path}")
        else:
            print(f"Warning: --pick {pick} out of range, got {len(saved_paths)} mask(s).")


def call_segment_binary(url, image_b64, points, out):
    payload = {"image": image_b64, "points": points}

    print(f"Sending {len(points)} point(s) to {url}/segment-binary ...")
    resp = requests.post(f"{url}/segment-binary", json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"Score: {data.get('score', 'n/a')}")

    mask_bytes = base64.b64decode(data["mask"])
    Path(out).write_bytes(mask_bytes)
    print(f"Mask saved to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image (PNG or JPEG)")
    parser.add_argument("--points", nargs="+", required=True, help="Click points as x,y pairs (e.g. 200,150 300,200)")
    parser.add_argument("--endpoint", choices=["segment", "segment-binary"], default="segment-binary",
                        help="Which endpoint to call (default: segment-binary)")
    parser.add_argument("--pick", type=int, default=None,
                        help="Which mask to save as final output (1-3, only for --endpoint segment)")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--out", default="mask_preview.png", help="Output mask file path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    points = []
    for p in args.points:
        x, y = p.split(",")
        points.append({"x": float(x), "y": float(y)})

    if args.endpoint == "segment":
        call_segment(args.url, image_b64, points, args.out, args.pick)
    else:
        call_segment_binary(args.url, image_b64, points, args.out)


if __name__ == "__main__":
    main()
