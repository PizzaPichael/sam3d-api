"""
Client fuer die sam3d-api: Bild hochladen, an einem Punkt segmentieren, GLB zurueckbekommen.

Drei Wege zur Maske:

    # 1. Klickpunkt -> SAM2 segmentiert
    python generate3d.py --url https://<pod-id>-8000.proxy.runpod.net \
                         --image stuhl.jpg --x 400 --y 300 --out stuhl.glb

    # 2. freigestelltes Bild (transparenter Hintergrund) -> Maske aus Alpha-Kanal
    python generate3d.py --url ... --image stuhl.png --out stuhl.glb

    # 3. fertige Binaermaske
    python generate3d.py --url ... --image stuhl.jpg --mask stuhl_mask.png

Ablauf: POST /segment (Klickpunkt -> Binaermaske) -> POST /generate-3d -> Polling
-> GET mesh_url.

Bewusst /segment und nicht /segment-binary: letzteres liefert das maskierte RGB-Bild,
keine Binaermaske. Als 'mask' an /generate-3d gegeben wuerden dunkle Bildbereiche innerhalb
des Objekts zu Loechern in der Maske.

Weg 1 und 3 brauchen nur die Standardbibliothek. Weg 2 braucht zusaetzlich Pillow
(pip install pillow), um den Alpha-Kanal zu lesen.
"""

import argparse
import base64
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request


# Cloudflare vor dem RunPod-Proxy blockt den urllib-Default-User-Agent
# ("Python-urllib/3.x") mit 403 Forbidden. Deshalb ueberall einen eigenen setzen.
USER_AGENT = "sam3d-api-client/1.0"


def post(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def download(url, dest, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def mask_from_alpha(path):
    """Binaermaske aus dem Alpha-Kanal eines freigestellten Bildes."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit(
            "Maske aus Alpha-Kanal braucht Pillow (pip install pillow).\n"
            "Alternativ --x/--y oder --mask angeben."
        )

    image = Image.open(path)
    if "A" not in image.getbands():
        sys.exit(
            f"{path} hat keinen Alpha-Kanal — bitte --x/--y (Klickpunkt) "
            "oder --mask angeben."
        )

    mask = image.getchannel("A").point([0] + [255] * 255, mode="L")
    opaque = mask.histogram()[255]
    if opaque == 0:
        sys.exit(f"{path} ist komplett transparent — keine Maske ableitbar.")
    if opaque == mask.width * mask.height:
        sys.exit(
            f"{path} ist komplett undurchsichtig — bitte --x/--y (Klickpunkt) "
            "oder --mask angeben."
        )

    print(f"Maske aus Alpha-Kanal: {opaque} von {mask.width * mask.height} Pixeln")
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Basis-URL der API, z.B. https://<pod-id>-8000.proxy.runpod.net")
    p.add_argument("--image", required=True)
    p.add_argument("--x", type=int, help="Klickpunkt X (Pixel)")
    p.add_argument("--y", type=int, help="Klickpunkt Y (Pixel)")
    p.add_argument("--mask", help="fertige Binaermaske (PNG); ueberspringt die Segmentierung")
    p.add_argument("--out", default="mesh.glb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=600, help="Sekunden bis zum Abbruch des Pollings")
    args = p.parse_args()

    if (args.x is None) != (args.y is None):
        p.error("--x und --y nur gemeinsam angeben")

    base = args.url.rstrip("/")
    image_b64 = b64_file(args.image)

    try:
        health = get(f"{base}/health")
    except urllib.error.URLError as e:
        sys.exit(f"API nicht erreichbar unter {base}: {e}")
    print(f"API ok — worker_ready={health.get('worker_ready')}")
    if not health.get("worker_ready"):
        print("  Worker laedt noch die Pipeline (~80s), der Request wartet automatisch.")

    if args.mask:
        mask_b64 = b64_file(args.mask)
        print(f"Maske aus Datei: {args.mask}")
    elif args.x is None:
        mask_b64 = mask_from_alpha(args.image)
    else:
        print(f"Segmentiere bei ({args.x}, {args.y}) ...")
        seg = post(
            f"{base}/segment",
            {"image": image_b64, "x": args.x, "y": args.y, "multimask_output": True},
        )
        if not seg.get("success"):
            sys.exit(f"Segmentierung fehlgeschlagen: {seg.get('error')}")
        best = max(seg["masks"], key=lambda m: m["score"])
        print(f"  {len(seg['masks'])} Masken, beste Score {best['score']:.3f}")
        mask_b64 = best["mask"]

    task = post(
        f"{base}/generate-3d",
        {"image": image_b64, "mask": mask_b64, "seed": args.seed},
    )
    task_id = task["task_id"]
    print(f"Task {task_id} — generiere ...")

    deadline = time.time() + args.timeout
    last = None
    while True:
        if time.time() > deadline:
            sys.exit(f"Timeout nach {args.timeout}s — Task {task_id}")
        time.sleep(5)
        status = get(f"{base}/generate-3d-status/{task_id}")
        if status["status"] != last:
            print(f"  {status['status']}")
            last = status["status"]
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            sys.exit(f"Generierung fehlgeschlagen: {status.get('error')}")

    download(f"{base}{status['mesh_url']}", args.out)
    print(
        f"Fertig: {args.out} ({status['mesh_size_bytes']} Bytes, "
        f"inference {status['inference_seconds']}s)"
    )


if __name__ == "__main__":
    main()
