"""
Client fuer die sam3d-api: Bild hochladen, an einem Punkt segmentieren, GLB zurueckbekommen.

Nur Standardbibliothek — laeuft mit jedem Python 3.8+ ohne pip install.

    python generate3d.py --url https://<pod-id>-8000.proxy.runpod.net \
                         --image stuhl.jpg --x 400 --y 300 --out stuhl.glb

Ablauf: POST /segment (Klickpunkt -> Binaermaske) -> POST /generate-3d -> Polling
-> GET mesh_url.

Bewusst /segment und nicht /segment-binary: letzteres liefert das maskierte RGB-Bild,
keine Binaermaske. Als 'mask' an /generate-3d gegeben wuerden dunkle Bildbereiche innerhalb
des Objekts zu Loechern in der Maske.

Mit --mask laesst sich eine fertige Maske uebergeben und die Segmentierung ueberspringen.
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request


def post(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Basis-URL der API, z.B. https://<pod-id>-8000.proxy.runpod.net")
    p.add_argument("--image", required=True)
    p.add_argument("--x", type=int, help="Klickpunkt X (Pixel) — noetig ohne --mask")
    p.add_argument("--y", type=int, help="Klickpunkt Y (Pixel) — noetig ohne --mask")
    p.add_argument("--mask", help="fertige Binaermaske (PNG); ueberspringt die Segmentierung")
    p.add_argument("--out", default="mesh.glb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=600, help="Sekunden bis zum Abbruch des Pollings")
    args = p.parse_args()

    if not args.mask and (args.x is None or args.y is None):
        p.error("entweder --mask oder --x/--y angeben")

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

    urllib.request.urlretrieve(f"{base}{status['mesh_url']}", args.out)
    print(
        f"Fertig: {args.out} ({status['mesh_size_bytes']} Bytes, "
        f"inference {status['inference_seconds']}s)"
    )


if __name__ == "__main__":
    main()
