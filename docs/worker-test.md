# Test: Persistent Worker (3D-Generierung)

Anleitung zum Verifizieren des Worker-Umbaus auf dem RunPod-Pod.
Erwartung: erste Generierung nach API-Start normal schnell (Modell lädt beim API-Start,
nicht beim Request), jede weitere deutlich unter der alten Cold-Start-Zeit.

## 0. Repo auf Pod aktualisieren

```bash
cd /workspace/sam3d-api
git pull
```

Neue/geänderte Dateien: `worker_3d.py` (neu), `api.py`, Startskript.

## 1. Diagnose vorab (einmalig)

```bash
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs
conda activate /workspace/envs/sam3d-objects

# flash-attn Build aus setup.sh Step 9 pruefen:
python -c "import flash_attn; print('flash-attn OK:', flash_attn.__version__)"
```

Schlägt der Import fehl → flash-attn-Build auf sm_120 kaputt, notieren (separates Thema, blockiert Worker nicht).

## 2. API starten

```bash
bash /workspace/install_conda_start_env_host_api.sh
```

Im Log prüfen (Reihenfolge):

1. `Copying checkpoints to local disk (/root/sam3d-checkpoints)...` — nur beim ersten Start pro Pod
2. `✓ SAM 2 model and processor initialized successfully`
3. `[API] Starting persistent 3D worker...`
4. `[Worker] Loading pipeline from /root/sam3d-checkpoints/hf/pipeline.yaml...`
5. `[Worker] Pipeline loaded in XXs` — **diese Zahl notieren** (das ist die Zeit, die vorher bei JEDEM Request anfiel)
6. `[API] ✓ 3D worker ready`

**Falls Schritt 4/5 mit Pfad-Fehler scheitert** (pipeline.yaml referenziert evtl. Pfade relativ zum Arbeitsverzeichnis): `export SAM3D_CHECKPOINT_DIR=` (leer) vor uvicorn-Start → Worker fällt auf alten Pfad `./sam-3d-objects/checkpoints/hf/pipeline.yaml` zurück. Dann bitte Fehlermeldung melden.

## 3. Health-Check

```bash
curl -s http://localhost:8000/health
```

Erwartet: `"worker_ready": true` (direkt nach Start ggf. noch `false` — Modell lädt; Requests warten dann automatisch).

## 4. Generierung testen

Mit vorhandenem Testbild + Maske (base64):

```bash
IMG=$(base64 -w0 test_img.png)
MASK=$(base64 -w0 test_img_mask.png)
TASK=$(curl -s -X POST http://localhost:8000/generate-3d \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"mask\":\"$MASK\",\"seed\":42}" | python -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
echo "Task: $TASK"

# Pollen bis completed/failed:
watch -n3 "curl -s http://localhost:8000/generate-3d-status/$TASK"
```

Erwartete Response bei Erfolg (KEINE base64-Blobs mehr):

```json
{"task_id":"...","status":"completed","progress":100,
 "mesh_url":"/assets/mesh_xxxxxxxx.glb","mesh_format":"glb",
 "mesh_size_bytes":..., "inference_seconds":...}
```

GLB herunterladen und prüfen:

```bash
curl -s -o /tmp/test.glb "http://localhost:8000$(curl -s http://localhost:8000/generate-3d-status/$TASK | python -c "import sys,json;print(json.load(sys.stdin)['mesh_url'])")"
ls -la /tmp/test.glb   # > 0 Bytes; lokal in glTF-Viewer oeffnen
```

## 5. Timing-Nachweis (Kernpunkt)

Direkt einen **zweiten** Request senden (Schritt 4 wiederholen).

- API-Log: **kein** erneutes `Loading pipeline` — Worker bleibt warm
- Gesamtzeit Request→completed stoppen; Ziel: **< 60s**
- `inference_seconds` in Response = reine Pipeline-Zeit; Differenz zur Gesamtzeit = GLB-Bake + Overhead

Zum Vergleich alte Architektur: Pipeline-Load (Schritt 2, Punkt 5) + Inferenz + GIF + ASCII-PLY pro Request.

## 6. Fehlerfall / Robustheit

Leere (schwarze) Maske senden → erwartet:

- Status `failed` mit `"error": "Mask is empty..."`
- API-Log: Worker lebt weiter (kein `worker exited`)
- Dritter, normaler Request funktioniert danach ohne Neustart

Crasht der Worker doch (z.B. CUDA OOM): nächster Request startet ihn automatisch neu (Log: `Worker not running, restarting...`) — der Request wartet dann die Ladezeit ab.

## 7. Unity

Editor-Play-Mode gegen Pod-API, kompletter Flow (Segmentieren → Generieren). Console erwartet:

```
GenerationManager: Completed — mesh_url=/assets/mesh_....glb, format=glb, size=... bytes, inference=...s
GenerationManager: Downloading mesh from http://...:8000/assets/mesh_....glb
GenerationManager: Mesh downloaded — ... bytes
```

`GenerationResult.MeshBytes` enthält danach das GLB binär (Weiterverarbeitung/Anzeige weiterhin TBD).

## Bekannte Grenzen

- Jobs laufen strikt nacheinander (eine GPU, ein Worker). Zweiter Request wartet, bis erster fertig ist.
- Nach Job-Timeout (600s) bleibt der hängende Job im Worker aktiv; nächster Request kann sich dahinter einreihen. Falls das auftritt: API neu starten und Logs melden.
- `generate_3d_subprocess.py` bleibt unverändert als Referenz, wird aber nicht mehr aufgerufen.
