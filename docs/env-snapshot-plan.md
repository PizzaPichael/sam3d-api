# Env-Snapshot statt Env-Copy — Plan & Runbook

Ziel: die Env-Bereitstellung beim Pod-Start von ~2 h auf ~1–3 min bringen.

Stand: 28.07.2026 — umgesetzt und gemessen. Env-Bereitstellung **74m42s → 22,5 s**.
Offen ist nur noch der Nachweis am echten Pod-Start (Schritt 5).

---

## 1. Problem

`install_conda_start_env_host_api.sh` spiegelt bei jedem Pod-Start das Conda-Env von
`/workspace/envs/sam3d-objects` nach `/root/sam3d-env-local` (Zeilen 87–103) und kopiert
zusätzlich 13,3 GB Checkpoints (Zeilen 130–137). Beides landet auf der Container-Disk und
ist nach jedem Pod-Stop wieder weg.

Gemessen: **~2 h** pro Pod-Start.

### Warum das Env so lange dauert, die Checkpoints aber nicht

| | Größe | Dateien | Engpass |
|---|---|---|---|
| Conda-Env | ~15–20 GB | ~100–200k | **FUSE-Latenz pro Datei** |
| Checkpoints | 13,3 GB | ~20 | Bandbreite |

`cp -a` ist single-threaded → ~150k × open/stat/read/close über MooseFS. Bandbreite ist
dabei fast idle. Das ist die eigentliche Wartezeit.

### Was die Copy *nicht* beschleunigt: die Inferenz

Pro Request liest nichts mehr von Storage. Modelle liegen nach `load_pipeline()`
(`worker_3d.py:128-187`) auf der GPU, Python-Module im RAM. Der ursprüngliche
„Inferenz zu langsam"-Effekt war Cold Start **pro Request** — gelöst durch den Persistent
Worker (Hebel 6 in [`performance-changes.md`](performance-changes.md)).

Env-Mirror und Checkpoint-Copy beschleunigen also nur noch den **einmaligen Start**.
Es gibt keinen Zielkonflikt „schnelles Setup vs. schnelle Inferenz".

---

## 2. Lösung

Das Env einmalig als **ein** komprimiertes Tar-Archiv auf dem Volume ablegen. Beim
Pod-Start wird entpackt statt kopiert: ~150k FUSE-Roundtrips werden zu einem
sequentiellen Stream, geschrieben wird auf lokale NVMe.

```
Snapshot einmalig bauen  →  /workspace/env-snapshot.tzst   (4,62 GiB, 16,5 s)
        ↓
jeder Pod-Start:  zstd -dc | tar -x  nach /root/sam3d-env-local   (22,5 s)
```

Gemessen am 28.07.2026: Env 12,7 GiB → Archiv 4,62 GiB (36,5 %). Erstellung vom lokalen
Mirror 16,5 s, Wiederherstellung inkl. Lesen vom Volume 22,5 s. Vorher: 74m42s.

Das Original unter `/workspace/envs/sam3d-objects` bleibt **unangetastet** — `resume.sh`
und Repair-Sessions arbeiten weiter dagegen.

### Warum vom lokalen Mirror tarren, nicht vom Volume

Der Snapshot muss das Env einmal komplett lesen. Vom Volume gelesen kostet das dieselben
~2 h. Läuft die laufende Copy durch, liegt das Env lokal auf NVMe — dann liest `tar` von
lokaler Platte (Minuten) und schreibt eine große sequentielle Datei aufs Volume (unkritisch).

**Deshalb: eine laufende Env-Copy nie abbrechen, um „schneller" mit dem Snapshot
anzufangen.** Sie ist genau die teure Leseoperation, die man einmal zahlen muss.

### Abbruch-Falle

Wird die Copy doch abgebrochen, zwingend:

```bash
rm -rf /root/sam3d-env-local
```

Der Check in `install_conda_start_env_host_api.sh:92` ist `[ -x "$LOCAL_ENV_MIRROR/bin/python" ]`.
`cp` kopiert alphabetisch, `bin/` ist früh fertig — ein Teil-Mirror besteht den Test, das
Script überspringt das Mirroring und startet gegen ein halbes Env. Gleiche Fehlerklasse wie
[`setup-fixes.md`](setup-fixes.md) Punkt 12, andere Ursache.

---

## 3. Runbook

Alles ab Schritt 1 in einer **zweiten Shell** (tmux: `Ctrl-b c`, oder zweite SSH-Session).

### Schritt 0 — Status prüfen, laufen lassen

```bash
du -sh /root/sam3d-env-local /root/sam3d-checkpoints 2>/dev/null
pgrep -fa "cp -a"
```

- Nur `sam3d-env-local`, wächst → Env-Mirror läuft
- Beide da, `sam3d-checkpoints` wächst → Env fertig, Checkpoints laufen
- Kein `cp -a` mehr, dafür `uvicorn` → Copies fertig

Warten auf `[API] ✓ 3D worker ready`. Bei Verbindungsabbruch: `tmux attach -t sam3d`.

### Schritt 1 — Env verifizieren

Erst danach snapshotten, sonst friert man einen Defekt ein.

```bash
curl -s http://localhost:8000/health          # "worker_ready": true
```

```bash
export LIDRA_SKIP_INIT=true XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
/root/sam3d-env-local/bin/python -c "
import torch; print('torch', torch.__version__, torch.cuda.is_available())
import numpy; print('numpy', numpy.__version__)
import kaolin, pytorch3d, nvdiffrast; print('native OK')
import flash_attn; print('flash-attn', flash_attn.__version__)
import xformers.ops; print('xformers OK')
import omegaconf, hydra; print('omegaconf/hydra OK')
import sam3d_objects; print('sam3d OK')
" 2>&1 | grep -v WARNING
```

Bewusst `/root/sam3d-env-local/bin/python` — das ist der Interpreter, der in den Snapshot
geht. `omegaconf/hydra OK` muss kommen (siehe [`setup-fixes.md`](setup-fixes.md) Punkt 19).

Testbild erzeugen, falls keins auf dem Pod liegt:

```bash
cd /workspace/sam3d-api
/root/sam3d-env-local/bin/python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (512,512), (120,120,120))
d = ImageDraw.Draw(img); d.ellipse([150,150,362,362], fill=(200,80,60))
img.save('/tmp/t.png')
m = Image.new('L', (512,512), 0)
d = ImageDraw.Draw(m); d.ellipse([150,150,362,362], fill=255)
m.save('/tmp/t_mask.png')
print('ok')
"
```

```bash
IMG=$(base64 -w0 /tmp/t.png)
MASK=$(base64 -w0 /tmp/t_mask.png)
TASK=$(curl -s -X POST http://localhost:8000/generate-3d \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"mask\":\"$MASK\",\"seed\":42}" \
  | /root/sam3d-env-local/bin/python -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
echo "Task: $TASK"

for i in $(seq 1 60); do
  curl -s http://localhost:8000/generate-3d-status/$TASK; echo
  sleep 5
done
```

**Gate:** `"status":"completed"` und `mesh_size_bytes` > 0. Bei `failed` erst den Fehler
klären, nicht snapshotten.

Baseline notieren: `[Worker] Pipeline loaded in XXs` und `inference_seconds`.

### Schritt 2 — Snapshot bauen

```bash
command -v zstd || apt-get install -y zstd
```

Kein zstd verfügbar → `| zstd ...` weglassen und direkt nach `/workspace/env-snapshot.tar`
schreiben. Funktioniert genauso, nur größer.

```bash
df -h /workspace          # ~15 GB frei als Puffer
du -sh /root/sam3d-env-local
```

Excludes in eine Datei, damit die tar-Zeile kurz bleibt — ein über mehrere Zeilen
umgebrochener Paste lässt `tar` ohne Quelle laufen und erzeugt ein **13-Byte-Archiv**, das
beim nächsten Start als gültiger Snapshot erkannt würde:

```bash
printf '*.a\n./include\n./share/doc\n./share/man\n./pkgs\n' > /root/snap-exclude.txt
```

```bash
time tar -C /root/sam3d-env-local -X /root/snap-exclude.txt -cf - . | zstd -T0 -3 -o /workspace/env-snapshot.tzst
```

Dauert ~17 s. Bricht es in unter einer Sekunde mit `Cowardly refusing to create an empty
archive` ab, war der Paste zerhackt → `rm -f /workspace/env-snapshot.tzst` und neu.

Die API läuft dabei weiter — reines Lesen.

**Excludes:** `*.a` sind die Static-Libs der `nvidia-*`-Wheels (mehrere GB) und `include/`
die CUDA-Header — beides reine Build-Zeit-Artefakte, zur Laufzeit tot. Verkleinern
Snapshot **und** Entpackzeit.

### Schritt 3 — Snapshot prüfen

```bash
ls -lh /workspace/env-snapshot.tzst
zstd -t /workspace/env-snapshot.tzst
zstd -dc /workspace/env-snapshot.tzst | tar -tf - | wc -l
zstd -dc /workspace/env-snapshot.tzst | tar -tf - | grep -c "site-packages/torch/"
```

Erwartung: sechsstellige Dateizahl, `torch`-Treffer > 0 (gemessen: 13.572). Beides null →
Snapshot unbrauchbar, neu bauen.

Restore vorab testen, ohne den laufenden Mirror anzufassen (braucht ~12,7 GB frei auf `/`):

```bash
mkdir -p /root/snap-test && time (zstd -dc /workspace/env-snapshot.tzst | tar -C /root/snap-test -xf -)
```
```bash
/root/snap-test/bin/python -c "import torch, kaolin, pytorch3d, nvdiffrast, sam3d_objects; print('RESTORE OK', torch.__version__)"
```
```bash
rm -rf /root/snap-test
```

### Schritt 4 — Script-Umbau ✅ erledigt

In `install_conda_start_env_host_api.sh`:

- Snapshot vorhanden → `zstd -dc | tar -x`, sonst Fallback auf `copy_with_progress`.
  Nach dem Entpacken wird `bin/python` geprüft und ein unvollständiger Mirror gelöscht —
  sonst besteht er beim nächsten Start den `-x`-Test und die API läuft gegen ein halbes Env.
- Checkpoint-Copy entfernt, `SAM3D_CHECKPOINT_DIR` zeigt aufs Volume (siehe Abschnitt 4)
- `TORCH_HOME=/workspace/torch-cache`, `HF_HOME=/workspace/hf-home`
- `XFORMERS_DISABLED=1` ([`setup-fixes.md`](setup-fixes.md) Punkt 20)

In `setup.sh`:

- Step 5: die 12 fehlenden Laufzeit-Pakete ([`setup-fixes.md`](setup-fixes.md) Punkt 21)
- Step 11: Verifikation importiert jetzt `notebook/inference.py` statt nur `sam3d_objects`

**Offen gelassen:** `ASSETS_DIR` (`api.py:100`) schreibt jedes GLB weiterhin aufs Volume.
Bei ~1,3 MB pro Mesh ist der Effekt marginal, und lokal abgelegte Assets wären nach einem
Pod-Stop weg.

### Schritt 5 — Nachweis

Pod stoppen → starten → `bash install_conda_start_env_host_api.sh`.
Erwartung: Env-Bereitstellung ~1–3 min statt ~2 h. Zeit gegen die Baseline halten.

### Reihenfolge-Regel

verifizieren → snapshotten → erst dann Script ändern. Umgekehrt wandert ein kaputtes Env in
den Snapshot und wird bei jedem künftigen Start wiederhergestellt.

### Snapshot-Wartung

Nach jedem Eingriff ins Env (Repair-Session, neues Paket) ist der Snapshot veraltet →
Schritt 2 wiederholen. Sonst überschreibt der nächste Pod-Start die Änderung stillschweigend.

---

## 4. Erledigt: Checkpoint-Copy ist raus

Gemessen am 28.07.2026, Pipeline-Load direkt vom Volume: **80,9 s**.

| Anteil | Zeit |
|---|---|
| moge `model.pt` Download (Internet) | 4 s |
| `ss_generator.ckpt` 6,3 GB vom Volume | 21 s |
| `slat_generator.ckpt` 4,6 GB vom Volume | 15 s |
| kleine Decoder | ~4 s |
| dinov2 Download (Internet) | ~10 s |
| ss_generator + slat_generator ein **zweites** Mal (Pipeline lädt doppelt) | 11 s |

Das Volume liefert ~300 MB/s. Eine lokale Kopie kostet allein ≥45 s, danach wird immer noch
geladen, und sie belegt 13 GB von 40 GB Container-Disk — auf der schon der 13-GB-Env-Mirror
liegt. Der Load fällt einmal pro Pod an, nicht pro Request (Persistent Worker).

`install_conda_start_env_host_api.sh` zeigt `SAM3D_CHECKPOINT_DIR` deshalb direkt aufs
Volume und bricht ab, wenn dort keine `hf/pipeline.yaml` liegt. Hebel 5 in
[`performance-changes.md`](performance-changes.md) ist damit hinfällig.

**Die beiden Internet-Downloads** landeten auf der Container-Disk und kamen bei jedem
Pod-Start neu. Sie nutzen unterschiedliche Caches: dinov2 (1,13 GB) geht über `torch.hub`
→ `TORCH_HOME`, moge (1,26 GB) über `huggingface_hub` → `HF_HOME`. Beide Variablen stehen
jetzt im Startscript; nur eine zu setzen behebt nur die Hälfte.

---

## 5. Verworfen fürs Erste: Custom Docker Image

Env + Checkpoints ins Image backen, in eine Registry pushen, als RunPod-Template setzen.
Pod-Start zieht das Image auf den Host — MooseFS kommt im Startpfad gar nicht mehr vor.
Wäre die sauberste Lösung.

Zwei Blocker:

1. **Kein Docker im Pod.** RunPod-Pods sind selbst Container ohne Docker-Daemon.
   Docker-in-Docker braucht `--privileged`/`CAP_SYS_ADMIN` — dieselbe Restriktion, die schon
   den `mount --bind` in `install_conda_start_env_host_api.sh:97` blockiert. Der Build
   müsste auf einer separaten Maschine laufen (GPU dafür nicht nötig, nur nvcc aus dem
   Base-Image).
2. **Upload.** ~35 GB in eine Registry pushen — von einer Heimleitung der eigentliche
   Schmerzpunkt, nicht der Build.

Ein Image auf dem Volume abzulegen bringt nichts: es müsste beim Start von dort gelesen
werden, also exakt das Problem, das gelöst werden soll.

Der Tar-Snapshot holt den Großteil des Gewinns ohne Registry und ohne Build-Maschine.
Docker lohnt erst bei sehr häufigem Stop/Start — dann eher als Kompromiss: Env im Image
(~12–15 GB nach Excludes), Checkpoints bleiben auf dem Volume.
