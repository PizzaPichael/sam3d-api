# RunPod Setup – SAM2 + sam3d-objects API

Vollständige Anleitung zum Aufsetzen eines neuen Pods, von der GPU-Auswahl bis zur
laufenden API.

Für **Fehlerdiagnose** siehe [`setup-fixes.md`](setup-fixes.md) — dort ist dokumentiert,
*warum* die einzelnen Steps in dieser Reihenfolge stehen.
Für **Verifikation des Workers** siehe [`worker-test.md`](worker-test.md).

---

## 1. Pod-Konfiguration

### 1.1 GPU auswählen

**Harte Anforderung: ≥32 GB VRAM auf *einer* Karte.**

Meta dokumentiert 32 GB als Minimum, 40+ GB als empfohlen. Der Peak liegt bei ~30 GB:
Model Loading ~15–20 GB, Forward Pass nochmal +10–15 GB.

Achtung bei den 32-GB-Karten: `nvidia-smi` meldet dort real ~32.623 MiB = **31,9 GiB**.
Das reicht (die PRO 4500 läuft produktiv), lässt aber keinen Spielraum — auf diesen
Karten strikt **ein Worker**.

> **VRAM addiert sich nicht über GPUs hinweg.** sam3d-objects unterstützt kein
> Model-Sharding — 2 × 24 GB sind *nicht* 48 GB nutzbar, sondern ein OOM beim Laden.
> Mehrere Karten helfen nur für mehrere unabhängige Worker, und dann braucht
> **jede einzelne** ≥32 GB.

#### Läuft ohne Rebuild auf dem bestehenden Volume (sm_120)

Die nativen Builds im `/workspace`-Env sind mit `TORCH_CUDA_ARCH_LIST="12.0"`
kompiliert. Diese Karten sind damit drop-in kompatibel:

| Prio | GPU | VRAM | Preis/h | Hinweis |
|------|-----|------|---------|---------|
| **1** | **RTX 5090** | 32 GB | ~$0,69 (Spot) / ~$0,99 | Schnellste *und* günstigste sm_120-Option, 21.760 CUDA-Cores |
| **2** | **RTX PRO 4500 Blackwell** | 32 GB | ~$1,15 | Referenz-Setup dieser Doku. Teurer, ~16 % langsamer |

#### Erfordern Neu-Kompilierung (~25 Min Setup)

Andere Architektur → alle `.so`-Dateien (pytorch3d, gsplat, nvdiffrast, kaolin)
müssen mit passender `TORCH_CUDA_ARCH_LIST` neu gebaut werden.

| GPU | VRAM | Arch | Preis/h | Hinweis |
|-----|------|------|---------|---------|
| A40 | 48 GB | Ampere sm_86 | ~$0,44 | Bestes Preis/VRAM-Verhältnis, gute Reserve |
| RTX A6000 | 48 GB | Ampere sm_86 | ~$0,49 | Gleichwertig zu A40 |
| L40S | 48 GB | Ada sm_89 | ~$0,86 | Trifft Metas „empfohlen" (Ada + 40 GB+) exakt |
| A100 80 GB | 80 GB | Ampere sm_80 | ~$1,39 | Viel Headroom, mehrere Worker möglich |
| H100 PCIe | 80 GB | Hopper sm_90 | ~$2,89 | Nur bei kritischem Durchsatz |

#### Ungeeignet

| GPU | Grund |
|-----|-------|
| RTX 4090 (24 GB) | Unter dem 32-GB-Minimum |
| RTX PRO 4000 (24 GB) | Unter Minimum — auch als 2× nicht, siehe Hinweis oben |

*Preise: RunPod Community Cloud, Stand Juli 2026. Secure Cloud kostet ca. das Doppelte.*

### 1.2 Template

```
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

Entscheidend ist **nur** der CUDA-Toolchain: **CUDA 12.8 mit `nvcc`** (Devel-Image,
kein Runtime-Image). Step 9 baut pytorch3d, gsplat und nvdiffrast aus Quellcode mit
`TORCH_CUDA_ARCH_LIST="12.0"` — CUDA 12.4 und älter kennen `compute_120` nicht und
brechen ab.

Sekundär: **Python 3.11** (das Env ist durchgehend cp311).

> ⚠️ Das Template hat einen bekannten RunPod-Bug
> ([runpod/containers#114](https://github.com/runpod/containers/issues/114)):
> trotz `torch280` im Tag wird **PyTorch 2.4.1 installiert**, weil es keine stabilen
> 2.8.0-cu128-Wheels gibt. **Für uns folgenlos** — Step 4 ersetzt torch ohnehin durch
> 2.7.0+cu128, Step 10 pinnt final. Wir nutzen vom Template nur den CUDA-Toolchain.

Auch für Multi-Arch-Builds bleibt dieses Image richtig: CUDA 12.8's nvcc kann
`sm_86` und `sm_89` ebenfalls targeten.

### 1.3 Storage

| Typ | Größe | Kosten | Begründung |
|-----|-------|--------|------------|
| **Network Volume** | **100 GB** | ~$7/Monat | Conda-Env ~15–20 GB + Checkpoints 13,3 GB + Repos/Builds ~3–5 GB + pip-Cache ~5–10 GB ≈ 40–50 GB belegt, Rest ist Puffer |
| **Container Disk** | **≥40 GB** | nur während Laufzeit | Das Startskript kopiert die 13,3 GB Checkpoints nach `/root/sam3d-checkpoints`, dazu Miniconda unter `/root/miniconda3` |

Der Puffer beim Volume ist kein Luxus: die torch-Pins in Steps 4/5b/8/10 installieren
torch mehrfach neu, und abgebrochene Setup-Läufe hinterlassen halbfertige
Build-Verzeichnisse.

> Network Volumes werden **auch bei gestopptem Pod** berechnet ($0,07/GB/Monat unter 1 TB).
> Ein gemeinsames Volume für den ganzen Cluster statt eins pro Pod.

---

## 2. Erstinstallation

`setup.sh` liegt auf dem Pod unter `/workspace/sam3d-api/setup.sh` und ist die
maßgebliche Quelle. Die folgende Übersicht beschreibt, was die Steps tun — nicht als
Ersatz für das Skript, sondern damit im Fehlerfall klar ist, an welcher Stelle es hakt.

### 2.1 Vorab-Check (spart 25 Minuten)

Direkt nach Pod-Start, **bevor** setup.sh läuft:

```bash
nvcc --version        # muss >= 12.8 sein — sonst abbrechen, falsches Template
nvidia-smi            # GPU + >= 32 GB VRAM bestätigen
du -sh /workspace     # tatsaechliche Belegung gegen das Kontingent
```

Zu `nvidia-smi`: die Zeile `CUDA Version: 13.0` im Header ist die maximal vom **Treiber**
unterstützte Runtime, nicht das installierte Toolkit — maßgeblich ist die Ausgabe von
`nvcc --version`. Abwärtskompatibel, kein Widerspruch.

> ⚠️ **`df -h /workspace` ist hier nutzlos.** Das Volume ist MooseFS
> (`mfs#...runpod.net:9421`) — `df` meldet die Kennzahlen des gesamten RunPod-Clusters
> (Größenordnung Petabyte), nicht euer Kontingent. Für den freien Platz gilt:
> `du -sh /workspace` gegen die in der RunPod-Console provisionierte Volume-Größe rechnen.

### 2.2 Setup starten

```bash
cd /workspace/sam3d-api
source setup.sh
```

**Wichtig:** `source`, nicht `bash` — das Skript setzt `CUDA_HOME`, `PATH` und
aktiviert die conda-Umgebung in der laufenden Shell.

Dauer: **~15–30 Minuten** (Checkpoint-Download + native Builds).

**SSH-Abbruch-Schutz:** Der Lauf ist zu lang, um an einer SSH-Verbindung zu hängen — ein
Verbindungsabbruch würde ihn per SIGHUP killen. setup.sh startet sich deshalb selbst in
einer tmux-Session `setup` neu, sofern es nicht schon in tmux läuft. Bei Verbindungsabbruch
einfach neu einloggen und wieder anhängen:

```bash
tmux attach -t setup
```

Loslösen ohne Abbruch: `Ctrl-b`, dann `d`. Fehlt tmux (`which tmux`), vorher
`apt-get install -y tmux` — das liegt außerhalb des conda-Envs.

Nach Step 3 muss diese Zeile erscheinen:

```
Env active: Python 3.11.x at /workspace/envs/sam3d-objects/bin/python
```

Steht dort ein Pfad unter `/usr/local/` oder Python 3.12, bricht setup.sh mit
`FATAL: conda env is not active` ab. Ohne diesen Guard lief das gesamte Setup still ins
System-Python des Containers — siehe [`setup-fixes.md`](setup-fixes.md) Punkt 12.

### 2.3 Was die Steps tun

| Step | Inhalt |
|------|--------|
| 1 | Repository `sam-3d-objects` klonen bzw. aktualisieren |
| 2 | Miniconda prüfen / nach `/root/miniconda3` installieren |
| 3 | Conda-Env unter `/workspace/envs/sam3d-objects` anlegen (Python 3.11) |
| 4 | **PyTorch 2.7.0+cu128** — Erstinstall, bevor sam3d-objects installiert wird |
| 5 | sam3d-objects + gsplat (kompiliert gegen torch 2.7.0) + kaolin |
| 5b | torch re-pin (sam3d-objects downgradet ggf.) + spconv-cu121 + xformers `--no-deps` |
| 6 | Modell-Checkpoints `facebook/sam-3d-objects` (13,3 GB) |
| 7 | `requirements.txt` + `hf_transfer` + nvdiffrast |
| 8 | xformers `--force-reinstall --no-deps`, torch re-pin, kaolin re-pin |
| 9 | **pytorch3d + gsplat + nvdiffrast aus Quellcode**, CUDA 12.8, `TORCH_CUDA_ARCH_LIST=12.0` |
| 10 | **Absoluter finaler Pin:** torch 2.7.0, numpy 1.26.4, cusparselt 0.6.3 |

Die vier torch-Pins (4, 5b, 8, 10) sind **alle** notwendig. Steps 4/5b/8 sichern, dass
die nativen Builds gegen die *richtige* torch-Version **kompilieren**; Step 10 fixt nur
die final *installierte* Version, nicht die ABI der bereits gebauten `.so`-Dateien.
Details in [`setup-fixes.md`](setup-fixes.md) Punkt 8.

### 2.4 HF-Zugang

Die Checkpoints sind gated — vorab auf
[huggingface.co/facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)
Zugriff beantragen und einloggen:

```bash
huggingface-cli login
```

---

## 3. Verifikation

**Alle folgenden Befehle laufen im Conda-Env** `/workspace/envs/sam3d-objects` — nicht in
einem venv und nicht im System-Python des Containers. Die Shell, in der `source setup.sh`
lief, ist bereits aktiviert. In einer **neuen** Shell zuerst:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs      # Envs liegen auf dem Volume
conda activate /workspace/envs/sam3d-objects

which python    # erwartet: /workspace/envs/sam3d-objects/bin/python
```

Zeigt der Prompt kein Env-Präfix und `which python` nicht diesen Pfad, ist conda in dieser
Shell nicht initialisiert — dann greifen alle folgenden `pip`- und `python`-Aufrufe am Env
vorbei.

Dann die eigentliche Verifikation:

```bash
# torch-Version und GPU-Erkennung
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# erwartet: 2.7.0+cu128 12.8 True

# Compute Capability der Karte
python -c "import torch; print(torch.cuda.get_device_capability())"
# erwartet: (12, 0) auf 5090 / PRO 4500

# Native Builds — die haeufigsten ABI-Bruchstellen
python -c "import kaolin;    print('kaolin OK')"
python -c "import pytorch3d; print('pytorch3d OK')"
python -c "import xformers;  print('xformers OK')"
python -c "import nvdiffrast; print('nvdiffrast OK')"

# sam3d-objects End-to-End importierbar
python -c "import sam3d_objects; print('sam3d-objects OK')"
```

Schlägt einer der Imports mit `undefined symbol: _ZN3c105Error...` fehl, wurde das
Paket gegen eine andere torch-Version gebaut → siehe [`setup-fixes.md`](setup-fixes.md)
Punkt 5.

Optional (blockiert den Worker nicht, siehe `worker-test.md`):

```bash
python -c "import flash_attn; print('flash-attn OK:', flash_attn.__version__)"
```

---

## 4. Täglicher Start

Nach Pod-Neustart — **kein** erneutes setup.sh nötig, solange dasselbe Volume verwendet wird:

```bash
cd /workspace
bash install_conda_start_env_host_api.sh
```

Das Skript aktiviert conda, kopiert die Checkpoints einmalig pro Pod auf die lokale
NVMe (`/root/sam3d-checkpoints`, deutlich schneller als das Volume), setzt
`SAM3D_CHECKPOINT_DIR` und startet uvicorn auf Port 8000.

Falls nur der uvicorn-Prozess gestoppt wurde:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs
conda activate /workspace/envs/sam3d-objects
cd /workspace/sam3d-api
uvicorn api:app --host 0.0.0.0 --port 8000
```

Health-Check:

```bash
curl -s http://localhost:8000/health   # erwartet: "worker_ready": true
```

Vollständiger Funktionstest → [`worker-test.md`](worker-test.md).

---

## 5. Optional: Multi-Arch-Build

Wenn GPU-**Verfügbarkeit** der Engpass ist, lohnt ein einmaliger Multi-Arch-Build.
In Step 9 statt `TORCH_CUDA_ARCH_LIST="12.0"`:

```bash
TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
```

Erzeugt Fatbinaries, die auf Ampere (A40/A6000/A100), Ada (L40S) **und** Blackwell
(5090/PRO 4500) laufen — danach ist jeder Pod-Typ aus Abschnitt 1.1 ohne Rebuild
nutzbar.

Kosten: deutlich längere Build-Zeit und einige GB mehr auf dem Volume.

> ⚠️ **Bisher nicht erprobt** — das aktuelle Env ist reines sm_120. Vor dem Umbau
> testen, nicht blind auf einem produktiven Volume ausführen.

---

## Hinweise

- `/workspace` ist persistenter Storage — Pakete überleben Pod-Neustarts.
- Bei **neuem Volume** muss `source setup.sh` einmalig vollständig laufen.
- `install_conda_start_env_host_api.sh` macht **kein** Paket-Management, nur Start.
- Ein Worker pro GPU. Mehrere gleichzeitige Inferenzen auf derselben Karte erzeugen
  zwangsläufig OOM — Jobs laufen deshalb strikt nacheinander.
- Der CUDA-Pfad wird in setup.sh dynamisch ermittelt (`find /usr/local -name "cuda-1*"`),
  weil er je nach Pod-Image variiert.
