# RunPod Performance-Änderungen

Protokoll aller Änderungen, die primär der Geschwindigkeit dienen — Setup-Zeit,
Download, native Builds und Inferenz-Laufzeit.

`setup.sh` liegt im **`sam3d-api`**-Repo, die übrigen Skripte/Dokumente in
`MixedRealityInteriorArrangement/runpod/`. Für Fehlerdiagnose siehe
[`setup-fixes.md`](setup-fixes.md), für den Ablauf [`setup.md`](setup.md).

## Status-Legende

- ✅ **getestet** — im Betrieb bestätigt
- ⏳ **ungetestet** — implementiert, aber noch nicht in einem vollständigen Lauf verifiziert

---

## Übersicht

| # | Änderung | Wirkt auf | Status |
|---|----------|-----------|--------|
| 1 | hf_transfer vor dem Download + aktiviert | Checkpoint-Download (Step 6) | ⏳ |
| 2 | `--max-workers 1 → 4` | Checkpoint-Download (Step 6) | ⏳ |
| 3 | `MAX_JOBS` dynamisch am RAM | Native Builds (Step 9) | ⏳ |
| 4 | conda-Paket-Cache auf lokale NVMe | Setup (conda create) | ⏳ |
| 5 | Checkpoints auf lokale NVMe kopieren | Inferenz / Worker-Start | ✅ |
| 6 | Persistent Worker (Pipeline einmal laden) | Inferenz pro Request | ✅ |

---

## 1. hf_transfer vor dem Checkpoint-Download

**Datei:** `sam3d-api/setup.sh`, Step 6 (+ Entfernung aus Step 7)

**Vorher:** `pip install hf_transfer` stand in Step 7 — also **nach** dem Download in
Step 6. Damit lief der 13,3-GB-Download ohne Beschleunigung.

**Nachher:** hf_transfer wird vor dem Download installiert und aktiviert:

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Warum:** hf_transfer ist ein Rust-Downloader, der jede Datei in parallele Chunks
aufteilt. Die zwei großen Checkpoints (`ss_generator.ckpt` 6,7 GB, `slat_generator.ckpt`
4,9 GB) machen ~87 % der Nutzlast aus — Chunk-Parallelismus innerhalb einer Datei bringt
hier deutlich mehr als Parallelismus über Dateien hinweg.

**Bezug:** [`setup-fixes.md`](setup-fixes.md) Punkt 7 dokumentiert, warum die Variable
`HF_HUB_ENABLE_HF_TRANSFER=1` überhaupt gesetzt ist (sie war gesetzt, aber das Paket
fehlte). Diese Änderung stellt die richtige Reihenfolge her.

---

## 2. `--max-workers` von 1 auf 4

**Datei:** `sam3d-api/setup.sh`, Step 6

**Warum:** parallelisiert die ~20 kleinen Checkpoint-Dateien. Kleinerer Hebel als #1
(die zwei großen Dateien dominieren), aber gratis und ohne Risiko.

---

## 3. `MAX_JOBS` dynamisch am RAM

**Datei:** `sam3d-api/setup.sh`, Step 9

**Vorher:** kein `MAX_JOBS` gesetzt — die CUDA-Compiles (pytorch3d, gsplat,
diff-gaussian-rasterization, **flash-attn**) liefen mit dem Default, oft zu wenig
parallel. flash-attn ist der zäheste Build.

**Nachher:** `MAX_JOBS` wird aus dem RAM-Budget berechnet, gedeckelt durch `nproc`:

```
MAX_JOBS = min( nproc, floor( RAM_GB × (100 − BUFFER_PCT) / 100 / PER_JOB_GB ) )
BUFFER_PCT = 20        # 20 % RAM freilassen
PER_JOB_GB = 3         # konservativer RAM-Peak pro nvcc-Job (flash-attn)
```

**Container-Fallstrick:** `/proc/meminfo` meldet im Container die **Host-RAM**, nicht das
Pod-Limit. Das Skript liest deshalb zusätzlich das cgroup-Limit
(`/sys/fs/cgroup/memory.max` bzw. `memory.limit_in_bytes`) und nimmt das Minimum. Ohne
das würde der Speicher massiv überschätzt und der cgroup-Killer bräche den Build per OOM
ab.

**Zwei Stellschrauben:**
- `PER_JOB_GB=3` ist eine begründete Schätzung, kein exakter Wert. Bei OOM in Step 9
  (`c++: fatal error: Killed signal`, `ninja: build stopped`) → auf 4–5 erhöhen.
- Die Echo-Zeile `Build parallelism: RAM …GB … MAX_JOBS=…` zeigt beim Lauf den
  berechneten Wert. Ein unplausibel hoher Wert (z.B. 40) deutet darauf hin, dass die
  cgroup-Erkennung nicht griff und Host-RAM erwischt wurde.

---

## 4. conda-Paket-Cache auf lokale NVMe

**Datei:** `sam3d-api/setup.sh`, `CONDA_PKGS_DIRS`

**Vorher:** `CONDA_PKGS_DIRS=/workspace/conda-pkgs` — auf dem MooseFS-Netzwerk-Volume.

**Nachher:** `CONDA_PKGS_DIRS=/root/conda-pkgs` — lokale NVMe.

**Warum:** primär ein Korrektheits-Fix (MooseFS korrumpierte den Cache bei zehntausenden
Kleinstdateien, siehe [`setup-fixes.md`](setup-fixes.md) Punkt 13), aber auch schneller:
Entpacken vieler kleiner Dateien auf lokaler NVMe statt über ein Netzwerk-Dateisystem.
Nachteil: bei neuem Pod wird neu heruntergeladen — irrelevant, da der Cache ohnehin
throwaway ist und nur das Env persistiert.

---

## 5. Checkpoints auf lokale NVMe kopieren

**Datei:** `install_conda_start_env_host_api.sh`

Beim ersten Start pro Pod werden die Checkpoints vom `/workspace`-Volume auf lokale
NVMe (`/root/sam3d-checkpoints`) kopiert; der Worker lädt sie von dort:

```bash
LOCAL_CKPT=/root/sam3d-checkpoints
cp -r /workspace/sam3d-api/sam-3d-objects/checkpoints/. "$LOCAL_CKPT/"
export SAM3D_CHECKPOINT_DIR="$LOCAL_CKPT"
```

**Warum:** der Worker lädt die Pipeline von schneller lokaler NVMe statt vom langsameren
MooseFS-Volume. Einmalige Kopierzeit pro Pod gegen schnelleren Worker-Start.

**Voraussetzung:** Container Disk ≥40 GB (13,3 GB Checkpoints + Miniconda), siehe
[`setup.md`](setup.md) Abschnitt 1.3.

---

## 6. Persistent Worker (Pipeline einmal laden)

**Dateien:** `sam3d-api/worker_3d.py`, `api.py`

Die Pipeline wird **einmal beim API-Start** geladen und bleibt warm, statt bei jedem
Request neu geladen zu werden. Der teure Pipeline-Load (`[Worker] Pipeline loaded in XXs`)
fällt damit nur einmal an.

**Wirkung:** erste Generierung nach API-Start normal schnell, jede weitere deutlich unter
der alten Cold-Start-Zeit (Ziel: < 60 s Gesamtzeit Request→completed).

**Grenzen:** ein Worker pro GPU, Jobs laufen strikt nacheinander (32-GB-VRAM erlaubt keine
parallele Inferenz). Details und Verifikation in [`worker-test.md`](worker-test.md).

---

## Hinweise

- Hebel 1–3 wirken v.a. beim **ersten** Lauf auf einem frischen Volume. Nach einem
  erfolgreichen Lauf sind Checkpoints (persistent) und gebaute Wheels (pip-Cache) gecacht;
  Re-Runs auf demselben Volume überspringen Download und Build-Downloads ohnehin.
- Hebel 1–3 sind noch **ungetestet** — beim ersten Lauf mit der neuen Fassung auf die
  beiden Echo-Zeilen (hf_transfer-Geschwindigkeit in Step 6, `MAX_JOBS` in Step 9) achten.
