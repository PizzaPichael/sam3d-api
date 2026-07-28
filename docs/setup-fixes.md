# RunPod Setup – Dokumentierte Fixes

Protokoll der Probleme und Lösungen beim Aufsetzen der SAM2 + sam3d-objects API
auf einem RunPod-Pod mit **NVIDIA RTX PRO 4500 Blackwell (sm_120)**.

---

## Probleme & Lösungen

### 1. `Sam2Processor` nicht importierbar

**Symptom:**
```
ImportError: cannot import name 'Sam2Processor' from 'transformers'
```

**Ursache:** `torchvision` und `torch` waren inkompatible Versionen — `torchvision` hatte einen ABI-Konflikt der beim Import von `transformers.image_utils` (das `torchvision` verwendet) den Fehler auslöste.

**Fix in `setup.sh`:** torchvision wird gemeinsam mit torch als passende Version installiert (Step 4 + Step 8).

---

### 2. Blackwell GPU (sm_120) nicht unterstützt

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
UserWarning: NVIDIA RTX PRO 4500 Blackwell with CUDA capability sm_120 is not compatible
```

**Ursache:** PyTorch 2.4.1 (vorinstalliert) unterstützt maximal sm_90. Blackwell (sm_120) benötigt PyTorch 2.7.0+cu128.

**Fix in `setup.sh`:**
- Step 4: PyTorch 2.7.0+cu128 installieren
- Step 8: torch mit `--force-reinstall --no-deps` als letztes pinnen
- `CUDA_HOME=/usr/local/cuda-12.8` früh setzen

---

### 3. xformers downgradet torch auf 2.5.1+cu121

**Symptom:** Nach dem Setup war torch 2.5.1+cu121 statt 2.7.0+cu128 installiert.

**Ursache:** `pip install xformers` zog torch 2.5.1+cu121 als Dependency rein und überschrieb den 2.7.0-Install.

**Fix in `setup.sh` (Step 8):** xformers zuerst, dann torch mit `--force-reinstall --no-deps`:
```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

---

### 4. xformers ABI-Konflikt (moge/dinov2 schlägt fehl)

**Symptom:**
```
WARNING[XFORMERS]: xFormers was built for PyTorch 2.5.1+cu121
AttributeError: Cannot set attribute 'src' directly.
```

**Ursache:** xformers wurde gegen torch 2.5.1+cu121 gebaut. moge/dinov2 importiert `SwiGLU` von xformers — schlägt fehl und verhindert den sam3d-objects Start.

**Fix in `setup.sh` (Step 8):**
```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

---

### 5. kaolin ABI-Konflikt nach torch-Wechsel

**Symptom:**
```
⚠ Sam-3d-objects import failed: kaolin/_C.so: undefined symbol: _ZN3c105ErrorC2E...
```

**Ursache:** kaolin wurde gegen eine andere torch-Version kompiliert. Jedes Mal wenn torch gewechselt wird, muss kaolin passend neu installiert werden.

**Fix in `setup.sh` (Step 8):**
```bash
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html \
    --force-reinstall --no-deps
```

---

### 6. pytorch3d kann nicht für sm_120 kompiliert werden

**Symptom:**
```
nvcc fatal : Unsupported gpu architecture 'compute_120'
```

**Ursache:** Die conda-Umgebung enthält einen eigenen nvcc (`/workspace/envs/sam3d-objects/bin/nvcc`) mit CUDA 12.1, der sm_120 nicht kennt. Das System hat CUDA 12.8 unter `/usr/local/cuda-12.8/`.

**Fix in `setup.sh`:** `CUDA_HOME` direkt nach `conda activate` auf System-CUDA setzen:
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
```

pytorch3d in Step 9 mit der Zielarchitektur aus Quellcode bauen:
```bash
TORCH_CUDA_ARCH_LIST="12.0" pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
    --no-build-isolation
```

---

### 7. `hf_transfer` fehlt

**Symptom:**
```
ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available
```

**Ursache:** Die Umgebungsvariable `HF_HUB_ENABLE_HF_TRANSFER=1` war gesetzt, aber das Paket nicht installiert.

**Fix in `setup.sh` (Step 7):**
```bash
pip install hf_transfer
```

---

### 8. gsplat/nvdiffrast/pytorch3d ziehen torch 2.11.0 nach Step 8

**Symptom:** Nach abgeschlossenem Setup ist torch 2.11.0+cu128 installiert statt 2.7.0, obwohl Step 8 gepinnt hatte.

**Ursache:** Step 9 baut gsplat, nvdiffrast und pytorch3d aus Quellcode — ohne `--no-deps`. Diese Pakete haben torch als Dependency und ziehen die aktuell neueste Version (2.11.0). Das überschreibt den torch-Pin aus Step 8, **nach** dem die nativen Builds bereits gegen 2.11.0 kompiliert wurden.

**Fix in `setup.sh` (Step 10):** Absolut letzter Step pinnt torch, numpy und cusparselt nochmals — nach allen nativen Builds:
```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12==0.6.3 --force-reinstall --no-deps
```

**Wichtig:** Die torch-Pins in Step 4, 5b und 8 sind trotzdem notwendig — sie stellen sicher dass gsplat, pytorch3d etc. gegen die richtige torch-Version **kompiliert** werden. Step 10 fixt nur den final installierten torch, nicht die ABI der `.so`-Dateien.

---

### 9. xformers ohne `--no-deps` upgradet torch auf 2.11.0

**Symptom:** `pip install xformers --force-reinstall` zieht torch 2.11.0 als Dependency.

**Ursache:** xformers 0.0.35 definiert `torch>=2.10` als Requirement. Ohne `--no-deps` installiert pip automatisch die neueste kompatible torch-Version (2.11.0).

**Fix:** Alle xformers-Installs in setup.sh bekommen `--no-deps`:
```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

---

### 10. nvidia-cusparselt-cu12 Versionskonflikt (libcusparseLt.so.0 fehlt)

**Symptom:**
```
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
```

**Ursache:** xformers-Install upgradet `nvidia-cusparselt-cu12` von 0.6.3 auf 0.7.1. Die Library-Datei änderte dabei ihren soname — torch 2.7.0 sucht `libcusparseLt.so.0` (aus 0.6.3), findet sie aber nicht mehr.

**Fix:**
```bash
pip install nvidia-cusparselt-cu12==0.6.3 --force-reinstall --no-deps
```

In setup.sh durch Step 10 abgedeckt.

---

### 11. CUDA-Pfad variiert je nach Pod-Image

**Symptom:** `error: [Errno 2] No such file or directory: '/usr/local/cuda-12.8/bin/nvcc'`

**Ursache:** Verschiedene RunPod-Images haben CUDA unter unterschiedlichen Pfaden (z.B. `cuda-12.4`, `cuda-12.8`). Ein hardkodierter Pfad schlägt auf anderen Images fehl.

**Fix in `setup.sh`:** CUDA-Pfad dynamisch ermitteln:
```bash
CUDA_HOME_CANDIDATE=$(find /usr/local -maxdepth 1 -name "cuda-1*" -type d | sort -V | tail -1)
if [ -n "$CUDA_HOME_CANDIDATE" ] && [ -f "$CUDA_HOME_CANDIDATE/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA_HOME_CANDIDATE"
    export PATH=$CUDA_HOME/bin:$PATH
fi
```

**Wichtig:** Das Pod-Image muss CUDA 12.8 **mit nvcc** enthalten. CUDA 12.4 und älter unterstützen sm_120 (Blackwell) nicht — pytorch3d und andere native Builds schlagen dann fehl. Empfohlenes Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

---

### 12. Leeres Conda-Env — gesamtes Setup laeuft ins System-Python

**Symptom:** Setup laeuft scheinbar durch, danach:

```
ModuleNotFoundError: No module named 'sam3d_objects'
WARNING[XFORMERS]: xFormers was built for PyTorch 2.10.0+cu128 (you have 2.7.0+cu128)
    Python 3.10.19 (you have 3.12.3)
AttributeError: module 'numpy' has no attribute 'long'
```

Auffaellig sind die Pfade in den Tracebacks: `/usr/local/lib/python3.12/dist-packages/`
statt `/workspace/envs/sam3d-objects/lib/python3.11/site-packages/`. `dist-packages` ist
Debian-Konvention fuer System-Pakete — Conda nutzt ausschliesslich `site-packages`.

**Diagnose:**

```bash
which python              # /usr/local/bin/python      <- falsch
python --version          # Python 3.12.3              <- falsch, Env ist 3.11
echo $CONDA_PREFIX        # /workspace/envs/sam3d-objects   <- sieht richtig aus!
ls /workspace/envs/sam3d-objects/bin/    # No such file or directory
```

**Ursache:** Step 3 prueft mit `if [ -d "$ENV_PATH" ]`, ob das Env existiert. Ein
**leeres Verzeichnis** besteht diesen Test — der Create-Zweig wird uebersprungen und das
Env nie wirklich angelegt. `conda activate` auf einen existierenden Prefix-Pfad schlaegt
nicht fehl: es setzt `CONDA_PREFIX`, aendert den Prompt auf `(sam3d-objects)` und ist
fertig. Mangels `bin/python` faellt danach jeder `python`- und `pip`-Aufruf still auf
`/usr/local/bin/python` (System 3.12) durch.

Folge: **alle Installationen landen im Container-Python** — auf der Container Disk, weg
beim naechsten Pod-Neustart. Die Folgefehler erklaeren sich daraus:

- `sam3d_objects` fehlt → editable Install scheiterte an `auto-gptq` (Build-Isolation) und wurde nie nachgeholt
- xformers gegen torch 2.10.0 gebaut → das ist die Version aus dem RunPod-Basisimage; die `--force-reinstall`-Kette lief am Env vorbei
- `np.long` → Step 10 pinnt `numpy==1.26.4`, das scipy des Templates braucht `numpy>=2.0`

**Tritt nur bei leerem Volume auf.** Auf einem gefuellten Volume existiert das Env real,
Step 3 meldet `Environment exists. Updating...` und alles funktioniert.

**Fix (manuell, Env neu anlegen):**

```bash
conda deactivate
rm -rf /workspace/envs/sam3d-objects

source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs
conda create -p /workspace/envs/sam3d-objects python=3.11 -y
conda activate /workspace/envs/sam3d-objects

which python      # MUSS /workspace/envs/sam3d-objects/bin/python sein
python --version  # MUSS 3.11.x sein
```

**Fix in `setup.sh` (Step 3):** auf das Python-Binary pruefen statt aufs Verzeichnis, und
nach der Aktivierung hart abbrechen statt still weiterzulaufen:

```bash
if [ -x "$ENV_PATH/bin/python" ]; then
    echo "Environment exists. Updating..."
else
    rm -rf "$ENV_PATH"
    conda create -p "$ENV_PATH" python=3.11 -y
fi
conda activate "$ENV_PATH"

# Guard: ohne Env-Python sofort abbrechen, sonst laeuft das gesamte Setup ins System-Python
if [ "$(command -v python)" != "$ENV_PATH/bin/python" ]; then
    echo "FATAL: Env nicht aktiv — ist: $(command -v python)"
    return 1   # setup.sh wird ge-source-t, daher return statt exit
fi
```

Derselbe Guard steckt in `install_conda_start_env_host_api.sh` (dort `exit 1`, da das
Skript mit `bash` gestartet wird).

**Aufraeumen des System-Pythons ist nicht noetig:** Conda-Envs erben `dist-packages`
nicht. Sobald das Env ein eigenes Python hat, ist das System-Python nicht mehr im
Suchpfad. Nur den Platz auf der Container Disk im Auge behalten (`df -h /`).

---

### 13. `environments/default.yml` scheitert auf frischem Volume (ClobberError + qt-main)

**Symptom:** `conda env create -f environments/default.yml` bricht ab mit hunderten:

```
CondaVerificationError: The package for qt-main located at /workspace/conda-pkgs/qt-main-5.15.8-hc9dc06e_21
appears to be corrupted. The path 'translations/qtxmlpatterns_zh_CN.qm' cannot be found.

ClobberError: This transaction has incompatible packages due to a shared path.
  packages: cuda-nsight, cuda-cuobjdump, cuda-cuxxfilt, cuda-nvprune,
            cuda-sanitizer-api, cuda-nvprof, cuda-nvvp
  path: 'LICENSE'
```

**Ursache (beide Fehler, eine Wurzel):** `default.yml` ist ein vollstaendiger conda-Export
von Meta und pinnt das komplette **CUDA-12.1-Toolkit** inklusive der grafischen Profiler
`cuda-nvvp` und `cuda-nsight`. Diese GUIs ziehen **qt-main** (Qt5) nach — ein Paket mit
zehntausenden winzigen Uebersetzungsdateien.

- **ClobberError:** die sieben cuda-Tool-Pakete legen jeweils ein eigenes `$PREFIX/LICENSE` ab.
- **CondaVerificationError:** qt-main ueber MooseFS (`/workspace` ist ein Netzwerk-Volume)
  zu entpacken schlaegt fehl bzw. hinterlaesst unvollstaendige Pakete im Cache.

`conda config --set path_conflict clobber` und `--solver=libmamba` beheben das **nicht**.

**Der eigentliche Punkt:** setup.sh ueberschreibt praktisch alles aus `default.yml` ohnehin:

| aus default.yml | ueberschrieben durch |
|---|---|
| `cuda-nvcc=12.1.105` + CUDA-12.1-Toolkit | `CUDA_HOME` auf System-CUDA 12.8 (Punkt 6 — conda's nvcc kennt sm_120 nicht) |
| `libcublas`/`libcufft`/`libcusolver` (12.1) | pip `nvidia-*`-Wheels in cu128 mit torch |
| `qt-main`, `gst-*`, `xorg-*`, `pulseaudio` | nie benutzt (headless Pod) |
| `python=3.11.0` | **wird gebraucht** |
| `gcc`/`gxx`/`c-compiler`/`cxx-compiler` | **wird gebraucht** (Step 9) |

**Fix in `setup.sh` (Step 3):** `default.yml` nicht verwenden, Env minimal anlegen:

```bash
conda create -p "$ENV_PATH" -c conda-forge -y \
    python=3.11 pip setuptools wheel c-compiler cxx-compiler
```

Kein CUDA-Toolkit, kein Qt, kein X11 → beide Fehlerklassen entfallen. Dauert ~2 Minuten
statt ~20 und spart mehrere GB auf dem Volume.

**Zusaetzlich:** `CONDA_PKGS_DIRS` von `/workspace/conda-pkgs` auf `/root/conda-pkgs`
(lokale NVMe) umgestellt. Der Paket-Cache muss nicht persistent sein, und MooseFS ist fuer
zehntausende Kleinstdateien der falsche Ort. Bei bereits korruptem Cache vorher:

```bash
conda clean --all -y
rm -rf /workspace/conda-pkgs/*
```

---

### 14. sdist-Deps scheitern an Build-Isolation (auto-gptq, nvidia-pyindex, flash-attn, pytorch3d)

**Symptom:** `pip install -e '.[dev]'` bricht bei einer Dependency ab mit
`ModuleNotFoundError: No module named 'torch'` bzw. `'pip'`.

**Ursache:** Diese vier Deps importieren in ihrem `setup.py` torch oder pip **zur
Build-Zeit**. pip baut sie standardmäßig in einer isolierten Umgebung ohne torch/pip.

**Aber:** sam3d-objects **selbst** baut mit hatchling + `hatch-requirements-txt` — dieses
Backend liefert nur die Build-**Isolation**. `--no-build-isolation` auf das ganze `.[dev]`
scheitert daher an `Cannot import 'hatchling.build'` / `Unknown metadata hook: requirements_txt`.

**Fix in `setup.sh` (Step 4):** die vier Übeltäter **einzeln vorab** mit
`--no-build-isolation` installieren (dann von pip als erfüllt gesehen), das eigentliche
`.[dev]` **mit** Isolation lassen:

```bash
BUILD_CUDA_EXT=0 pip install auto-gptq==0.7.1 --no-build-isolation
pip install nvidia-pyindex --no-build-isolation
pip install flash-attn==2.8.3 --no-build-isolation
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install \
    "pytorch3d @ git+...@75ebeeaea..." --no-build-isolation --no-deps
pip install -e '.[dev]'   # Isolation AN → hatchling wird auto-bereitgestellt
```

pytorch3d und flash-attn sind auf **bestimmte Commits/Versionen** gepinnt — installiert man
eine andere, baut `.[dev]` die gepinnte Version neu (und scheitert wieder). Deshalb genau
den gepinnten pytorch3d-Commit `75ebeeaea` und `flash-attn==2.8.3` vorinstallieren.

---

### 15. flash-attn `Invalid cross-device link` (EXDEV) über MooseFS

**Symptom:**
```
Guessing wheel URL: .../flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-...whl
error: [Errno 18] Invalid cross-device link
```

**Ursache:** flash-attn 2.8.3 hat ein **vorgebautes Wheel** für genau dieses Env (kein
Compile nötig). Sein `setup.py` lädt es nach `TMPDIR` (Default `/tmp`, lokale Disk) und
verschiebt es per `rename()` in den pip-Cache auf `/workspace` (MooseFS). `rename()` kann
nicht über Dateisystemgrenzen → `EXDEV`.

**Fix:** `TMPDIR` und `PIP_CACHE_DIR` auf **dasselbe** Dateisystem legen. setup.sh setzt
beide auf `/workspace` (Step 2), also kein Problem — der Fehler trat nur in manuellen
Sessions ohne diese Exports auf.

---

### 16. xformers ohne Versionspin lädt keine C++/CUDA-Extensions

**Symptom:**
```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions.
    xFormers was built for PyTorch 2.10.0+cu128 ... (you have 2.7.0+cu128)
```

**Ursache:** `pip install xformers --no-deps` holt die **neueste** Version (gebaut für
torch 2.10/2.11). Deren Extensions laden gegen torch 2.7 nicht — und moge/dinov2 braucht
xformers SwiGLU (siehe Punkt 4), also startet sam3d-objects nicht.

**Fix in `setup.sh` (Step 5b + 8):** auf die zu torch 2.7.0 passende Version pinnen:

```bash
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

---

### 17. Native Builds ziehen torch 2.11 nach → ABI-Bruch (via PIP_CONSTRAINT gelöst)

**Symptom (latent):** nach Step 9 `undefined symbol` beim Import von gsplat/pytorch3d.

**Ursache:** Die Builds in Step 9 nutzen `--force-reinstall` **ohne** `--no-deps`. Das
zieht torch als Dependency in der neuesten Version (2.11) nach, kompiliert die Extension
dagegen — dann pinnt Step 10 torch zurück auf 2.7.0 und die `.so` sind ABI-inkompatibel.
(Beschrieben in Punkt 8, dort nur teilweise gelöst.)

**Fix in `setup.sh` (Step 9):** eine pip-Constraints-Datei fixiert torch **nur für Step 9**
(nicht früher — `.[dev]` in Step 4 muss torch bewusst downgraden dürfen):

```bash
cat > /workspace/tmp/torch-constraint.txt <<EOF
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
numpy==1.26.4
EOF
export PIP_CONSTRAINT=/workspace/tmp/torch-constraint.txt
# ... Step 9 Builds ...
unset PIP_CONSTRAINT   # nicht in die ge-source-te Shell lecken lassen
```

Damit können die Builds ihre Nicht-torch-Deps installieren, aber torch bleibt bei 2.7.0.

---

### 18. Verifikations-Block am Ende (Step 11)

Step 11 importiert alle ABI-sensiblen Pakete + `sam3d_objects`. Ein kaputter torch-Pin oder
ein gegen die falsche torch-Version gebautes Paket fällt so **am Ende des Setups** auf,
nicht erst beim API-Start. Gibt `=== VERIFICATION PASSED/FAILED ===` aus (nicht-fatal, da
`set -e` aus ist).

---

### 19. `omegaconf`/`hydra` fehlen — 3D-Worker crasht beim ersten Job

**Symptom:** API startet sauber (`Uvicorn running on ...`), aber der 3D-Worker stirbt sofort:
```
[Worker] GPU available: True
File ".../sam-3d-objects/notebook/inference.py", line 12, in <module>
    from omegaconf import OmegaConf, DictConfig, ListConfig
ModuleNotFoundError: No module named 'omegaconf'
[API] 3D worker exited (returncode=None)
```
Nach dem Fix für `omegaconf` derselbe Fehler nochmal, diesmal mit `hydra`:
```
ModuleNotFoundError: No module named 'hydra'
```

**Ursache:** `inference.py` importiert **beide** direkt (bestätigt per Blick in die
Original-Datei: `numpy, PIL, omegaconf, hydra, torch, utils3d, seaborn, matplotlib, kaolin,
pytorch3d, sam3d_objects, gradio`). Normalerweise zieht `pip install -e '.[dev]'` (Step 4)
sie transitiv über `hydra-core` mit — aber jede Repair-Session, die `sam3d_objects`
stattdessen mit `-e '.' --no-deps` neu registriert (siehe Punkt 12/14 bzw.
`resume-schekclist-24_07.md`), überspringt beide komplett. Fällt erst beim ersten echten
`/generate-3d`-Call auf, nicht beim Import-Verifikationsschritt (`import sam3d_objects`
allein reicht nicht, `inference.py` wird erst beim Worker-Start geladen).

**Fix:**
```bash
pip install omegaconf hydra-core
```

**Fix in `setup.sh` (Step 5):** jetzt explizit installiert, unabhängig davon ob `.[dev]` es
transitiv mitbringt:
```bash
pip install omegaconf hydra-core
```

---

### 20. `no kernel image` aus xformers' Hopper-Kerneln auf sm_120

**Symptom:** Setup und Pipeline-Load laufen sauber durch, `[API] ✓ 3D worker ready` kommt —
aber der erste `/generate-3d`-Call killt den Worker mitten in `pipe.run()`:

```
sam3d_objects.pipeline.inference_pipeline:get_condition_input:633 - Running condition embedder ...
CUDA error (/__w/xformers/xformers/third_party/flash-attention/hopper/flash_fwd_launch_template.h:175):
no kernel image is available for execution on the device
[API] 3D worker exited (returncode=None)
```

**Ursache:** Der Condition-Embedder ist DINOv2. Dessen Attention nutzt
`xformers.ops.memory_efficient_attention`, und xformers 0.0.30 dispatcht das in seine
mitgelieferten **FlashAttention-3-Kernel**. Die sind ausschliesslich fuer **sm_90/Hopper**
gebaut — auf sm_120/Blackwell existiert kein passendes Kernel-Image.

Neu bauen hilft nicht: FA3 ist architekturbedingt Hopper-only. Eine andere xformers-Version
auch nicht (siehe Punkt 16 — nur 0.0.30 passt zu torch 2.7.0).

**Fix:** DINOv2 diesen Pfad gar nicht nehmen lassen. `dinov2/layers/attention.py` und
`swiglu_ffn.py` werten eine eigene Variable aus:

```python
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
```

Also:

```bash
export XFORMERS_DISABLED=1
```

Beide Module fallen dann auf ihre reinen PyTorch-Implementierungen zurueck. Steht in
`install_conda_start_env_host_api.sh` und im Verifikationsblock von `setup.sh` Step 11.

**Nicht verwechseln mit Punkt 4/16:** dort ging es darum, dass xformers seine Extensions
ueberhaupt *laedt* (Import-Zeit). Hier laedt alles korrekt, nur das ausgefuehrte Kernel
passt nicht zur GPU. `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1` bleibt deshalb zusaetzlich
gesetzt.

**Kosten:** DINOv2 rechnet Attention in Standard-PyTorch statt memory-efficient. Bei ViT-L
auf 518 px vernachlaessigbar gegen die 12 Diffusion-Steps.

---

### 21. Laufzeit-Pakete fehlen — Worker stirbt erst beim ersten Job

**Symptom:** Kette von `ModuleNotFoundError` beim Worker-Start, jeweils ein Paket pro
Neustart: `loguru`, danach `timm`, danach `open3d`, ... (vorher schon `omegaconf`/`hydra`,
Punkt 19).

**Ursache:** `setup.sh` Step 5 installierte die Inferenz-Deps handverlesen.
`requirements.inference.txt` enthaelt nur vier Pakete (kaolin, gsplat, seaborn, gradio) —
alles Weitere zieht der Pipeline-Code direkt, ohne dass es dort deklariert waere.

**Die Paket-Metadaten sind als Quelle unbrauchbar:** `pip check` meldet ~77 fehlende
Pakete, weil `hatch-requirements-txt` die grosse `requirements.txt` in die Metadaten
uebernimmt — inklusive komplettem Trainings-/Dev-Stack (`wandb`, `tensorboard`, `jupyter`,
`sagemaker`, `bpy`, `bitsandbytes`, `lightning`, ...). Davon braucht der Inferenz-Pfad fast
nichts. Diese Meldungen ignorieren.

**Ermittlung der echten Liste:** `notebook/inference.py` so lange importieren, bis kein
`ModuleNotFoundError` mehr kommt. `CONDA_PREFIX` muss dabei gesetzt sein — `inference.py`
Zeile 5 liest sie unbedingt und wirft sonst `KeyError`, was wie „keine fehlenden Module"
aussieht.

```bash
export CONDA_PREFIX=/root/sam3d-env-local LIDRA_SKIP_INIT=true XFORMERS_DISABLED=1
export PIP_CONSTRAINT=/root/torch-constraint.txt   # sonst zieht ein Dep torch 2.11 nach
for i in $(seq 1 30); do out=$(python -c "import sys; sys.path.insert(0,'/workspace/sam3d-api/sam-3d-objects/notebook'); import inference" 2>&1); m=$(echo "$out" | grep -oP "No module named '\K[^']+" | head -1); if [ -z "$m" ]; then echo "=== ENDE ==="; echo "$out" | tail -5; break; fi; echo ">>> fehlt: $m"; pip install -q "${m//_/-}"; done
```

**Ergebnis (Stand 28.07.2026), jetzt fest in `setup.sh` Step 5:**

```bash
pip install loguru timm open3d optree astor easydict lightning xatlas pyvista pymeshfix igraph imageio
```

`gsplat` tauchte in derselben Schleife auf und wurde dabei versehentlich von PyPI
installiert statt aus dem gepinnten Commit — beim Nachziehen unbedingt die git-Variante aus
Step 5/9 verwenden, sonst laeuft eine nicht fuer sm_120 gebaute Version.

**Versions-Drift:** Die Pakete kommen in aktuellen Versionen (timm 1.0.28, open3d 0.19.0,
lightning 2.6.5), Meta pinnt aeltere (0.9.16 / 0.18.0 / 2.3.3). Mit den neuen laeuft ein
vollstaendiger `/generate-3d` durch (verifiziert). Bei `TypeError`/`AttributeError` aus
einem dieser Pakete gezielt auf den Meta-Pin zurueckgehen.

**Fix in `setup.sh` (Step 11):** die Verifikation importiert jetzt `notebook/inference.py`
statt nur `sam3d_objects` — genau der Pfad, den `worker_3d.py:132` nimmt. Damit faellt so
ein fehlendes Paket am Ende des Setups auf statt beim ersten `/generate-3d`.

---

## Finale Reihenfolge in setup.sh (kritische Steps)

```
Step 4  → torch 2.7.0+cu128 (Erstinstall, bevor sam3d-objects installiert wird)
Step 5  → sam3d-objects + gsplat (kompiliert gegen torch 2.7.0) + kaolin + omegaconf + hydra-core
Step 5b → torch re-pin (sam3d-objects hat ggf. downgegradet) + spconv-cu121 + xformers --no-deps
Step 6  → Modell-Checkpoints (facebook/sam-3d-objects)
Step 7  → requirements.txt + hf_transfer + nvdiffrast
Step 8  → xformers --force-reinstall --no-deps, torch re-pin, kaolin re-pin
           (stellt sicher dass Step 9 Builds gegen torch 2.7.0 kompilieren)
Step 9  → pytorch3d + gsplat + nvdiffrast aus Quellcode mit CUDA 12.8 + TORCH_CUDA_ARCH_LIST=12.0
Step 10 → Absoluter finaler Pin: torch 2.7.0, numpy 1.26.4, cusparselt 0.6.3
```

---

## Hinweise

- `/workspace` ist persistenter Storage — Pakete müssen nach Pod-Neustart **nicht** neu installiert werden, solange dasselbe Volume verwendet wird.
- Bei einem **neuen Pod** (neues Volume) muss `source setup.sh` einmalig vollständig ausgeführt werden. Das dauert ~15-30 Minuten (Checkpoint-Download + native Builds).
- `install_conda_start_env_host_api.sh` ist nur für den täglichen Start zuständig (conda aktivieren + uvicorn starten) — kein Paket-Management.
- sam3d-objects erfordert `torchaudio==2.5.1+cu121` und `xformers==0.0.28.post3` als Dependencies — bekannte Konflikte, die durch die torch-Pins überschrieben werden. Die pip-Warnings sind erwartet und nicht kritisch.
- Die torch-Pins in Step 4, 5b, 8 und 10 sind alle notwendig: jeder sichert die korrekte torch-Version für die nativen Builds im jeweils nächsten Step.
