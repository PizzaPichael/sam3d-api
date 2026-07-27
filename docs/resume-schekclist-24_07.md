# Resume-Checkliste — 24.07.2026

Stand beim Pausieren des SAM2 + sam3d-objects Setups auf dem RunPod-Pod
(RTX PRO 4500 Blackwell, sm_120). Diese Datei sagt: **was schon steht, wie du das Env
wieder aktivierst, welche Schritte offen sind.**

Siehe auch: [`setup-fixes.md`](setup-fixes.md) (alle Fehler + Fixes),
[`setup.md`](setup.md) (Gesamtablauf), [`performance-changes.md`](performance-changes.md).

---

## Was bereits auf dem Volume steht (überlebt Pod-Neustart)

Alles Teure liegt auf `/workspace` (Network Volume) und ist persistent:

- ✅ conda-Env `/workspace/envs/sam3d-objects` (Python 3.11.15, gcc 12.4)
- ✅ torch 2.7.0+cu128
- ✅ kaolin, pytorch3d, nvdiffrast, gsplat, diff-gaussian-rasterization
- ✅ flash-attn 2.8.3 (vorgebautes Wheel)
- ✅ Checkpoints (13,3 GB) unter `sam-3d-objects/checkpoints/hf`
- ✅ pip-Cache `/workspace/pip-cache`

**Nicht auf dem Volume:** `miniconda` liegt unter `/root/miniconda3` (Container-Disk).
Bei „Terminate" + neuem Pod ist es weg — kein Datenverlust, nur Neuinstall des
Aktivierers nötig (siehe Schritt 1).

---

## Was NOCH offen ist

| # | Offen | Status |
|---|-------|--------|
| 1 | `sam3d_objects` registrieren | ❌ Install scheitert an pytorch3d/nvidia-pyindex Rebuild unter Isolation |
| 2 | xformers ABI-Fix (`==0.0.30` für torch 2.7) | ❌ aktuell für torch 2.10 gebaut → Extensions laden nicht |
| 3 | Laufzeit-Test (Worker-Start, Inferenz) | ❌ nie gelaufen |
| 4 | Backup nach erfolgreichem Env | ⏳ erst wenn 1+2 erledigt |

---

## Schritt 1 — conda + Env aktivieren

```bash
# Nur nötig, falls 'conda' fehlt (nach Terminate + neuem Pod):
bash /workspace/sam3d-api/resume.sh   # installiert miniconda neu, falls weg
#   (dieses Skript startet auch die API — mit Ctrl-C abbrechen, sobald das Env steht,
#    falls du erst reparieren willst)

# Env + Variablen setzen:
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/workspace/envs
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu121"
export LIDRA_SKIP_INIT=true   # sam3d_objects/__init__.py sonst ImportError (init.py fehlt in dieser Distribution, gewollt)
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1   # xformers 0.0.30 verlangt flash-attn 2.7.1-2.7.4, installiert ist 2.8.3 (siehe Schritt 4)
conda activate /workspace/envs/sam3d-objects

# CUDA für etwaige Builds:
export CUDA_HOME=$(find /usr/local -maxdepth 1 -name "cuda-1*" -type d | sort -V | tail -1)
export PATH=$CUDA_HOME/bin:$PATH
```

Prüfen, dass das Env steht:

```bash
command -v python                                    # /workspace/envs/sam3d-objects/bin/python
python -c "import torch; print(torch.__version__)"   # 2.7.0+cu128
```

Zeigt `python` auf `/usr/local/...` statt aufs Env → conda nicht aktiv, nochmal Schritt 1.

---

## Schritt 2 — sam3d_objects registrieren (offener Fehler)

Der Install scheitert, weil sam3d-objects einen bestimmten **pytorch3d-Commit**
(`75ebeeaea`) und `nvidia-pyindex` pinnt, die unter Build-Isolation neu gebaut werden und
an `No module named 'torch'/'pip'` scheitern. Funktional ist aber alles installiert.

**Pragmatischer Abschluss** — Deps nicht neu bauen, nur das Paket registrieren:

```bash
cd /workspace/sam3d-api/sam-3d-objects
pip install appdirs
pip install nvidia-pyindex --no-build-isolation      # der offene Vorab-Install
pip install -e '.' --no-deps                          # sam3d_objects ohne Dep-Rebuild
python -c "import sam3d_objects; print('sam3d OK')"
```

**Falls `import sam3d_objects` an einem fehlenden Modul scheitert** (z.B. `moge`,
`utils3d`), dieses einzeln nachinstallieren und erneut prüfen:

```bash
pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" --no-build-isolation
pip install "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
```

---

## Schritt 3 — Re-Pin (falls `.[dev]`/Deps torch gedowngradet haben)

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12==0.6.3 --force-reinstall --no-deps

python -c "import torch; print(torch.__version__)"   # muss wieder 2.7.0+cu128 sein
```

---

## Schritt 4 — xformers ABI-Fix

Aktuell installiertes xformers ist für torch 2.10 gebaut → C++/CUDA-Extensions laden
nicht → moge/dinov2 (SwiGLU) startet nicht. Auf die zu torch 2.7 passende Version pinnen:

```bash
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 --no-deps
python -c "import xformers.ops; print('xformers ext OK')"   # darf KEINE C++-Warnung mehr werfen
```

**Bekannter Folgefehler:** xformers 0.0.30 pinnt zusätzlich flash-attn `>=2.7.1,<=2.7.4`,
installiert ist aber `2.8.3` (vorgebautes Wheel für sm_120/Blackwell, siehe Schritt 1) →
`ImportError: Requires Flash-Attention version >=2.7.1,<=2.7.4 but got 2.8.3.`
(`xformers/ops/fmha/flash.py`). Umgangen über `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`
(bereits in Schritt 1 exportiert) — das überspringt nur den Versions-Check, die
flash-attn-Bindings (`flash_attn_cuda`/`flash_attn_gpu`) werden trotzdem geladen.

**Korrektheits-Check (empfohlen, da der Versions-Check ja nur umgangen wird):**
Da 2.8.3 außerhalb des von xformers getesteten Bereichs liegt, könnte die C-API sich
geändert haben — das würde sich nicht als Crash zeigen, sondern als leise falsches
Ergebnis. Kurzer Numerik-Vergleich gegen eine Referenz-Attention, mit erzwungenem
Flash-Backend:

```bash
python -c "
import torch
from xformers.ops import fmha, memory_efficient_attention

B, M, H, K = 2, 128, 8, 64
q = torch.randn(B, M, H, K, device='cuda', dtype=torch.float16)
k = torch.randn(B, M, H, K, device='cuda', dtype=torch.float16)
v = torch.randn(B, M, H, K, device='cuda', dtype=torch.float16)

out = memory_efficient_attention(q, k, v, op=(fmha.flash.FwOp, None))

qf, kf, vf = q.float(), k.float(), v.float()
attn = torch.einsum('bmhd,bnhd->bhmn', qf, kf) / (K ** 0.5)
ref = torch.einsum('bhmn,bnhd->bmhd', attn.softmax(dim=-1), vf)

diff = (out.float() - ref).abs().max().item()
print('max diff:', diff)
assert diff < 1e-2, 'flash-attn 2.8.3 API scheint mit xformers 0.0.30 nicht kompatibel zu sein'
print('flash backend numerically OK')
"
```

`op=fmha.flash.FwOp` erzwingt das Flash-Backend (sonst könnte xformers still auf ein
anderes Backend ausweichen und der Test würde nichts über flash-attn 2.8.3 aussagen).
`max diff` im fp16-üblichen Bereich (<1e-2) → API-kompatibel, sicher weiterverwenden.
Deutlich höher oder Crash → flash-attn auf eine Version im erlaubten Bereich (2.7.1-2.7.4)
downgraden, sofern dafür ein sm_120-kompatibles Wheel existiert.

---

## Schritt 5 — Voll-Verifikation

```bash
python -c "
import torch; print('torch', torch.__version__, torch.cuda.is_available())
import numpy; print('numpy', numpy.__version__)
import kaolin, pytorch3d, nvdiffrast; print('native OK')
import flash_attn; print('flash-attn', flash_attn.__version__)
import xformers.ops; print('xformers OK')
import sam3d_objects; print('sam3d OK')
"
```

Alles grün → Env vollständig.

---

## Schritt 6 — Backup (ERST jetzt, nicht vorher)

```bash
tar czvf /workspace/sam3d-backup-24_07.tar.gz -C / workspace/envs/sam3d-objects workspace/sam3d-api
# danach OFF-Volume sichern (RunPod-Egress ist kostenlos) — NICHT nur auf demselben Volume lassen
```

⚠️ Backup ist **architektur-gebunden** (sm_120): restaurierbar nur auf 5090 / PRO 4500,
gleichem Pfad, CUDA 12.8, Python 3.11. Auf anderer GPU-Arch → neu bauen.

---

## Schritt 7 — Laufzeit-Test

Erst danach der eigentliche Funktionstest → [`worker-test.md`](worker-test.md):
Worker-Start (Pipeline-Load), Health-Check, eine Generierung.

---

## Offene Skript-Vorkehrungen (für den nächsten frischen Lauf)

In `sam3d-api/setup.sh` noch einzubauen (unabhängig von dieser manuellen Reparatur):

- xformers-Pin `==0.0.30` an beiden Stellen (Step 5b + Step 8)
- pytorch3d-Commit + nvidia-pyindex Vorab-Install (analog auto-gptq/flash-attn)
- Constraint-Datei gegen torch-Downgrade durch `--force-reinstall` in Step 9
- Verifikations-Block am Ende (Schritt 5 oben)
