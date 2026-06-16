# OoD-ROI-Benchmark — Benchmarking OoD Detection Methods for Semantic Image Segmentation

Code for the Bachelor's thesis **"OoD detection methods for image segmentation"**
(Gerd Schmiemann, Carl von Ossietzky Universität Oldenburg, 2026).

The central research question: **How do evaluation-protocol choices — in particular the
definition of the Region of Interest (ROI) — affect benchmark metrics for pixel-level
out-of-distribution (OoD) detection?**

Four OoD detection methods are compared:

| Method | Backbone | Type |
|---|---|---|
| Energy (+ MSP, Entropy baselines) | SegFormer-B2 (Cityscapes) | Logit-based uncertainty |
| DINOv2-kNN | DINOv2 ViT-B/14 (frozen) | Feature-distance, zero training |
| PixOOD | Official checkpoints | Parametric prototype distance |
| RbA | Mask2Former (Swin-B, 1 dec. layer) | Mask-based outlier rejection |

evaluated under four ROI variants:

| Variant | Definition |
|---|---|
| **A** | Full image |
| **B** | Fixed trapezoid (driving corridor) |
| **C** | Road ROI from the model's own segmentation (trainId 0, optionally +sidewalk) |
| **D** | Negative filtering (exclude high-confidence InD regions, MSP > 0.95) |

on three datasets: **Lost & Found** (test split, 1 096 images), **SMIYC RoadAnomaly21**
(validation, 10 images) and **SMIYC RoadObstacle21** (validation, 30 images).

---

## 1. Repository layout

```
ood-roi-benchmark/
├── README.md                  ← you are here
├── install.ps1 / install.sh   ← one-shot environment setup (Windows / Linux)
├── requirements.txt
├── run_all.py                 ← master pipeline, runs every experiment of the thesis
├── main.py                    ← Chapter-2 baseline entry point
├── configs/eval_config.yaml   ← baseline configuration
├── src/
│   ├── paths.py               ← ALL input/output paths are defined here, nowhere else
│   ├── dataloaders/           Lost & Found / Cityscapes / SMIYC dataset classes
│   ├── models/                SegFormer-B2 loading + forward helpers
│   ├── scoring/               score-map computation & RbA merging (expensive, cached)
│   ├── evaluation/            metric computation: ROI variants, ablations, baseline
│   └── visualization/         all thesis figures
├── data/                      datasets (NOT in git) — see data/README.md
├── cache/                     dinov2_gallery.pt (built once, NOT in git)
├── rba_score_maps/            RbA maps from Colab (laf/, smiyc/<Track>/, NOT in git)
├── scripts/download_score_maps.py   fetch precomputed caches from Zenodo
├── colab/README.md            recipe to (re)compute RbA score maps on Google Colab
└── results/                   ALL outputs land here (NOT in git, see §6)
```

**Path policy:** every script imports its input/output locations from
[`src/paths.py`](src/paths.py). There are no per-script relative paths anymore — all
scripts can be run from any working directory, and all results end up under `results/`.

---

## 2. Installation

### Requirements
- Python 3.11
- NVIDIA GPU with CUDA (CPU works but is impractically slow for score-map computation)
- ~20 GB free disk space for datasets + caches

### Windows (PowerShell)
```powershell
.\install.ps1
```
(If PowerShell blocks the script: `Get-ChildItem -Recurse | Unblock-File` once, then retry.)

### Linux / macOS
```bash
bash install.sh
```

The script creates a virtual environment `.venv`, installs PyTorch 2.2.2 (CUDA 12.1) and
`requirements.txt`, and clones **PixOOD** as a sibling directory (`../PixOOD`). Download
the official PixOOD Cityscapes checkpoints as described in the PixOOD README. SegFormer-B2
and DINOv2 weights download automatically on first use (HuggingFace / torch.hub).

### Activate the environment (required before running anything)

All dependencies live in `.venv`. **Activate it in every new terminal session before
running any script**, otherwise Python won't find the installed packages:

```powershell
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
```
```bash
source .venv/bin/activate           # Linux / macOS
```

Once active, the prompt is prefixed with `(.venv)` and every `python ...` call uses the
correct interpreter. Verify with `python -c "import sys; print(sys.executable)"` — the
path must contain `.venv`.

**In PyCharm:** set the interpreter once under Settings → Project → Python Interpreter →
Add Interpreter → Existing → `.venv\Scripts\python.exe`. The Run button and the built-in
terminal then use `.venv` automatically.

> **Most common error:** `ModuleNotFoundError: No module named 'numpy'` (or torch, …) when
> running `run_all.py` or any script means the `.venv` is **not active** — the system
> Python is being used instead. Activate it as shown above and retry.

> **Note on RbA:** RbA inference requires Detectron2 plus a compiled CUDA kernel
> (MSDeformAttn) and is **not** installed locally — it does not build reliably on
> Windows. Either download the precomputed score maps
> (`python scripts/download_score_maps.py`, recommended) or recompute them on Google
> Colab following [`colab/README.md`](colab/README.md) and place them under
> `rba_score_maps/` (`laf/`, `smiyc/RoadAnomaly21/`, `smiyc/RoadObstacle21/`).

---

## 3. Datasets

Datasets are **not** included. See [`data/README.md`](data/README.md) for download links
(Cityscapes, Lost & Found, SMIYC) and the exact folder structure under `data/`.

---

## 4. Quickstart — reproduce everything

```powershell
.\install.ps1                              # 1. install
.\.venv\Scripts\Activate.ps1               # 2. activate the environment (every new terminal!)
# 3. place datasets under data/ (see data/README.md)
python scripts/download_score_maps.py      # 4. fetch caches (skips ~5-6 h GPU inference, includes RbA)
python run_all.py --skip gallery laf_scores rba_merge smiyc_scores   # 5. evaluate + figures (minutes)
```

Or fully from scratch (computes everything yourself, RbA maps via Colab needed):

```powershell
python run_all.py
```

> ⚠️ **Runtime warning:** Without the precomputed caches the full pipeline takes
> **several hours up to a full day** depending on your GPU — the score maps for the
> 1 096 Lost & Found images alone are ~2–3 h, the DINOv2 gallery + kNN evaluation ~3 h,
> the Chapter-2 baseline ~0.5–1 h. `run_all.py` warns before starting expensive stages.

Useful flags:

```powershell
python run_all.py --list                 # show all stages
python run_all.py --only laf_eval        # run a single stage
python run_all.py --from smiyc_eval      # resume from a stage
python run_all.py --continue-on-error
```

---

## 5. Scripts: inputs → outputs

All scripts can be called from the repo root. Paths below are relative to the root.

### 5.1 Scoring (`src/scoring/`) — expensive, run once, cached

| Script | Input | Output |
|---|---|---|
| `dinov2_knn_ood.py` | `data/id` (train), `data/ood` | `cache/dinov2_gallery.pt` + `results/dinov2_knn/dinov2_knn_results.{csv,tex}` (Table 2: SegFormer baselines + DINOv2-kNN; reads the baseline numbers from `results/baseline/metrics_table.csv`, so run `main.py` first) |
| `compute_score_maps.py` `[--skip_existing --skip_pixood --max_images N]` | `data/ood`, `cache/dinov2_gallery.pt`, `../PixOOD` | `results/roi_variants/score_maps/<stem>.npz` (energy/knn/msp/pixood maps + pred + GT) |
| `compute_score_maps_smiyc.py --track <Track>` | `data/smiyc/...`, gallery, PixOOD | `results/smiyc/<Track>/score_maps/<stem>.npz` |
| `merge_rba_into_score_maps.py` `[--rba_dir … --dry_run]` | `rba_score_maps/laf/<stem>_rba.npz` | adds `rba_map` field to the L&F `.npz` cache |
| `merge_rba_into_score_maps_smiyc.py` `[--rba_dir … --dry_run]` | `rba_score_maps/smiyc/<Track>/` | adds `rba_map` to the SMIYC caches (both tracks) |

### 5.2 Evaluation (`src/evaluation/`) — seconds to minutes, cache-based

| Script | Input | Output |
|---|---|---|
| `run_evaluation.py` (via `python main.py`) | `data/id` (val), `data/ood`, `configs/eval_config.yaml` | `results/baseline/`: `metrics_table.{csv,tex}` (Table 1), `summary_table.*`, `score_distributions.png`, `config.json` |
| `evaluate_roi_variants.py` | L&F score-map cache | `results/roi_variants/`: `roi_results.{csv,tex}` (Table 5), `per_image_auroc.csv` (per-image raw AUROC, variants A & D), `per_image_auroc_stats.{csv,tex}` (Mean/Std/Min/Median per method × variant, Table 6) |
| `evaluate_roi_closing.py` | L&F cache + `data/ood` | `results/roi_closing/closing_results.{csv,tex}` (Table 7) + example figures in `results/figures/roi_closing/` |
| `evaluate_roi_closing_sw.py` | dito | `results/roi_closing_sw/closing_sw_results.{csv,tex}` (Table 7) |
| `evaluate_smiyc_variants.py` | SMIYC caches (both tracks) | `results/smiyc/<Track>/smiyc_results.{csv,tex}`, `per_image_auroc.csv` |
| `measure_segformer_iou.py` | `data/id` (val) | console + `results/segformer_iou.csv` (Table 8) |

### 5.3 Figures (`src/visualization/`)

| Thesis figure | Command (from repo root) | Output |
|---|---|---|
| Ch. 2 qualitative SegFormer figures | `python src/visualization/visualize.py --imgs 02_Hanns_Klemm_Str_44_000002_000180 02_Hanns_Klemm_Str_44_000006_000180` (omit `--imgs` for the first N images) | `results/figures/chapter2/<name>_leftImg8bit.png` |
| Fig. 3 — DINOv2-kNN success case | `python src/visualization/single_image_analysis.py --img "02_Hanns_Klemm_Str_44_000006_000180"` | `results/figures/single_image/single_<name>.png` + appends a row to `single_image_metrics.csv` |
| Fig. 4 — DINOv2-kNN failure case (chalk drawing) | `python src/visualization/single_image_analysis.py --img "04_Maurener_Weg_8_000004_000100"` | dito |
| Fig. 5/6 — ROI-variant examples | `python src/visualization/visualize_roi_variants.py --img "<name>_leftImg8bit"` with `02_Hanns_Klemm_Str_44_000006_000180`, `04_Maurener_Weg_8_000004_000100`, `15_Rechbergstr_Deckenpfronn_000004_000210` | `results/figures/roi_variants/roi_variants_<name>.png` |
| ROI schematic overview | `python src/visualization/make_roi_figure.py --image <img> --label <gt>` | `results/figures/roi_schematic/roi_variants.pdf` |
| SMIYC heatmap grids (Ch. 4) | `python src/visualization/visualize_smiyc_heatmaps.py [--track <Track>] [--imgs …]` | `results/smiyc/<Track>/heatmaps/<stem>_roi.png` — images auto-selected per track: 2 best + 1 median + 2 worst by per-image AUROC |

<!-- TODO(Gerd): verify image names against the final thesis figure list. -->

---

## 6. Output tree (created at runtime)

```
results/
├── baseline/                  Chapter-2 baseline tables + plots (Table 1)
├── dinov2_knn/                Table 2 (SegFormer baselines + DINOv2-kNN)
├── roi_variants/              Table 5 + Table 6 stats + score_maps/ cache (L&F)
├── roi_closing/               Table 7 (road ROI)
├── roi_closing_sw/            Table 7 (road+sidewalk)
├── smiyc/<Track>/             SMIYC tables + score_maps/ + heatmaps/
├── figures/                   chapter2/, single_image/ (+ single_image_metrics.csv),
│                              roi_variants/, roi_closing*/, roi_schematic/
└── segformer_iou.csv          Table 8
```

## 7. Reproducibility

- Global seed **42**, fixed in `run_evaluation.py`; all hyperparameters of each run are
  written to `results/**/config.json`.
- The two-stage cached pipeline (score maps → evaluation) makes every ROI-variant result
  exactly reproducible without re-inference.
- SMIYC GT encoding: `0 = InD`, `1 = OoD`, `255 = ignore`. For the ROI-variant comparison
  the **official SMIYC ROI is deliberately ignored** (`valid_mask` = all ones); GT=255 is
  treated as InD/negative, so the evaluation base is identical across all datasets and
  ROI variants A–D.

## 8. Precomputed artifacts (Zenodo)

The full score-map caches (including the RbA maps computed on Google Colab, Tesla T4)
are published as a Zenodo archive:

> **DOI: 10.5281/zenodo.20722623**

`scripts/download_score_maps.py` downloads and unpacks them into the expected locations.
How the RbA maps were created is documented in [`colab/README.md`](colab/README.md).

## 9. References

- Lost & Found: Pinggera et al., 2016
- SegmentMeIfYouCan: Chan et al., 2021 — RoadAnomaly21 (Zenodo 5270237), RoadObstacle21 (Zenodo 5281633)
- RbA: Nayal et al., ICCV 2023 — https://github.com/NazirNayal8/RbA
- PixOOD: Vojíř et al., 2024 — https://github.com/vojirt/PixOOD
- DINOv2: Oquab et al., 2023
- SegFormer: Xie et al., 2021 — checkpoint `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`
