"""
paths.py
--------
Zentrale Pfad-Definition für das gesamte Projekt.

Alle Skripte importieren ihre Eingabe- und Ausgabe-Pfade von hier —
es gibt KEINE verstreuten `SCRIPT_DIR.parent / ...`-Konstruktionen mehr.
Dadurch funktionieren alle Skripte unabhängig vom Arbeitsverzeichnis,
und sämtliche Ergebnisse landen einheitlich unter `results/`.

Struktur (relativ zum Repo-Root):

    data/                       Datensätze (siehe data/README.md)
        id/                     Cityscapes
        ood/                    Lost & Found
        smiyc/dataset_AnomalyTrack/    RoadAnomaly21
        smiyc/dataset_ObstacleTrack/   RoadObstacle21
    cache/                      dinov2_gallery.pt (einmalig erzeugt)
    rba_score_maps/             RbA-Maps von Colab (laf/, smiyc/<Track>/)
    results/
        baseline/               Kapitel-2-Baseline (run_evaluation.py)
        roi_variants/           ROI-Evaluation L&F + score_maps/-Cache
        roi_closing/            Closing-Ablation (road)
        roi_closing_sw/         Closing-Ablation (road+sidewalk)
        smiyc/<Track>/          SMIYC-Ergebnisse + score_maps/ + heatmaps/
        figures/                alle Abbildungs-Skripte
        segformer_iou.csv       Tabelle 8
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent

# ---------------------------------------------------------------------------
# Eingaben
# ---------------------------------------------------------------------------
DATA_DIR        = ROOT / "data"
DATA_CITYSCAPES = DATA_DIR / "id"
DATA_LAF        = DATA_DIR / "ood"
DATA_SMIYC = {
    "RoadAnomaly21":  DATA_DIR / "smiyc" / "dataset_AnomalyTrack",
    "RoadObstacle21": DATA_DIR / "smiyc" / "dataset_ObstacleTrack",
}

# DINOv2-Galerie (einmalig gebaut von dinov2_knn_ood.py, ~2 GB, nicht in git)
CACHE_DIR    = ROOT / "cache"
GALLERY_PATH = CACHE_DIR / "dinov2_gallery.pt"

# RbA-Score-Maps aus Google Colab (siehe colab/README.md)
RBA_UPLOAD_DIR = ROOT / "rba_score_maps"

# PixOOD-Repo als Schwester-Ordner (geklont durch install.ps1 / install.sh)
PIXOOD_DIR = ROOT.parent / "PixOOD"

# ---------------------------------------------------------------------------
# Ausgaben
# ---------------------------------------------------------------------------
RESULTS_DIR        = ROOT / "results"
BASELINE_DIR       = RESULTS_DIR / "baseline"
ROI_VARIANTS_DIR   = RESULTS_DIR / "roi_variants"
SCORE_MAPS_LAF     = ROI_VARIANTS_DIR / "score_maps"
ROI_CLOSING_DIR    = RESULTS_DIR / "roi_closing"
ROI_CLOSING_SW_DIR = RESULTS_DIR / "roi_closing_sw"
SMIYC_RESULTS_DIR  = RESULTS_DIR / "smiyc"
FIGURES_DIR        = RESULTS_DIR / "figures"
