"""
dinov2_knn_ood.py
-----------------
DINOv2 kNN OOD Detection — Proof of Concept
============================================
Thesis: Frozen DINOv2 features implicitly encode semantic novelty,
enabling competitive OOD detection without any task-specific training.

Expects datasets.py in the same directory (ood-segmentation-benchmark/evaluation/).

Setup:
    pip install torch torchvision scikit-learn tqdm Pillow numpy

Usage:
    python dinov2_knn_ood.py

Smoke-test (fast, ~5 min):
    Set GALLERY_MAX_IMAGES = 50 in CONFIG below.
Full run (~30 min on RTX 3090 with 500 gallery images):
    Set GALLERY_MAX_IMAGES = 500.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from dataloaders.laf_datasets import CityscapesInDDataset, LostAndFoundOoDDataset
from paths import DATA_CITYSCAPES, DATA_LAF, GALLERY_PATH, RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — only edit this block
# ─────────────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT    = str(DATA_CITYSCAPES)   # contains leftImg8bit-normal/ and gtFine/
LAF_ROOT           = str(DATA_LAF)          # contains leftImg8bit-Objekte/ and gtCoarse/

DINOV2_MODEL       = "dinov2_vitb14"  # dinov2_vits14 (faster) | dinov2_vitl14 (better)
GALLERY_MAX_IMAGES = 500              # how many Cityscapes train images to index
KNN_K              = 10               # k nearest neighbours
BATCH_SIZE         = 8               # for gallery extraction (reduce if OOM)
GALLERY_SAVE_PATH  = str(GALLERY_PATH)      # cache/dinov2_gallery.pt (see paths.py)
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv2 input: must be divisible by patch_size=14
# 518 = 37 × 14  →  37×37 = 1369 patch tokens per image
DINO_INPUT_SIZE    = 518
NUM_PATCHES_1D     = DINO_INPUT_SIZE // 14   # 37
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_dinov2():
    print(f"[DINOv2] Loading {DINOV2_MODEL} ...")
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL)
    model.eval().to(DEVICE)
    print(f"[DINOv2] Embed dim: {model.embed_dim}  |  device: {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GALLERY  (Cityscapes train → frozen DINOv2 patch features)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_gallery(model):
    """
    Extract DINOv2 patch tokens from Cityscapes train images.

    We pass size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE) directly into
    CityscapesInDDataset so images are already the right resolution
    and ImageNet-normalized before they reach the model.

    Returns:
        gallery : (N_total_patches, D) float16 CPU tensor
    """
    if os.path.exists(GALLERY_SAVE_PATH):
        print(f"[Gallery] Loading cached gallery from {GALLERY_SAVE_PATH}")
        return torch.load(GALLERY_SAVE_PATH, map_location="cpu")

    # CityscapesInDDataset returns dicts: {"image": (3,H,W), "ood_label": ..., ...}
    full_ds = CityscapesInDDataset(
        root  = CITYSCAPES_ROOT,
        split = "train",
        size  = (DINO_INPUT_SIZE, DINO_INPUT_SIZE),
    )

    if GALLERY_MAX_IMAGES and GALLERY_MAX_IMAGES < len(full_ds):
        gallery_ds = Subset(full_ds, list(range(GALLERY_MAX_IMAGES)))
        print(f"[Gallery] Using {GALLERY_MAX_IMAGES}/{len(full_ds)} Cityscapes train images")
    else:
        gallery_ds = full_ds

    loader = DataLoader(gallery_ds, batch_size=BATCH_SIZE, num_workers=4,
                        pin_memory=True, shuffle=False)

    all_features = []
    for batch in tqdm(loader, desc="Extracting gallery features"):
        imgs = batch["image"].to(DEVICE)          # (B, 3, 518, 518)

        # get_intermediate_layers(n=1) → list of 1 tensor: (B, N_patches, D)
        out = model.get_intermediate_layers(imgs, n=1, return_class_token=False)[0]
        # flatten batch and spatial dims → (B * N_patches, D)
        feats = out.reshape(-1, out.shape[-1])
        all_features.append(feats.half().cpu())

    gallery = torch.cat(all_features, dim=0)    # (N_total, D)
    print(f"[Gallery] Built: {gallery.shape[0]:,} patches  |  dim={gallery.shape[1]}")
    torch.save(gallery, GALLERY_SAVE_PATH)
    print(f"[Gallery] Saved to {GALLERY_SAVE_PATH}")
    return gallery


# ─────────────────────────────────────────────────────────────────────────────
# kNN ANOMALY SCORE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def knn_anomaly_score(query_feats: torch.Tensor,
                      gallery: torch.Tensor,
                      k: int = KNN_K,
                      chunk_size: int = 2048) -> torch.Tensor:
    """
    Mean cosine distance to k nearest gallery patches per query patch.
    Higher score = more anomalous.

    Args:
        query_feats : (N_q, D)  float32, on DEVICE
        gallery     : (N_g, D)  float16, on CPU  (loaded to DEVICE here)
    Returns:
        scores      : (N_q,)    float32, on CPU
    """
    gallery_f = gallery.float().to(DEVICE)

    q_norm = F.normalize(query_feats, dim=-1)
    g_norm = F.normalize(gallery_f,   dim=-1)

    scores = []
    for i in range(0, q_norm.shape[0], chunk_size):
        q_chunk = q_norm[i : i + chunk_size]        # (C, D)
        sim     = q_chunk @ g_norm.T                # (C, N_g)  cosine similarity
        dist    = 1.0 - sim                         # cosine distance ∈ [0, 2]
        topk, _ = torch.topk(dist, k, dim=-1, largest=False)
        scores.append(topk.mean(dim=-1).cpu())

    del gallery_f
    torch.cuda.empty_cache()
    return torch.cat(scores, dim=0)                 # (N_q,)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  (Lost & Found)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, gallery) -> dict:
    """
    For each LAF test image:
      1. Resize image tensor to DINO_INPUT_SIZE for feature extraction.
      2. Compute kNN anomaly score per patch.
      3. Upsample score map back to original image resolution.
      4. Apply valid_mask from LostAndFoundOoDDataset (excludes labelId=255).
      5. Collect pixel-level scores & OoD ground-truth labels.

    Returns dict with AUROC, AP, FPR95.
    """
    # size=None → images and labels stay at original resolution (1024×2048).
    # valid_mask and ood_label from datasets.py are already pixel-accurate.
    # OoD label mapping is handled entirely by _laf_ood_mask() in datasets.py:
    #   labelId >= 2, excl. {31,33,34,36,37,38,39} → OoD=1
    #   labelId == 1 (free road) or non-hazards     → OoD=0
    #   labelId == 255                               → excluded by valid_mask
    laf_ds = LostAndFoundOoDDataset(
        root           = LAF_ROOT,
        split          = "test",
        size           = None,
        min_ood_pixels = 100,
    )

    all_scores = []
    all_labels = []

    for sample in tqdm(laf_ds, desc="Evaluating on Lost & Found"):
        img        = sample["image"]       # (3, H_orig, W_orig) — already ImageNet-normalized
        ood_label  = sample["ood_label"]   # (H_orig, W_orig)  uint8  {0=InD, 1=OoD}
        valid_mask = sample["valid_mask"]  # (H_orig, W_orig)  uint8  {0=ignore, 1=valid}

        H_orig, W_orig = img.shape[1], img.shape[2]

        # ── Resize image to DINOv2 input size ─────────────────────────────
        img_input = F.interpolate(
            img.unsqueeze(0), size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
            mode="bilinear", align_corners=False
        ).to(DEVICE)                                          # (1, 3, 518, 518)

        # ── DINOv2 forward pass ────────────────────────────────────────────
        out = model.get_intermediate_layers(
            img_input, n=1, return_class_token=False
        )[0]                                                  # (1, 1369, D)
        patch_feats = out.squeeze(0)                          # (1369, D)

        # ── kNN anomaly score per patch ────────────────────────────────────
        patch_scores = knn_anomaly_score(patch_feats, gallery)  # (1369,)

        # ── Upsample score map → original resolution ───────────────────────
        score_map = F.interpolate(
            patch_scores.reshape(1, 1, NUM_PATCHES_1D, NUM_PATCHES_1D).float(),
            size=(H_orig, W_orig),
            mode="bilinear", align_corners=False
        ).squeeze().numpy()                                   # (H_orig, W_orig)

        # ── Mask out ignore pixels (labelId=255 in LAF) ────────────────────
        valid = valid_mask.astype(bool)
        all_scores.append(score_map[valid])
        all_labels.append(ood_label[valid])

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    n_ood = int(all_labels.sum())
    n_tot = len(all_labels)
    print(f"\n[Eval] Pixels: {n_tot:,}  |  OoD: {n_ood:,}  "
          f"({100 * n_ood / n_tot:.3f}%)  |  ratio 1:{n_tot // max(n_ood, 1)}")

    auroc = roc_auc_score(all_labels, all_scores)
    ap    = average_precision_score(all_labels, all_scores)
    fpr95 = _fpr_at_tpr(all_labels, all_scores, tpr_target=0.95)

    return {"AUROC": auroc, "AP": ap, "FPR95": fpr95}


def _fpr_at_tpr(labels, scores, tpr_target=0.95):
    """False Positive Rate at tpr_target True Positive Rate."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_target)
    return float(fpr[min(idx, len(fpr) - 1)])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  DINOv2 kNN OOD Detection — Proof of Concept")
    print(f"  Model: {DINOV2_MODEL}  |  k={KNN_K}  |  "
          f"gallery={GALLERY_MAX_IMAGES} imgs")
    print("=" * 58)

    model   = load_dinov2()
    gallery = build_gallery(model)
    results = evaluate(model, gallery)

    print("\n" + "=" * 48)
    print(f"  RESULTS — DINOv2 kNN  (k={KNN_K})")
    print("=" * 48)
    print(f"  AUROC  :  {results['AUROC']:.4f}")
    print(f"  AP     :  {results['AP']:.4f}")
    print(f"  FPR95  :  {results['FPR95']:.4f}")
    print("=" * 48)
    print("\n  Your DeepLab baseline (same LAF test set):")
    print("  ┌───────────┬────────┬────────┬────────┐")
    print("  │  Score    │ AUROC  │   AP   │ FPR95  │")
    print("  ├───────────┼────────┼────────┼────────┤")
    print("  │  MSP      │ 0.8374 │ 0.0055 │ 0.4502 │")
    print("  │  Entropy  │ 0.8434 │ 0.0066 │ 0.4483 │")
    print("  │  Energy   │ 0.8499 │ 0.0113 │ 0.5752 │")
    print("  └───────────┴────────┴────────┴────────┘")
    print("\n  DINOv2 used zero task-specific training.")
    print("  Any improvement motivates the full benchmark in Ch. 4/5.\n")

    # ------------------------------------------------------------------
    # Ergebnisdateien (Tabelle 2: SegFormer-Baselines + DINOv2 kNN)
    # Die SegFormer-Werte stammen aus results/baseline/metrics_table.csv,
    # falls vorhanden; sonst werden die in der Arbeit berichteten Werte genutzt.
    # ------------------------------------------------------------------
    import csv as _csv
    out_dir = RESULTS_DIR / "dinov2_knn"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = []
    baseline_csv = RESULTS_DIR / "baseline" / "metrics_table.csv"
    if baseline_csv.exists():
        with open(baseline_csv, newline="") as bf:
            for r in _csv.DictReader(bf):
                baseline_rows.append(("SegFormer-B2", r["Score"],
                                      float(r["AUROC"]), float(r["AP"]), float(r["FPR95"])))
    else:
        print(f"[Warn] {baseline_csv} nicht gefunden — verwende Baseline-Werte aus der Arbeit.")
        baseline_rows = [
            ("SegFormer-B2", "MSP",     0.8374, 0.0055, 0.4502),
            ("SegFormer-B2", "Entropy", 0.8434, 0.0066, 0.4483),
            ("SegFormer-B2", "Energy",  0.8499, 0.0113, 0.5752),
        ]

    rows = baseline_rows + [
        ("DINOv2 ViT-B/14", f"kNN (k={KNN_K})",
         results["AUROC"], results["AP"], results["FPR95"])
    ]

    csv_path = out_dir / "dinov2_knn_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["Modell", "Score", "AUROC", "AP", "FPR95"])
        for model_name, score, auroc, ap, fpr in rows:
            w.writerow([model_name, score, f"{auroc:.4f}", f"{ap:.4f}", f"{fpr:.4f}"])
    print(f"[Results] Saved: {csv_path}")

    tex_path = out_dir / "dinov2_knn_results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by dinov2_knn_ood.py (Tabelle 2)\n")
        f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        f.write("Modell & Score & AUROC $\\uparrow$ & AP $\\uparrow$ & FPR@95 $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for i, (model_name, score, auroc, ap, fpr) in enumerate(rows):
            # Trennlinie vor der DINOv2-Zeile (letzte Zeile)
            if i == len(rows) - 1:
                f.write("\\midrule\n")
            f.write(f"{model_name} & {score} & {auroc:.4f} & {ap:.4f} & {fpr:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[Results] Saved: {tex_path}")


if __name__ == "__main__":
    main()
