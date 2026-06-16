"""
single_image_analysis.py
------------------------
Analysiert ein einzelnes LAF-Bild mit DINOv2 kNN.
Layout: 2×2 Grid (wie visualize.py), Metriken als Zeile darunter.

    (a) Originalbild          (b) kNN Score Map
    (c) Heatmap Overlay       (d) Ground Truth

Usage:
    python single_image_analysis.py --img "04_Maurener_Weg_8_000004_000100"
    python single_image_analysis.py --idx 42
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from dataloaders.laf_datasets import LostAndFoundOoDDataset
from paths import DATA_LAF, GALLERY_PATH as _GALLERY_PATH, FIGURES_DIR

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LAF_ROOT        = str(DATA_LAF)
GALLERY_PATH    = str(_GALLERY_PATH)
DINOV2_MODEL    = "dinov2_vitb14"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
KNN_K           = 10
DINO_INPUT_SIZE = 518
NUM_PATCHES_1D  = DINO_INPUT_SIZE // 14   # 37
OUTPUT_DIR      = str(FIGURES_DIR / "single_image")

# Gleiche Colormap wie visualize.py: blau (sicher) → rot (anomal)
OOD_CMAP = LinearSegmentedColormap.from_list(
    "ood", ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#b2182b"]
)
# ─────────────────────────────────────────────────────────────────────────────


def fpr_at_tpr(labels, scores, tpr_target=0.95):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = min(np.searchsorted(tpr, tpr_target), len(fpr) - 1)
    return float(fpr[idx]), float(thresholds[idx])


@torch.no_grad()
def knn_score(query_feats, gallery, k=KNN_K, chunk=2048):
    gf = gallery.float().to(DEVICE)
    q  = F.normalize(query_feats, dim=-1)
    g  = F.normalize(gf, dim=-1)
    out = []
    for i in range(0, q.shape[0], chunk):
        dist = 1.0 - (q[i:i+chunk] @ g.T)
        topk, _ = torch.topk(dist, k, dim=-1, largest=False)
        out.append(topk.mean(dim=-1).cpu())
    del gf
    torch.cuda.empty_cache()
    return torch.cat(out)


def score_image(model, gallery, img_tensor):
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    inp = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
        mode="bilinear", align_corners=False
    ).to(DEVICE)
    out = model.get_intermediate_layers(inp, n=1, return_class_token=False)[0]
    scores = knn_score(out.squeeze(0), gallery)
    score_map = F.interpolate(
        scores.reshape(1, 1, NUM_PATCHES_1D, NUM_PATCHES_1D).float(),
        size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    return score_map


def denormalize(t):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = t.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def compute_metrics(score_map, ood_label, valid_mask):
    valid  = valid_mask.astype(bool)
    scores = score_map[valid].ravel()
    labels = ood_label[valid].ravel()
    n_ood  = int(labels.sum())
    n_ind  = int((labels == 0).sum())
    n_tot  = len(labels)

    if n_ood == 0:
        return None, "Keine OoD-Pixel in diesem Bild (nach valid_mask)"

    auroc        = roc_auc_score(labels, scores)
    ap           = average_precision_score(labels, scores)
    fpr95, thr95 = fpr_at_tpr(labels, scores)
    ood_s        = scores[labels == 1]
    ind_s        = scores[labels == 0]

    return {
        "AUROC":                        auroc,
        "AP":                           ap,
        "FPR95":                        fpr95,
        "Threshold@95":                 thr95,
        "n_OoD":                        n_ood,
        "n_InD":                        n_ind,
        "Imbalance":                    f"1:{n_tot // max(n_ood, 1)}",
        "OoD score — mean":             float(ood_s.mean()),
        "OoD score — max":              float(ood_s.max()),
        "InD score — mean":             float(ind_s.mean()),
        "InD score — max":              float(ind_s.max()),
        "Gap (OoD mean − InD mean)":    float(ood_s.mean() - ind_s.mean()),
    }, None


def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


def plot(img_tensor, score_map, ood_label, valid_mask, metrics, img_name, save_path):
    rgb   = denormalize(img_tensor)
    H, W  = rgb.shape[:2]
    valid = valid_mask.astype(bool)

    # Maskiertes Score-Display (ignore → NaN)
    score_display        = np.full(score_map.shape, np.nan)
    score_display[valid] = score_map[valid]
    score_norm           = norm01(score_map)
    score_display_norm   = np.full(score_map.shape, np.nan)
    score_display_norm[valid] = score_norm[valid]

    # Ground-Truth-Bild: InD=dunkelgrau, OoD=rot, ignore=schwarz
    gt = np.zeros((*ood_label.shape, 3))
    gt[valid & (ood_label == 0)] = [0.15, 0.15, 0.15]
    gt[valid & (ood_label == 1)] = [1.0,  0.2,  0.1]

    has_ood = (ood_label is not None) and int(ood_label.sum()) > 0

    # ── Layout: 2×2 wie visualize.py ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.axis("off")

    ax_orig, ax_score, ax_overlay, ax_gt = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # ── Panel A: Originalbild ─────────────────────────────────────────────
    ax_orig.imshow(rgb)
    ax_orig.set_title("(a) Originalbild", fontsize=10, color="black",
                      loc="left", pad=4)

    # ── Panel B: kNN Score Map ────────────────────────────────────────────
    im_score = ax_score.imshow(score_display_norm, cmap=OOD_CMAP,
                               vmin=0, vmax=1, interpolation="bilinear")
    ax_score.set_title("(b) kNN Score Map  (37×37 Patch-Token, upsampled)",
                       fontsize=10, color="black", loc="left", pad=4)

    # Max-Score annotieren
    if valid.any():
        max_idx        = np.unravel_index(
            np.nanargmax(score_display_norm), score_display_norm.shape)
        max_y, max_x   = int(max_idx[0]), int(max_idx[1])
        text_x = W * 0.55 if max_x < W * 0.5 else W * 0.02
        text_y = H * 0.04
        ax_score.annotate(
            f"max = {score_map[max_y, max_x]:.3f}",
            xy=(max_x, max_y), xytext=(text_x, text_y),
            fontsize=7.5, color="black", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888888", linewidth=0.8, alpha=0.90),
            zorder=6
        )

    cbar = fig.colorbar(im_score, ax=ax_score, fraction=0.030, pad=0.02)
    cbar.set_label("normierter kNN-Abstand", fontsize=7.5, color="black")
    cbar.ax.tick_params(labelsize=7, color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    # ── Panel C: Heatmap Overlay ──────────────────────────────────────────
    ax_overlay.imshow(rgb, alpha=0.30)
    im_ov = ax_overlay.imshow(score_display_norm, cmap=OOD_CMAP,
                               vmin=0, vmax=1, alpha=0.75,
                               interpolation="bilinear")
    ax_overlay.set_title("(c) Heatmap Overlay  (grün = GT-Kontur)",
                          fontsize=10, color="black", loc="left", pad=4)

    # GT-Objekte mit Bounding Box + Annotation wie in visualize.py
    if has_ood:
        labeled, n_obj = ndi.label(ood_label)
        corners = [
            (W * 0.02, H * 0.04), (W * 0.55, H * 0.04),
            (W * 0.02, H * 0.72), (W * 0.55, H * 0.72),
        ]
        for obj_idx in range(1, n_obj + 1):
            obj_mask = (labeled == obj_idx)
            if obj_mask.sum() < 1:
                continue
            coords        = np.argwhere(obj_mask)
            y_min, x_min  = coords.min(axis=0)
            y_max, x_max  = coords.max(axis=0)
            cy, cx        = int((y_min + y_max) / 2), int((x_min + x_max) / 2)
            pad           = max(6, int(min(H, W) * 0.018))
            rx = max(0, x_min - pad)
            ry = max(0, y_min - pad)
            rw = min(W - rx, (x_max - x_min) + 2 * pad)
            rh = min(H - ry, (y_max - y_min) + 2 * pad)

            rect = mpatches.Rectangle(
                (rx, ry), rw, rh,
                linewidth=2.0, edgecolor="#00bb66", facecolor="none",
                linestyle="-", zorder=5
            )
            ax_overlay.add_patch(rect)

            mean_score = float(score_map[obj_mask].mean())
            tx, ty     = corners[(obj_idx - 1) % len(corners)]
            ax_overlay.annotate(
                f"GT OoD {obj_idx}  Score={mean_score:.3f}",
                xy=(cx, cy), xytext=(tx, ty),
                fontsize=7, color="#007744", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#007744", lw=1.2,
                                connectionstyle="arc3,rad=0.15"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#007744", linewidth=1.0, alpha=0.92),
                zorder=6
            )

    cbar2 = fig.colorbar(im_ov, ax=ax_overlay, fraction=0.030, pad=0.02)
    cbar2.set_label("normierter kNN-Abstand", fontsize=7.5, color="black")
    cbar2.ax.tick_params(labelsize=7, color="black")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="black")

    # ── Panel D: Ground Truth ─────────────────────────────────────────────
    ax_gt.imshow(gt)
    ax_gt.set_title("(d) Ground-Truth-Maske  (rot = OoD, grau = InD)",
                    fontsize=10, color="black", loc="left", pad=4)
    if has_ood:
        ax_gt.contour(ood_label, levels=[0.5], colors=["#00bb66"],
                      linewidths=1.2)

    # ── Metriken als Fußzeile (wie stats-Zeile in visualize.py) ──────────
    if metrics:
        line1 = (f"AUROC: {metrics['AUROC']:.4f}  |  "
                 f"AP: {metrics['AP']:.4f}  |  "
                 f"FPR@95 %: {metrics['FPR95']:.4f}  |  "
                 f"Gap: {metrics['Gap (OoD mean − InD mean)']:+.4f}  |  "
                 f"OoD-Pixel: {metrics['n_OoD']:,}")
        line2 = (f"OoD Score — Mean: {metrics['OoD score — mean']:.4f}  "
                 f"Max: {metrics['OoD score — max']:.4f}  |  "
                 f"InD Score — Mean: {metrics['InD score — mean']:.4f}  "
                 f"Max: {metrics['InD score — max']:.4f}  |  "
                 f"Imbalance: {metrics['Imbalance']}")
        stats = line1 + "\n" + line2
    else:
        stats = "Keine OoD-Pixel — keine Metriken berechenbar."

    fig.text(0.5, -0.01, stats, ha="center", va="top",
             fontsize=7.5, color="#444444", style="italic")

    plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → Gespeichert: {save_path}")


def print_metrics(metrics, img_name):
    print(f"\n{'='*55}")
    print(f"  Einzel-Bild Analyse: {img_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.6f}")
        else:
            print(f"  {k:<35} {v}")
    print(f"{'='*55}")
    print(f"\n  Globale Vergleichswerte (alle 1096 Bilder):")
    print(f"  {'AUROC':<35} 0.9535")
    print(f"  {'AP':<35} 0.1296")
    print(f"  {'FPR95':<35} 0.2576")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", type=str,
                       help="Bildname-Stamm, z.B. '04_Maurener_Weg_8_000004_000100'")
    group.add_argument("--idx", type=int, help="Dataset-Index (0-basiert)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[DINOv2] Lade {DINOV2_MODEL} ...")
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL)
    model.eval().to(DEVICE)

    print(f"[Gallery] Lade {GALLERY_PATH} ...")
    gallery = torch.load(GALLERY_PATH, map_location="cpu")

    laf_ds = LostAndFoundOoDDataset(
        root=LAF_ROOT, split="test", size=None, min_ood_pixels=0
    )

    if args.img is not None:
        stem = args.img
        idx  = next(
            (i for i, (p, _) in enumerate(laf_ds.samples) if stem in p.name),
            None
        )
        if idx is None:
            print(f"[Fehler] Bild '{stem}' nicht gefunden. Verfügbare Beispiele:")
            for p, _ in laf_ds.samples[:5]:
                print(f"  {p.stem.replace('_leftImg8bit', '')}")
            sys.exit(1)
    else:
        idx = args.idx
        if idx >= len(laf_ds):
            print(f"[Fehler] Index {idx} >= Dataset-Größe {len(laf_ds)}")
            sys.exit(1)

    sample     = laf_ds[idx]
    img_t      = sample["image"]
    ood_label  = sample["ood_label"]
    valid_mask = sample["valid_mask"]
    img_name   = os.path.basename(sample["path"]).replace("_leftImg8bit.png", "")

    print(f"[Info] Bild:       {img_name}")
    print(f"[Info] Auflösung:  {img_t.shape[1]}×{img_t.shape[2]}")
    print(f"[Info] OoD-Pixel:  {int(ood_label.sum()):,}")

    with torch.no_grad():
        score_map = score_image(model, gallery, img_t)

    metrics, err = compute_metrics(score_map, ood_label, valid_mask)
    if err:
        print(f"[Warnung] {err}")
        metrics = None

    save_path = os.path.join(OUTPUT_DIR, f"single_{img_name[:40]}.png")
    plot(img_t, score_map, ood_label, valid_mask, metrics, img_name, save_path)

    if metrics:
        print_metrics(metrics, img_name)

        # Metriken in eine kumulative CSV schreiben (eine Zeile pro Bild).
        # Mehrfachaufrufe fuer verschiedene Bilder ergaenzen die Tabelle.
        import csv as _csv
        csv_path = os.path.join(OUTPUT_DIR, "single_image_metrics.csv")
        fieldnames = ["image", "AUROC", "AP", "FPR95",
                      "Gap (OoD mean − InD mean)", "n_OoD", "n_InD", "Imbalance",
                      "OoD score — mean", "OoD score — max",
                      "InD score — mean", "InD score — max"]
        row = {"image": img_name}
        for k in fieldnames[1:]:
            v = metrics.get(k)
            row[k] = round(v, 4) if isinstance(v, float) else v

        existing = []
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as f:
                existing = [r for r in _csv.DictReader(f) if r["image"] != img_name]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in existing:
                w.writerow(r)
            w.writerow(row)
        print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
