"""
visualize_smiyc_heatmaps.py
---------------------------
Erzeugt pro ausgewähltem SMIYC-Bild eine 3×2-Abbildung:

    Zeile 1:  Original + GT-Kontur        kNN Heatmap + GT-Kontur
    Zeile 2:  Variante A (Volles Bild)    Variante B (Trapez)
    Zeile 3:  Variante C (Road)           Variante D (Neg. Filter)

Die kNN-Score-Maps und ood_label/pred_class werden aus den gecachten
.npz-Dateien gelesen (results/smiyc/<Track>/score_maps/), die RGB-Bilder
aus data/smiyc/dataset_*/images/. So sind die Heatmaps exakt konsistent
zu den Tabellenzahlen aus evaluate_smiyc_variants.py.

Bildauswahl: automatisch 2 beste + 1 Mitte + 2 schlechteste nach
Per-Image-AUROC (Variante A) aus per_image_auroc.csv (Methode DINOv2 kNN).

Aufruf:
    python visualize_smiyc_heatmaps.py
    python visualize_smiyc_heatmaps.py --track RoadAnomaly21
    python visualize_smiyc_heatmaps.py --imgs validation0000 validation0003

Output:
    results/smiyc/<Track>/heatmaps/<stem>_roi.png
"""

import os, sys, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import DATA_SMIYC, SMIYC_RESULTS_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRACK_IMG_DIRS = {t: root / "images" for t, root in DATA_SMIYC.items()}

# Gleiche Colormap wie single_image_analysis.py
OOD_CMAP = LinearSegmentedColormap.from_list(
    "ood", ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#b2182b"]
)

# ROI-Parameter — identisch zu evaluate_smiyc_variants.py
TRAP_TOP_Y, TRAP_BOT_Y = 0.28, 0.90
TRAP_TL_X,  TRAP_TR_X  = 0.38, 0.62
TRAP_BL_X,  TRAP_BR_X  = 0.05, 0.95
BG_TRAIN_IDS  = {2, 3, 4, 5, 8, 10}
MSP_THRESHOLD = 0.95


def make_trapezoid_mask(H, W):
    mask  = np.zeros((H, W), dtype=bool)
    top_y = int(TRAP_TOP_Y * H)
    bot_y = int(TRAP_BOT_Y * H)
    for y in range(top_y, bot_y + 1):
        t     = (y - top_y) / max(bot_y - top_y, 1)
        left  = int((TRAP_TL_X * (1 - t) + TRAP_BL_X * t) * W)
        right = int((TRAP_TR_X * (1 - t) + TRAP_BR_X * t) * W)
        mask[y, left:right] = True
    return mask


def build_masks(pred_class, msp_map):
    H, W = pred_class.shape
    mask_a = np.ones((H, W), dtype=bool)
    mask_b = make_trapezoid_mask(H, W)
    mask_c = (pred_class == 0)
    bg = np.zeros((H, W), dtype=bool)
    for tid in BG_TRAIN_IDS:
        bg |= ((pred_class == tid) & (msp_map > MSP_THRESHOLD))
    mask_d = ~bg
    return {
        "A: Volles Bild": mask_a,
        "B: Trapez":      mask_b,
        "C: Road":        mask_c,
        "D: Neg. Filter": mask_d,
    }


def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


def find_image(track, stem):
    img_dir = TRACK_IMG_DIRS[track]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def select_images(track, requested):
    """Wähle Bilder: explizit angefragte oder automatisch 2 beste+Mitte+2 schlechteste."""
    if requested:
        return requested

    per_img_csv = SMIYC_RESULTS_DIR / track / "per_image_auroc.csv"
    if not per_img_csv.exists():
        print(f"[{track}] [Warn] {per_img_csv} nicht gefunden — nehme alle Bilder.")
        sm_dir = SMIYC_RESULTS_DIR / track / "score_maps"
        return [p.stem for p in sorted(sm_dir.glob("*.npz"))]

    rows = []
    with open(per_img_csv) as f:
        for r in csv.DictReader(f):
            if r["method"] == "DINOv2 kNN" and r["variant"] == "A: Volles Bild":
                try:
                    rows.append((r["image"], float(r["auroc"])))
                except ValueError:
                    pass  # nan

    rows = [r for r in rows if not np.isnan(r[1])]
    rows.sort(key=lambda x: x[1])  # aufsteigend

    if len(rows) <= 5:
        return [r[0] for r in rows]

    worst2 = [rows[0][0], rows[1][0]]
    best2  = [rows[-1][0], rows[-2][0]]
    mid    = [rows[len(rows) // 2][0]]
    # Reihenfolge: best, best, mitte, schlecht, schlecht
    selection = [best2[0], best2[1], mid[0], worst2[0], worst2[1]]
    # Duplikate vermeiden, Reihenfolge erhalten
    seen, out = set(), []
    for s in selection:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def plot_image(track, stem, save_path):
    sm_path = SMIYC_RESULTS_DIR / track / "score_maps" / f"{stem}.npz"
    if not sm_path.exists():
        print(f"[{track}] [Warn] Score-Map fehlt: {sm_path}")
        return False

    d = np.load(sm_path)
    knn_map    = d["knn_map"].astype(np.float32)
    ood_label  = d["ood_label"].astype(np.int32)
    pred_class = d["pred_class"].astype(np.int32)
    msp_map    = d["msp_map"].astype(np.float32)
    H, W       = ood_label.shape

    img_path = find_image(track, stem)
    if img_path is None:
        print(f"[{track}] [Warn] RGB-Bild nicht gefunden für {stem}")
        return False
    rgb = np.array(Image.open(img_path).convert("RGB"))
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

    masks   = build_masks(pred_class, msp_map)
    knn_norm = norm01(knn_map)
    n_ood   = int(ood_label.sum())
    has_ood = n_ood > 0

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.axis("off")

    # ── Zeile 1: Original + GT,  kNN Heatmap + GT ────────────────────────
    ax_orig, ax_heat = axes[0, 0], axes[0, 1]

    ax_orig.imshow(rgb)
    ax_orig.set_title("(a) Originalbild",
                      fontsize=10, color="black", loc="left", pad=4)

    im = ax_heat.imshow(knn_norm, cmap=OOD_CMAP, vmin=0, vmax=1,
                        interpolation="bilinear")
    if has_ood:
        ax_heat.contour(ood_label, levels=[0.5], colors=["#00ee66"], linewidths=1.8)
    ax_heat.set_title("(b) DINOv2 kNN Heatmap + GT-Kontur",
                      fontsize=10, color="black", loc="left", pad=4)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.030, pad=0.02)
    cbar.set_label("normierter kNN-Abstand", fontsize=7.5, color="black")
    cbar.ax.tick_params(labelsize=7)

    # ── Zeilen 2+3: ROI-Varianten A/B/C/D ────────────────────────────────
    panel_order = ["A: Volles Bild", "B: Trapez", "C: Road", "D: Neg. Filter"]
    panel_axes  = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
    panel_tags  = ["(c)", "(d)", "(e)", "(f)"]

    for tag, vname, ax in zip(panel_tags, panel_order, panel_axes):
        mask = masks[vname]
        # Abgedunkeltes Overlay: ausgeschlossene Bereiche dunkler
        overlay = rgb.astype(float) / 255.0
        overlay[~mask] = overlay[~mask] * 0.25 + 0.10
        ax.imshow(overlay)
        # GT-Kontur drauf (grün), damit man sieht ob OoD drin/draußen ist
        if has_ood:
            ax.contour(ood_label, levels=[0.5], colors=["#00ee66"], linewidths=1.8)

        roi_pct = mask.sum() / (H * W) * 100
        ood_in  = int((ood_label[mask] == 1).sum())
        ret_pct = ood_in / max(n_ood, 1) * 100
        ax.set_title(f"{tag} Variante {vname}  —  ROI {roi_pct:.0f}%, "
                     f"OoD-Ret. {ret_pct:.0f}%",
                     fontsize=10, color="black", loc="left", pad=4)

    # Fußzeile
    foot = (f"{track}  |  {stem}  |  OoD-Pixel: {n_ood:,}  "
            f"({n_ood / (H * W) * 100:.1f}% des Bildes)")
    fig.text(0.5, 0.005, foot, ha="center", va="bottom",
             fontsize=8, color="#444444", style="italic")

    plt.tight_layout(pad=1.2, h_pad=1.6, w_pad=1.0, rect=(0, 0.02, 1, 1))
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → {save_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["RoadAnomaly21", "RoadObstacle21"],
                        default=None, help="Nur ein Track (sonst beide)")
    parser.add_argument("--imgs", nargs="*", default=None,
                        help="Konkrete Bild-Stems (sonst Auto-Auswahl)")
    args = parser.parse_args()

    tracks = [args.track] if args.track else ["RoadAnomaly21", "RoadObstacle21"]

    for track in tracks:
        print(f"\n{'#' * 60}\n#  {track}\n{'#' * 60}")
        out_dir = SMIYC_RESULTS_DIR / track / "heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)

        stems = select_images(track, args.imgs)
        print(f"[{track}] Ausgewählte Bilder: {stems}")

        for stem in stems:
            save_path = out_dir / f"{stem}_roi.png"
            plot_image(track, stem, save_path)

    print(f"\nFertig. Heatmaps in results/smiyc/<Track>/heatmaps/")


if __name__ == "__main__":
    main()
