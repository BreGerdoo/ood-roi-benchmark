"""
visualize_adaptive_hull.py
--------------------------
Vergleichs-Visualisierung der drei ROI-Ansaetze nebeneinander:

    [ B: Festes Trapez ]  [ C: Road+SW ]  [ E: Adaptive Huelle ]

Jede ROI wird als halbtransparentes Overlay auf das Originalbild gelegt,
die OoD-Ground-Truth als gruene Kontur eingezeichnet. So wird sichtbar:
  - B folgt der Strasse nicht (festes Trapez, kann Kurven verfehlen)
  - C hat Loecher dort, wo Objekte auf der Strasse liegen
  - E folgt der Strasse UND schliesst die Loecher

Basis ist fest Road+Sidewalk (trainId 0+1), die Margins fest auf 0.08 / 0.05
gesetzt -- konsistent zu evaluate_roi_adaptive_hull.py.

Aendert nichts an bestehenden Skripten. Ausgabe nach
results/figures/adaptive_hull/.

Aufruf (aus dem Repo-Root):
    python src/visualization/visualize_adaptive_hull.py --imgs \
        02_Hanns_Klemm_Str_44_000006_000180 \
        04_Maurener_Weg_8_000004_000100 \
        04_Maurener_Weg_8_000009_000210
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import DATA_LAF, FIGURES_DIR
from dataloaders.laf_datasets import LostAndFoundOoDDataset

LAF_ROOT   = str(DATA_LAF)
OUTPUT_DIR = FIGURES_DIR / "adaptive_hull"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGFORMER_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# Feste Parameter (konsistent zu evaluate_roi_adaptive_hull.py)
MARGIN_X = 0.08
MARGIN_Y = 0.05

# Festes Trapez — identisch zu visualize_roi_variants.py / eval_config.yaml
TRAP_TOP_Y, TRAP_BOT_Y = 0.28, 0.90
TRAP_TL_X,  TRAP_TR_X  = 0.38, 0.62
TRAP_BL_X,  TRAP_BR_X  = 0.05, 0.95


def make_trapezoid_mask(H, W):
    mask = np.zeros((H, W), dtype=bool)
    top_y = int(TRAP_TOP_Y * H)
    bot_y = int(TRAP_BOT_Y * H)
    for y in range(top_y, bot_y + 1):
        t     = (y - top_y) / max(bot_y - top_y, 1)
        left  = int((TRAP_TL_X * (1 - t) + TRAP_BL_X * t) * W)
        right = int((TRAP_TR_X * (1 - t) + TRAP_BR_X * t) * W)
        mask[y, left:right] = True
    return mask


def make_adaptive_hull(road_mask, margin_x=MARGIN_X, margin_y=MARGIN_Y):
    """Road-folgende, lochfreie Huelle (identisch zu evaluate_roi_adaptive_hull.py)."""
    H, W = road_mask.shape
    hull = np.zeros((H, W), dtype=bool)
    mx   = int(round(margin_x * W))
    rows = []
    for y in range(H):
        xs = np.where(road_mask[y])[0]
        if xs.size == 0:
            continue
        left  = max(0, xs.min() - mx)
        right = min(W, xs.max() + 1 + mx)
        hull[y, left:right] = True
        rows.append(y)
    if rows:
        my = int(round(margin_y * H))
        y_top, y_bot = rows[0], rows[-1]
        for y in range(max(0, y_top - my), y_top):
            hull[y] = hull[y_top]
        for y in range(y_bot + 1, min(H, y_bot + 1 + my)):
            hull[y] = hull[y_bot]
    return hull


def overlay(ax, img_rgb, mask, ood_label, title, color):
    """Bild + halbtransparente ROI + gruene OoD-Kontur."""
    ax.imshow(img_rgb)
    dark = np.zeros((*mask.shape, 4))
    dark[~mask] = [0, 0, 0, 0.55]
    ax.imshow(dark)
    tint = np.zeros((*mask.shape, 4))
    tint[mask] = color
    ax.imshow(tint)
    if ood_label.sum() > 0:
        ax.contour(ood_label, levels=[0.5], colors="lime", linewidths=1.5)
    roi_pct = mask.sum() / mask.size * 100
    ood_in  = int((ood_label[mask] == 1).sum())
    ood_ret = ood_in / max(int(ood_label.sum()), 1) * 100
    ax.set_title(f"{title}\nROI: {roi_pct:.1f}%   OoD-Ret.: {ood_ret:.0f}%",
                 fontsize=12)
    ax.axis("off")


def find_index(ds, name):
    target = name.replace("_leftImg8bit", "")
    for i in range(len(ds)):
        if Path(ds.samples[i][0]).stem.replace("_leftImg8bit", "") == target:
            return i
    for i in range(len(ds)):           # Fallback: Teilstring
        if target in Path(ds.samples[i][0]).stem:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", nargs="+", required=True,
                    help="Ein oder mehrere L&F-Bildnamen (Stem, mit/ohne _leftImg8bit)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LostAndFoundOoDDataset(root=LAF_ROOT, split="test", size=None, min_ood_pixels=0)

    print(f"[SegFormer] Lade {SEGFORMER_ID} ...")
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
    model.to(device).eval()

    for name in args.imgs:
        idx = find_index(ds, name)
        if idx is None:
            print(f"[Warn] '{name}' nicht gefunden, uebersprungen.")
            continue

        sample    = ds[idx]
        img_path  = sample["path"]
        ood_label = sample["ood_label"].astype(np.uint8)
        stem      = Path(img_path).stem.replace("_leftImg8bit", "")
        img_rgb   = np.array(Image.open(img_path).convert("RGB"))
        H, W      = img_rgb.shape[:2]
        print(f"[Info] {stem}  ({H}×{W}, OoD-Pixel: {ood_label.sum():,})")

        img_input = F.interpolate(sample["image"].unsqueeze(0).to(device),
                                  size=(512, 1024), mode="bilinear", align_corners=False)
        with torch.no_grad():
            logits = model(pixel_values=img_input).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred_class = torch.softmax(logits, dim=1)[0].argmax(0).cpu().numpy()

        road = (pred_class == 0) | (pred_class == 1)   # Road + Sidewalk

        mask_b = make_trapezoid_mask(H, W)
        mask_c = road
        mask_e = make_adaptive_hull(road)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        overlay(axes[0], img_rgb, mask_b, ood_label,
                "B: Festes Trapez", [0.20, 0.40, 0.95, 0.30])
        overlay(axes[1], img_rgb, mask_c, ood_label,
                "C: Road+SW", [0.95, 0.55, 0.10, 0.30])
        overlay(axes[2], img_rgb, mask_e, ood_label,
                f"E: Adaptive Huelle (mx={MARGIN_X}, my={MARGIN_Y})",
                [0.15, 0.75, 0.30, 0.35])

        fig.tight_layout()
        out = OUTPUT_DIR / f"adaptive_hull_compare_{stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {out}")

    print(f"\nFertig. Bilder in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
