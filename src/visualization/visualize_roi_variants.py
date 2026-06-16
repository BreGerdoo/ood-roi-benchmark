"""
visualize_roi_variants.py
-------------------------
Visualisiert die vier ROI-Varianten (A–D) für ein Lost & Found-Beispielbild.
Erzeugt ein 2×3 Panel-Bild für Kapitel 4 der Bachelorarbeit.

Aufruf:
    python visualize_roi_variants.py                          # erstes Bild mit OoD
    python visualize_roi_variants.py --idx 5                  # nach Index
    python visualize_roi_variants.py --img "04_Maurener_Weg"  # nach Name (Teilstring)

Ausgabe:
    results/figures/roi_variants/roi_variants_<name>.png
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — same convention as existing scripts
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import DATA_LAF, FIGURES_DIR

LAF_ROOT = str(DATA_LAF)
OUTPUT_DIR = FIGURES_DIR / "roi_variants"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
from dataloaders.laf_datasets import LostAndFoundOoDDataset

# ---------------------------------------------------------------------------
# SegFormer loading
# ---------------------------------------------------------------------------
SEGFORMER_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# Cityscapes class names for trainIds 0–18
CS_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

# Background trainIds for negative filtering (Variant D)
BG_TRAIN_IDS = {2, 3, 4, 5, 8, 10}  # building, wall, fence, pole, vegetation, sky
BG_NAMES = [CS_CLASSES[i] for i in sorted(BG_TRAIN_IDS)]

# ---------------------------------------------------------------------------
# Trapezoid ROI (Variant B) — same defaults as eval_config.yaml
# ---------------------------------------------------------------------------
TRAP_TOP_Y = 0.28
TRAP_BOT_Y = 0.90
TRAP_TL_X = 0.38
TRAP_TR_X = 0.62
TRAP_BL_X = 0.05
TRAP_BR_X = 0.95


def make_trapezoid_mask(H, W):
    """Create a binary trapezoid mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    top_y = int(TRAP_TOP_Y * H)
    bot_y = int(TRAP_BOT_Y * H)
    for y in range(top_y, bot_y + 1):
        t = (y - top_y) / max(bot_y - top_y, 1)
        left = int((TRAP_TL_X * (1 - t) + TRAP_BL_X * t) * W)
        right = int((TRAP_TR_X * (1 - t) + TRAP_BR_X * t) * W)
        mask[y, left:right] = 1
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--msp_thresh", type=float, default=0.95,
                        help="MSP threshold for Variant D (default: 0.95)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset (full resolution) ---
    ds = LostAndFoundOoDDataset(root=LAF_ROOT, split="test", size=None,
                                min_ood_pixels=100)

    # --- Find image ---
    idx = 0
    if args.idx is not None:
        idx = args.idx
    elif args.img is not None:
        for i in range(len(ds)):
            if args.img in ds.samples[i][0].stem:
                idx = i
                break
        else:
            print(f"[Error] '{args.img}' nicht gefunden.")
            return

    sample = ds[idx]
    img_path = sample["path"]
    ood_label = sample["ood_label"]
    img_name = Path(img_path).stem

    # Load raw RGB for display
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_rgb.shape[:2]

    print(f"[Info] Bild: {img_name}")
    print(f"[Info] Auflösung: {H}×{W}")
    print(f"[Info] OoD-Pixel: {ood_label.sum():,}")

    # --- Load SegFormer ---
    print(f"[SegFormer] Lade {SEGFORMER_ID} ...")
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
    model.to(device).eval()

    # --- Run inference ---
    img_tensor = sample["image"].unsqueeze(0).to(device)
    # SegFormer expects variable resolution, we resize to 512×1024 for speed
    img_input = F.interpolate(img_tensor, size=(512, 1024), mode="bilinear",
                              align_corners=False)

    with torch.no_grad():
        outputs = model(pixel_values=img_input)
        logits = outputs.logits  # [1, 19, h, w]
        # Upsample to original resolution
        logits_full = F.interpolate(logits, size=(H, W), mode="bilinear",
                                    align_corners=False)

    softmax = torch.softmax(logits_full, dim=1)[0]  # [19, H, W]
    pred_class = softmax.argmax(dim=0).cpu().numpy()  # [H, W]
    msp_map = softmax.max(dim=0).values.cpu().numpy()  # [H, W]

    # --- Build four ROI masks ---

    # Variant A: Full image
    mask_a = np.ones((H, W), dtype=np.uint8)

    # Variant B: Trapezoid
    mask_b = make_trapezoid_mask(H, W)

    # Variant C: Segmentation-based (road only)
    mask_c_road = (pred_class == 0).astype(np.uint8)

    # Variant C: road + sidewalk
    mask_c_road_sw = ((pred_class == 0) | (pred_class == 1)).astype(np.uint8)

    # Variant D: Negative filtering (exclude confident background)
    bg_mask = np.zeros((H, W), dtype=bool)
    for tid in BG_TRAIN_IDS:
        bg_mask |= ((pred_class == tid) & (msp_map > args.msp_thresh))
    mask_d = (~bg_mask).astype(np.uint8)

    # --- Compute stats ---
    masks = {
        "A: Volles Bild": mask_a,
        "B: Trapez": mask_b,
        f"C: Road": mask_c_road,
        f"C: Road+SW": mask_c_road_sw,
        f"D: Neg. Filter\n(MSP>{args.msp_thresh})": mask_d,
    }

    n_ood_total = int(ood_label.sum())
    print(f"\n{'Variante':<28} {'ROI %':>7} {'OoD in ROI':>12} {'OoD Ret.':>9}")
    print("-" * 60)
    for name, m in masks.items():
        roi_pct = m.sum() / (H * W) * 100
        ood_in = int((ood_label[m == 1] == 1).sum())
        ood_ret = ood_in / max(n_ood_total, 1) * 100
        print(f"{name:<28} {roi_pct:6.1f}% {ood_in:>8,} / {n_ood_total:,} {ood_ret:7.1f}%")

    # --- Visualize ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # Helper: overlay mask on image
    def show_roi(ax, mask, title, color_excluded=(0.2, 0.2, 0.2)):
        overlay = img_rgb.copy().astype(float) / 255.0
        excluded = mask == 0
        overlay[excluded] = overlay[excluded] * 0.25 + np.array(color_excluded) * 0.75

        # Draw OoD contours in green
        from scipy.ndimage import binary_dilation, binary_erosion
        if ood_label.sum() > 0:
            contour = binary_dilation(ood_label, iterations=2) & ~binary_erosion(ood_label, iterations=1)
            overlay[contour > 0] = [0, 1, 0]

        ax.imshow(overlay)

        roi_pct = mask.sum() / (H * W) * 100
        ood_in = int((ood_label[mask == 1] == 1).sum())
        ood_ret = ood_in / max(n_ood_total, 1) * 100
        ax.set_title(f"{title}\nROI: {roi_pct:.0f}% | OoD-Ret.: {ood_ret:.0f}%",
                     fontsize=11, fontweight="bold")
        ax.axis("off")

    # Panel (0,0): Original + GT
    axes[0, 0].imshow(img_rgb)
    if ood_label.sum() > 0:
        from scipy.ndimage import binary_dilation, binary_erosion
        contour = binary_dilation(ood_label, iterations=3) & ~binary_erosion(ood_label, iterations=1)
        overlay_orig = img_rgb.copy().astype(float) / 255.0
        overlay_orig[contour > 0] = [0, 1, 0]
        axes[0, 0].imshow(overlay_orig)
    axes[0, 0].set_title(f"A: Original + GT\n(OoD-Pixel: {n_ood_total:,})",
                         fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel (0,1): Segmentierungsvorhersage
    seg_colors = plt.cm.tab20(np.linspace(0, 1, 19))
    seg_vis = seg_colors[pred_class][..., :3]
    axes[0, 1].imshow(seg_vis)
    axes[0, 1].set_title("SegFormer-B2 Vorhersage\n(19 Cityscapes-Klassen)",
                         fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel (0,2): MSP confidence map
    im_msp = axes[0, 2].imshow(msp_map, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[0, 2].set_title(f"MSP Konfidenz\n(Schwelle D: {args.msp_thresh})",
                         fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im_msp, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Panel (1,0): Variant B — Trapez
    show_roi(axes[1, 0], mask_b, "B: Festes Trapez")

    # Panel (1,1): Variant C — Road only
    show_roi(axes[1, 1], mask_c_road, "C: Road-ROI (trainId=0)")

    # Panel (1,2): Variant D — Negative Filterung
    show_roi(axes[1, 2], mask_d,
             f"D: Neg. Filterung (MSP>{args.msp_thresh})")

    # plt.suptitle(f"ROI-Varianten — {Path(img_path).stem}",
    #             fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"roi_variants_{img_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {save_path}")


if __name__ == "__main__":
    main()
