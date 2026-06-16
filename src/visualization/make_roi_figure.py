"""
make_roi_figure.py  (v3: 2x2 layout, only ROI variants)
========================================================
Generates a paper-quality figure showing the four ROI variants on one
Lost & Found example image. All four variants (A, B, C, D) are shown
in a 2x2 grid. The ground-truth OoD contour is overlaid in green so
the reader can immediately see which variants retain the OoD object.

Usage
-----
    python make_roi_figure.py \\
        --image  /path/to/.../_leftImg8bit.png \\
        --label  /path/to/.../_gtCoarse_labelTrainIds.png \\
        --out    roi_variants.pdf
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import FIGURES_DIR


TRAPEZOID_REL = {
    "y_top":  0.28, "x_top_l": 0.38, "x_top_r": 0.62,
    "y_bot":  0.90, "x_bot_l": 0.05, "x_bot_r": 0.95,
}

BACKGROUND_IDS = [2, 3, 4, 8, 10]
MSP_THRESHOLD  = 0.95
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
SEGFORMER_INFER_SIZE = (512, 1024)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True)
    p.add_argument("--label",  required=True)
    p.add_argument("--out",    default=str(FIGURES_DIR / "roi_schematic" / "roi_variants.pdf"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dpi",    type=int, default=300)
    return p.parse_args()


def preprocess(pil_image):
    transform = T.Compose([
        T.Resize(SEGFORMER_INFER_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(pil_image).unsqueeze(0)


def trapezoid_mask(H, W):
    pts = np.array([
        [int(TRAPEZOID_REL["x_top_l"] * W), int(TRAPEZOID_REL["y_top"] * H)],
        [int(TRAPEZOID_REL["x_top_r"] * W), int(TRAPEZOID_REL["y_top"] * H)],
        [int(TRAPEZOID_REL["x_bot_r"] * W), int(TRAPEZOID_REL["y_bot"] * H)],
        [int(TRAPEZOID_REL["x_bot_l"] * W), int(TRAPEZOID_REL["y_bot"] * H)],
    ], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def segformer_predict(model, pil_image, device):
    H, W = pil_image.height, pil_image.width
    pixel_values = preprocess(pil_image).to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
    logits_full = F.interpolate(logits, size=(H, W),
                                mode="bilinear", align_corners=False)
    probs = logits_full.softmax(dim=1).squeeze(0).cpu()
    pred  = probs.argmax(0).numpy()
    msp   = probs.max(0).values.numpy()
    return pred, msp


def roi_masks(pred, msp, H, W):
    road_mask = (pred == 0)
    bg_mask = np.zeros((H, W), dtype=bool)
    for c in BACKGROUND_IDS:
        bg_mask |= ((pred == c) & (msp > MSP_THRESHOLD))
    return {
        "A": np.ones((H, W), dtype=bool),
        "B": trapezoid_mask(H, W),
        "C": road_mask,
        "D": ~bg_mask,
    }


def darken_outside_roi(image, roi_mask, dim_factor=0.25):
    """Variant A returns the original image unchanged."""
    out = image.astype(np.float32)
    out[~roi_mask] *= dim_factor
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_gt_contour(ax, gt_binary, colour="#22cc55", linewidth=1.4):
    ax.contour(gt_binary.astype(float), levels=[0.5],
               colors=colour, linewidths=linewidth)


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#aaaaaa")


def main():
    args = get_args()

    pil_img = Image.open(args.image).convert("RGB")
    img_np  = np.array(pil_img)
    H, W    = img_np.shape[:2]

    gt_full = np.array(Image.open(args.label))
    gt_ood  = (gt_full == 2)
    n_ood   = int(gt_ood.sum())
    print(f"[Info] Image: {Path(args.image).name}")
    print(f"[Info] Size: {W}x{H},  OoD pixels: {n_ood}")

    print("[Info] Loading SegFormer-B2 ...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        ).to(args.device).eval()
    pred, msp = segformer_predict(model, pil_img, args.device)

    masks = roi_masks(pred, msp, H, W)

    stats = {}
    for name, m in masks.items():
        cov = m.sum() / (H * W)
        ret = (m & gt_ood).sum() / max(n_ood, 1)
        stats[name] = (cov, ret)
        print(f"[Info] Variant {name}: ROI={cov*100:.1f}%, "
              f"OoD-Retention={ret*100:.1f}%")

    # ---- Figure: 2x2 grid, all four ROI variants -----------------------
    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 5.0), dpi=args.dpi,
        constrained_layout=True,
    )
    title_fs = 11

    variant_layout = [
        ("A", "Variant A: full image"),
        ("B", "Variant B: trapezoid"),
        ("C", "Variant C: road-ROI"),
        ("D", "Variant D: negative filter"),
    ]

    for ax, (key, name) in zip(axes.flat, variant_layout):
        dimmed = darken_outside_roi(img_np, masks[key])
        ax.imshow(dimmed)
        overlay_gt_contour(ax, gt_ood)
        clean_axis(ax)
        cov, ret = stats[key]
        title = (f"{name}\n"
                 f"ROI {cov*100:.0f}%   OoD-Ret. {ret*100:.0f}%")
        ax.set_title(title, fontsize=title_fs, pad=4, linespacing=1.15)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"\n[Saved] {out_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
