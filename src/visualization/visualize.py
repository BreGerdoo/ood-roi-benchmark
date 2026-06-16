"""
visualize.py
------------
Qualitative visualization for Chapter 2.3.

For each image produces a 3-panel figure:
  [Original + Segmentation] | [Entropy Heatmap] | [Energy Heatmap]

With annotations showing:
  - Predicted class labels on segmentation overlay
  - OoD score values on suspicious regions
  - Colorbar for uncertainty scale

Usage:
    python evaluation/visualize.py
    python evaluation/visualize.py --n_images 4 --split laf
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from evaluation.uncertainty import SCORE_FUNCTIONS
from dataloaders.laf_datasets import CityscapesInDDataset, LostAndFoundOoDDataset
from models.load_model import load_deeplabv3_cityscapes, forward_logits
from paths import FIGURES_DIR

# ---------------------------------------------------------------------------
# Cityscapes 19-class color palette and names
# ---------------------------------------------------------------------------
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

CITYSCAPES_COLORS = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)

# Custom heatmap: blue (certain/InD) → yellow → red (uncertain/OoD)
OOD_CMAP = LinearSegmentedColormap.from_list(
    "ood", ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#b2182b"]
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalized tensor [3,H,W] back to uint8 RGB image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def logits_to_segmentation_rgb(logits):
    """Convert logits [C,H,W] to RGB segmentation map."""
    pred = logits.argmax(dim=0).cpu().numpy()   # [H,W]
    seg_rgb = CITYSCAPES_COLORS[pred % len(CITYSCAPES_COLORS)]
    return seg_rgb, pred


def visualize_sample(image_tensor, logits, ood_label, save_path, title=""):
    """
    2x2 paper-ready figure with white background.

    Layout:
      [Original image]        [Segmentation overlay]
      [Entropy heatmap]       [Energy heatmap + GT boxes]
    """
    from scipy import ndimage as ndi
    import matplotlib.ticker as ticker

    H, W = image_tensor.shape[1], image_tensor.shape[2]

    orig_img           = denormalize(image_tensor)
    seg_rgb, pred_cls  = logits_to_segmentation_rgb(logits)
    entropy_map        = SCORE_FUNCTIONS["entropy"](logits)
    energy_map         = SCORE_FUNCTIONS["energy"](logits)
    msp_map            = SCORE_FUNCTIONS["msp"](logits)

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-10)

    # --- Figure setup: 2x2, white background, tight ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.axis("off")

    ax_img, ax_seg, ax_ent, ax_nrg = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    # ------------------------------------------------------------------
    # Panel A: Original image
    # ------------------------------------------------------------------
    ax_img.imshow(orig_img)
    ax_img.set_title("(a) Eingabebild", fontsize=10, color="black",
                     loc="left", pad=4)

    # ------------------------------------------------------------------
    # Panel B: Segmentation overlay
    # ------------------------------------------------------------------
    blend = (0.55 * orig_img + 0.45 * seg_rgb).astype(np.uint8)
    ax_seg.imshow(blend)
    ax_seg.set_title("(b) Segmentierung (SegFormer-B2)", fontsize=10,
                     color="black", loc="left", pad=4)

    visible_ids = np.unique(pred_cls)
    patches = []
    for cid in visible_ids[:7]:
        color = CITYSCAPES_COLORS[cid] / 255.0
        patches.append(mpatches.Patch(color=color, label=CITYSCAPES_CLASSES[cid]))
    ax_seg.legend(handles=patches, loc="lower left", fontsize=6.5,
                  framealpha=0.85, facecolor="white", edgecolor="#cccccc",
                  labelcolor="black", handlelength=1.0, borderpad=0.4,
                  handletextpad=0.4)

    # ------------------------------------------------------------------
    # Panel C: Entropy heatmap
    # ------------------------------------------------------------------
    ax_ent.imshow(orig_img, alpha=0.25)
    im_ent = ax_ent.imshow(norm(entropy_map), cmap=OOD_CMAP,
                            alpha=0.80, vmin=0, vmax=1)
    ax_ent.set_title("(c) Prädiktive Entropie", fontsize=10,
                     color="black", loc="left", pad=4)

    # Annotate max entropy location — text always in top strip, arrow points down
    max_idx = np.unravel_index(entropy_map.argmax(), entropy_map.shape)
    max_y, max_x = int(max_idx[0]), int(max_idx[1])
    # Text goes to top-left or top-right depending on x position of max
    text_x = W * 0.55 if max_x < W * 0.5 else W * 0.02
    text_y = H * 0.04
    ax_ent.annotate(
        f"max H = {entropy_map.max():.3f}",
        xy=(max_x, max_y),
        xytext=(text_x, text_y),
        fontsize=7.5, color="black", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#888888", linewidth=0.8, alpha=0.90),
        zorder=6
    )

    # Draw GT contour dashed on entropy panel
    has_ood = (ood_label is not None) and (ood_label.sum() > 0)
    if has_ood:
        ax_ent.contour(ood_label, levels=[0.5], colors=["#00bb66"],
                       linewidths=1.2, linestyles="dashed")

    cbar_ent = fig.colorbar(im_ent, ax=ax_ent, fraction=0.030, pad=0.02)
    cbar_ent.set_label("normierte Entropie", fontsize=7.5, color="black")
    cbar_ent.ax.tick_params(labelsize=7, color="black")
    plt.setp(cbar_ent.ax.yaxis.get_ticklabels(), color="black")

    # ------------------------------------------------------------------
    # Panel D: Energy heatmap + GT bounding boxes
    # ------------------------------------------------------------------
    ax_nrg.imshow(orig_img, alpha=0.25)
    im_nrg = ax_nrg.imshow(norm(energy_map), cmap=OOD_CMAP,
                            alpha=0.80, vmin=0, vmax=1)
    ax_nrg.set_title("(d) Energy Score", fontsize=10,
                     color="black", loc="left", pad=4)

    if has_ood:
        labeled, n_objects = ndi.label(ood_label)
        print(f"[Viz] Found {n_objects} OoD object(s) in ground truth")

        corners = [
            (W * 0.02, H * 0.04),
            (W * 0.55, H * 0.04),
            (W * 0.02, H * 0.72),
            (W * 0.55, H * 0.72),
        ]

        for obj_idx in range(1, n_objects + 1):
            obj_mask = (labeled == obj_idx)
            n_px = int(obj_mask.sum())
            if n_px < 1:
                continue

            coords = np.argwhere(obj_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            cy = int((y_min + y_max) / 2)
            cx = int((x_min + x_max) / 2)

            pad = max(6, int(min(H, W) * 0.018))
            rx  = max(0, x_min - pad)
            ry  = max(0, y_min - pad)
            rw  = min(W - rx, (x_max - x_min) + 2 * pad)
            rh  = min(H - ry, (y_max - y_min) + 2 * pad)

            rect = mpatches.Rectangle(
                (rx, ry), rw, rh,
                linewidth=2.0, edgecolor="#007744", facecolor="none",
                linestyle="-", zorder=5
            )
            ax_nrg.add_patch(rect)

            mean_entropy = float(entropy_map[obj_mask].mean())
            mean_energy  = float(energy_map[obj_mask].mean())

            tx, ty = corners[(obj_idx - 1) % len(corners)]
            ax_nrg.annotate(
                f"GT OoD {obj_idx}  H={mean_entropy:.3f}  E={mean_energy:.2f}",
                xy=(cx, cy),
                xytext=(tx, ty),
                fontsize=7, color="#007744", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#007744", lw=1.2,
                                connectionstyle="arc3,rad=0.15"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#007744", linewidth=1.0, alpha=0.92),
                zorder=6
            )

    cbar_nrg = fig.colorbar(im_nrg, ax=ax_nrg, fraction=0.030, pad=0.02)
    cbar_nrg.set_label("normierter Energy Score", fontsize=7.5, color="black")
    cbar_nrg.ax.tick_params(labelsize=7, color="black")
    plt.setp(cbar_nrg.ax.yaxis.get_ticklabels(), color="black")

    # ------------------------------------------------------------------
    # Score summary — compact, no box, directly below figure
    # ------------------------------------------------------------------
    if has_ood:
        ood_mask = ood_label.astype(bool)
        ind_mask = ~ood_mask
        line1 = (f"MSP: {msp_map.mean():.4f}  |  "
                 f"Entropy: {entropy_map.mean():.4f}  |  "
                 f"Energy: {energy_map.mean():.4f}")
        line2 = (f"Entropie OoD-Pixel: {entropy_map[ood_mask].mean():.4f}  |  "
                 f"Entropie InD-Pixel: {entropy_map[ind_mask].mean():.4f}  |  "
                 f"Differenz: {entropy_map[ood_mask].mean() - entropy_map[ind_mask].mean():+.4f}")
        stats = line1 + "\n" + line2
    else:
        stats = (f"MSP: {msp_map.mean():.4f}  |  "
                 f"Entropy: {entropy_map.mean():.4f}  |  "
                 f"Energy: {energy_map.mean():.4f}")

    fig.text(0.5, -0.01, stats, ha="center", va="top",
             fontsize=7.5, color="#444444", style="italic")

    plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=3,
                        help="Number of images to visualize per dataset")
    parser.add_argument("--split",    default="both",
                        choices=["laf", "cityscapes", "both"])
    parser.add_argument("--output_dir", default=str(FIGURES_DIR / "chapter2"))
    parser.add_argument("--filter", default=None,
                        help="Only visualize images whose filename contains this string")
    parser.add_argument("--imgs", nargs="+", default=None,
                        help="Exact image stems to visualize (e.g. "
                             "02_Hanns_Klemm_Str_44_000006_000180). Overrides --n_images "
                             "for L&F, bypasses the min-OoD-pixel filter, and saves each "
                             "figure under its own image name.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config_path = PROJECT_ROOT / "configs" / "eval_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    size     = tuple(data_cfg["img_size"]) if data_cfg.get("img_size") else None

    # Resolve relative data paths against project root
    for key in ("laf_root", "cityscapes_root"):
        if key in data_cfg:
            p = Path(data_cfg[key])
            if not p.is_absolute():
                data_cfg[key] = str(PROJECT_ROOT / p)

    model, device = load_deeplabv3_cityscapes(config.get("device", "cpu"))

    # ------------------------------------------------------------------
    # Lost & Found (OoD) — most interesting for the thesis
    # ------------------------------------------------------------------
    if args.split in ("laf", "both"):
        # Bei expliziter Bildauswahl (--imgs): min_ood_pixels-Filter umgehen,
        # damit kein angefordertes Bild vorab herausfaellt.
        laf_ds = LostAndFoundOoDDataset(
            root=data_cfg["laf_root"], size=size,
            min_ood_pixels=0 if args.imgs else 500
        )

        if args.imgs:
            # Genau die angegebenen Bildnamen, in der angegebenen Reihenfolge.
            wanted = [s.replace("_leftImg8bit", "") for s in args.imgs]
            selected = []
            for stem in wanted:
                match = [(p, l) for p, l in laf_ds.samples
                         if Path(p).stem.replace("_leftImg8bit", "") == stem]
                if match:
                    selected.append(match[0])
                else:
                    print(f"[Viz] WARNUNG: Bild nicht gefunden: {stem}")
            laf_ds.samples = selected
            print(f"\n[Viz] Visualizing {len(selected)} explizit angeforderte L&F-Bilder...")
            for p, l in selected:
                idx      = laf_ds.samples.index((p, l))
                sample   = laf_ds[idx]
                logits   = forward_logits(model, sample["image"], device)
                stem     = Path(sample["path"]).stem.replace("_leftImg8bit", "")
                try:
                    visualize_sample(
                        image_tensor = sample["image"],
                        logits       = logits,
                        ood_label    = sample["ood_label"],
                        save_path    = os.path.join(args.output_dir, f"{stem}_leftImg8bit.png"),
                        title        = f"Lost & Found — OoD-Szenario | {stem}",
                    )
                except Exception as e:
                    import traceback
                    print(f"[Viz] ERROR on {stem}: {e}")
                    traceback.print_exc()
            # Bei expliziter Auswahl keine Cityscapes-Bilder mit erzeugen
            print(f"\n[Viz] All visualizations saved to: {args.output_dir}/")
            return

        # Apply optional filename filter
        if args.filter:
            laf_ds.samples = [(p, l) for p, l in laf_ds.samples
                              if args.filter in str(p)]
            print(f"[Viz] After filter '{args.filter}': {len(laf_ds.samples)} images")

        print(f"\n[Viz] Visualizing {args.n_images} Lost & Found images...")
        for i in range(min(args.n_images, len(laf_ds))):
            sample   = laf_ds[i]
            logits   = forward_logits(model, sample["image"], device)
            img_name = Path(sample["path"]).stem[:40]

            try:
                visualize_sample(
                    image_tensor = sample["image"],
                    logits       = logits,
                    ood_label    = sample["ood_label"],
                    save_path    = os.path.join(args.output_dir, f"laf_{i:02d}_{img_name}.png"),
                    title        = f"Lost & Found — OoD-Szenario | {img_name}",
                )
            except Exception as e:
                import traceback
                print(f"[Viz] ERROR on image {i}: {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Cityscapes (InD baseline) — for comparison
    # ------------------------------------------------------------------
    if args.split in ("cityscapes", "both"):
        cs_ds = CityscapesInDDataset(
            root=data_cfg["cityscapes_root"], split="val", size=size
        )
        print(f"\n[Viz] Visualizing {args.n_images} Cityscapes images...")
        for i in range(min(args.n_images, len(cs_ds))):
            sample   = cs_ds[i]
            logits   = forward_logits(model, sample["image"], device)
            img_name = Path(sample["path"]).stem[:40]

            visualize_sample(
                image_tensor = sample["image"],
                logits       = logits,
                ood_label    = None,   # no OoD ground truth for Cityscapes
                save_path    = os.path.join(args.output_dir, f"cs_{i:02d}_{img_name}.png"),
                title        = f"Cityscapes (InD-Baseline) | {img_name}",
            )

    print(f"\n[Viz] All visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
