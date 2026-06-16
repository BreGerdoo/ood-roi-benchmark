"""
evaluate_roi_closing.py
-----------------------
Kapitel 4: Erweiterte ROI-Varianten mit morphologischem Closing
und Canny-Edge-basierter Fahrbahnfuellung. Methoden: Energy, DINOv2 kNN,
PixOOD, RbA.

Voraussetzung: compute_score_maps.py + merge_rba_into_score_maps.py
"""

import argparse
import sys
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import cv2
from scipy.ndimage import binary_fill_holes
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from dataloaders.laf_datasets import LostAndFoundOoDDataset
from paths import DATA_LAF, SCORE_MAPS_LAF, ROI_CLOSING_DIR, FIGURES_DIR

LAF_ROOT      = str(DATA_LAF)
SCORE_MAP_DIR = SCORE_MAPS_LAF
OUTPUT_DIR    = ROI_CLOSING_DIR
VIS_DIR       = FIGURES_DIR / "roi_closing"

CLOSING_KERNELS = [30, 50, 80]
CANNY_LOW  = 50
CANNY_HIGH = 150


def make_road_mask(pred_class):
    return (pred_class == 0).astype(np.uint8)

def make_road_closed(pred_class, kernel_size=50):
    road = make_road_mask(pred_class)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)

def make_road_canny(pred_class, img_rgb, kernel_size=50):
    road = make_road_mask(pred_class)
    gray      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges     = cv2.Canny(gray_blur, CANNY_LOW, CANNY_HIGH)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_thick = cv2.dilate(edges, edge_kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
    edge_barrier = (edges_thick > 0) & (road == 0)
    closed[edge_barrier] = 0
    closed = binary_fill_holes(closed).astype(np.uint8)
    return closed, edges


class StreamingScoreAggregator:
    def __init__(self, score_min, score_max, n_bins=10_000):
        self.bin_edges = np.linspace(score_min, score_max, n_bins + 1)
        self.hist_pos  = np.zeros(n_bins, dtype=np.int64)
        self.hist_neg  = np.zeros(n_bins, dtype=np.int64)
        self.n_pos = 0
        self.n_neg = 0

    def update(self, scores, labels):
        if len(scores) == 0: return
        scores_c = np.clip(scores, self.bin_edges[0], self.bin_edges[-1])
        ood_mask = labels == 1
        ind_mask = labels == 0
        if ood_mask.any():
            h, _ = np.histogram(scores_c[ood_mask], bins=self.bin_edges)
            self.hist_pos += h
            self.n_pos    += int(ood_mask.sum())
        if ind_mask.any():
            h, _ = np.histogram(scores_c[ind_mask], bins=self.bin_edges)
            self.hist_neg += h
            self.n_neg    += int(ind_mask.sum())

    def compute_metrics(self):
        if self.n_pos == 0 or self.n_neg == 0:
            return {"auroc": float("nan"), "ap": float("nan"), "fpr95": float("nan")}
        cum_pos = np.cumsum(self.hist_pos[::-1])[::-1].astype(np.float64)
        cum_neg = np.cumsum(self.hist_neg[::-1])[::-1].astype(np.float64)
        tpr = cum_pos / self.n_pos
        fpr = cum_neg / self.n_neg
        fpr_full = np.concatenate([[0.0], fpr[::-1], [1.0]])
        tpr_full = np.concatenate([[0.0], tpr[::-1], [1.0]])
        auroc = float(np.trapz(tpr_full, fpr_full))
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where(cum_pos + cum_neg > 0,
                                 cum_pos / (cum_pos + cum_neg), 1.0)
        recall_full    = np.concatenate([[0.0], tpr[::-1]])
        precision_full = np.concatenate([[1.0], precision[::-1]])
        ap = float(np.sum(np.diff(recall_full) * precision_full[1:]))
        valid = tpr >= 0.95
        fpr95 = float(fpr[valid].min()) if valid.any() else 1.0
        return {"auroc": auroc, "ap": ap, "fpr95": fpr95}


def visualize_closing(img_rgb, ood_label, pred_class, masks_dict, edges, save_path):
    from scipy.ndimage import binary_dilation, binary_erosion
    import matplotlib.pyplot as plt

    n_masks = len(masks_dict)
    fig, axes = plt.subplots(2, max(n_masks, 3), figsize=(6 * max(n_masks, 3), 10))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    H, W  = img_rgb.shape[:2]
    n_ood = int(ood_label.sum())
    contour = np.zeros_like(ood_label)
    if n_ood > 0:
        contour = (binary_dilation(ood_label, iterations=3) &
                   ~binary_erosion(ood_label, iterations=1)).astype(np.uint8)

    axes[0, 0].imshow(img_rgb)
    overlay = img_rgb.copy().astype(float) / 255.0
    overlay[contour > 0] = [0, 1, 0]
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title(f"Original + GT\n(OoD: {n_ood:,})", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    seg_colors = plt.cm.tab20(np.linspace(0, 1, 19))
    axes[0, 1].imshow(seg_colors[pred_class][..., :3])
    axes[0, 1].set_title("SegFormer Vorhersage", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    if edges is not None:
        axes[0, 2].imshow(edges, cmap="gray")
        axes[0, 2].set_title("Canny Kanten", fontsize=10, fontweight="bold")
    else:
        axes[0, 2].axis("off")
    axes[0, 2].axis("off")

    for i in range(3, max(n_masks, 3)):
        axes[0, i].axis("off")

    for i, (name, mask) in enumerate(masks_dict.items()):
        overlay = img_rgb.copy().astype(float) / 255.0
        overlay[mask == 0] = overlay[mask == 0] * 0.25 + 0.15
        if n_ood > 0:
            overlay[contour > 0] = [0, 1, 0]
        axes[1, i].imshow(overlay)
        roi_pct = mask.sum() / (H * W) * 100
        ood_in  = int((ood_label[mask == 1] == 1).sum())
        ood_ret = ood_in / max(n_ood, 1) * 100
        axes[1, i].set_title(f"{name}\nROI: {roi_pct:.0f}% | Ret.: {ood_ret:.0f}%",
                              fontsize=10, fontweight="bold")
        axes[1, i].axis("off")

    for i in range(len(masks_dict), max(n_masks, 3)):
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--idx",       type=int, default=None)
    parser.add_argument("--img",       type=str, default=None)
    parser.add_argument("--vis_samples", type=int, default=5)
    parser.add_argument("--n_bins", type=int, default=10_000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORE_MAP_DIR.exists() or not list(SCORE_MAP_DIR.glob("*.npz")):
        print(f"[Error] Keine Score-Maps in {SCORE_MAP_DIR}.")
        sys.exit(1)

    ds = LostAndFoundOoDDataset(root=LAF_ROOT, split="test", size=None, min_ood_pixels=100)
    stem_to_path = {Path(s[0]).stem: s[0] for s in ds.samples}
    if not stem_to_path:
        print(f"[Error] Keine Lost & Found RGB-Bilder unter {LAF_ROOT} gefunden.\n"
              f"        Pruefe data/README.md und den Pfad data/ood/. Ohne RGB-Bilder\n"
              f"        kann die Canny-Variante nicht berechnet werden und alle Metriken\n"
              f"        waeren leer (nan).")
        sys.exit(1)

    npz_files = sorted(SCORE_MAP_DIR.glob("*.npz"))
    n = len(npz_files) if args.max_images < 0 else min(args.max_images, len(npz_files))
    print(f"[Score-Maps] {n} Dateien")

    if args.visualize and args.img is not None:
        for i, f in enumerate(npz_files):
            if args.img in f.stem:
                args.idx = i
                break
        else:
            print(f"[Error] '{args.img}' nicht gefunden.")
            return

    if args.visualize and args.idx is not None:
        start_idx = args.idx
        n = args.idx + 1
    else:
        start_idx = 0

    method_names = ["Energy", "DINOv2 kNN", "PixOOD", "RbA"]
    method_keys  = {
        "Energy":     "energy_map",
        "DINOv2 kNN": "knn_map",
        "PixOOD":     "pixood_map",
        "RbA":        "rba_map",
    }

    print("[Init] Bestimme globale Score-Ranges aus Stichprobe ...")
    sample_size = min(50, n)
    score_samples = {m: [] for m in method_names}
    for f in npz_files[:sample_size]:
        d = np.load(f)
        for m in method_names:
            if method_keys[m] in d.files:
                score_samples[m].append(d[method_keys[m]].astype(np.float32).flatten())

    score_ranges = {}
    for m in method_names:
        if not score_samples[m]:
            print(f"[Warn] {m}: keine Maps, uebersprungen.")
            score_ranges[m] = None
            continue
        c = np.concatenate(score_samples[m])
        margin = max(0.1, (c.max() - c.min()) * 0.05)
        score_ranges[m] = (float(c.min()) - margin, float(c.max()) + margin)
        print(f"[Init] {m:12s} range: [{score_ranges[m][0]:.4f}, {score_ranges[m][1]:.4f}]")

    method_names = [m for m in method_names if score_ranges[m] is not None]

    variant_names = (
        ["C: Road"] +
        [f"C'-Morph (k={k})" for k in CLOSING_KERNELS] +
        ["C'-Canny (k=50)"]
    )

    aggs = {}
    for v in variant_names:
        aggs[v] = {}
        for m in method_names:
            smin, smax = score_ranges[m]
            aggs[v][m] = StreamingScoreAggregator(smin, smax, n_bins=args.n_bins)

    roi_stats    = {v: {"roi_pixels": 0, "ood_in_roi": 0} for v in variant_names}
    total_pixels = 0
    total_ood    = 0

    print(f"\n[Eval] Verarbeite {n} Bilder ab Index {start_idx} ...")
    for idx in tqdm(range(start_idx, n), desc="Closing-Eval"):
        f = npz_files[idx]
        d = np.load(f)
        pred_class = d["pred_class"].astype(np.int32)
        ood_label  = d["ood_label"].astype(np.int32)
        H, W       = ood_label.shape

        total_pixels += H * W
        total_ood    += int(ood_label.sum())

        score_maps = {}
        for m in method_names:
            if method_keys[m] in d.files:
                score_maps[m] = d[method_keys[m]].astype(np.float32)

        # Road- und Morph-Masken benoetigen KEIN RGB-Bild -> immer berechnen.
        mask_c = make_road_mask(pred_class)
        masks_morph = {k: make_road_closed(pred_class, k) for k in CLOSING_KERNELS}

        all_masks = {"C: Road": mask_c}
        for k in CLOSING_KERNELS:
            all_masks[f"C'-Morph (k={k})"] = masks_morph[k]

        # Canny braucht das RGB-Bild. Fehlt es, wird NUR diese Variante ausgelassen,
        # die uebrigen Varianten werden trotzdem korrekt ausgewertet.
        rgb_path = stem_to_path.get(f.stem)
        edges = None
        if rgb_path is not None:
            img_rgb = np.array(Image.open(rgb_path).convert("RGB"))
            mask_canny, edges = make_road_canny(pred_class, img_rgb, kernel_size=50)
            all_masks["C'-Canny (k=50)"] = mask_canny
        else:
            print(f"[Warn] Kein RGB-Pfad fuer {f.stem}: C'-Canny fuer dieses Bild uebersprungen.")

        for vname, mask in all_masks.items():
            mask_bool = mask.astype(bool)
            roi_stats[vname]["roi_pixels"] += int(mask_bool.sum())
            roi_stats[vname]["ood_in_roi"] += int((ood_label[mask_bool] == 1).sum())
            for mname, smap in score_maps.items():
                aggs[vname][mname].update(smap[mask_bool], ood_label[mask_bool])

        if args.visualize and rgb_path is not None:
            if args.idx is not None and idx != args.idx:
                continue
            elif args.idx is None and idx >= args.vis_samples:
                continue
            vis_masks = {
                "C: Road (Basis)":  mask_c,
                "C'-Morph (k=50)":  masks_morph[50],
                "C'-Canny (k=50)":  mask_canny,
            }
            save_path = VIS_DIR / f"roi_closing_{f.stem}.png"
            visualize_closing(img_rgb, ood_label, pred_class,
                              vis_masks, edges, save_path)

    if args.visualize and args.idx is not None:
        return

    print("\n" + "=" * 95)
    print(f"  Kapitel 4 -- Closing-Varianten Ergebnisse ({n} Bilder)")
    print("=" * 95)
    header = (f"{'Variante':<22} {'Methode':<14} {'AUROC':>7} {'AP':>8} "
              f"{'FPR95':>7} {'ROI %':>7} {'OoD-Ret.':>9}")
    print(header)
    print("-" * 95)

    result_rows = []
    for vname in variant_names:
        roi_pct = roi_stats[vname]["roi_pixels"] / total_pixels * 100
        ood_ret = roi_stats[vname]["ood_in_roi"] / max(total_ood, 1) * 100
        for mname in method_names:
            m = aggs[vname][mname].compute_metrics()
            print(f"  {vname:<22} {mname:<14} {m['auroc']:7.4f} {m['ap']:8.4f} "
                  f"{m['fpr95']:7.4f} {roi_pct:6.1f}% {ood_ret:7.1f}%")
            result_rows.append({
                "Variante": vname, "Methode": mname,
                "AUROC": round(m["auroc"], 4),
                "AP": round(m["ap"], 4),
                "FPR95": round(m["fpr95"], 4),
                "ROI_pct": round(roi_pct, 1),
                "OoD_Retention_pct": round(ood_ret, 1),
            })
        print()

    csv_path = OUTPUT_DIR / "closing_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"[Saved] {csv_path}")

    tex_path = OUTPUT_DIR / "closing_results.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by evaluate_roi_closing.py\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{OoD-Retention und Detektionsmetriken unter verschiedenen\n")
        f.write("         Closing-Strategien. Lost~\\&~Found, " + str(n) + " Bilder.}\n")
        f.write("\\label{tab:closing_results}\n")
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("\\textbf{Variante} & \\textbf{Methode} & \\textbf{AUROC}$\\uparrow$")
        f.write(" & \\textbf{AP}$\\uparrow$ & \\textbf{FPR95}$\\downarrow$")
        f.write(" & \\textbf{ROI} & \\textbf{OoD-Ret.} \\\\\n\\midrule\n")
        prev = None
        for row in result_rows:
            if prev and row["Variante"] != prev:
                f.write("\\midrule\n")
            prev = row["Variante"]
            v = row["Variante"].replace("&", "\\&").replace("'", "'")
            f.write(f"{v} & {row['Methode']} & {row['AUROC']:.4f} & "
                    f"{row['AP']:.4f} & {row['FPR95']:.4f} & "
                    f"{row['ROI_pct']:.0f}\\,\\% & "
                    f"{row['OoD_Retention_pct']:.0f}\\,\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[Saved] {tex_path}")

    print(f"\n{'=' * 60}")
    print(f"Fertig. Ergebnisse in: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
