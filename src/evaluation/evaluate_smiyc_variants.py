"""
evaluate_smiyc_variants.py
--------------------------
Wertet die ROI-Varianten (A-D) auf den vorab berechneten SMIYC-Score-Maps aus.
Methoden: Energy, DINOv2 kNN, PixOOD, RbA.
Beide Tracks (RoadAnomaly21 + RoadObstacle21) in EINEM Aufruf.

Voraussetzung:
    compute_score_maps_smiyc.py (beide Tracks) UND
    merge_rba_into_score_maps_smiyc.py wurden ausgeführt.

Aufruf:
    python evaluate_smiyc_variants.py
    python evaluate_smiyc_variants.py --msp_thresh 0.95

Output (pro Track):
    results/smiyc/<Track>/smiyc_results.csv
    results/smiyc/<Track>/smiyc_results.tex
    results/smiyc/<Track>/per_image_auroc.csv

Hinweis: Auswertung auf dem öffentlichen validation-Split (n=10 bzw. n=30),
da die offiziellen Test-Labels zurückgehalten werden. Bei n=10
(RoadAnomaly21) sind Per-Image-Statistiken grobkörnig — gepoolte
Pixel-Metriken bleiben aussagekräftig.
"""

import argparse
import csv
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import SMIYC_RESULTS_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRACKS = ["RoadAnomaly21", "RoadObstacle21"]

TRAP_TOP_Y, TRAP_BOT_Y = 0.28, 0.90
TRAP_TL_X,  TRAP_TR_X  = 0.38, 0.62
TRAP_BL_X,  TRAP_BR_X  = 0.05, 0.95

BG_TRAIN_IDS  = {2, 3, 4, 5, 8, 10}
MSP_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


class StreamingScoreAggregator:
    def __init__(self, score_min, score_max, n_bins=10_000):
        self.bin_edges = np.linspace(score_min, score_max, n_bins + 1)
        self.hist_pos  = np.zeros(n_bins, dtype=np.int64)
        self.hist_neg  = np.zeros(n_bins, dtype=np.int64)
        self.n_pos = 0
        self.n_neg = 0

    def update(self, scores, labels):
        if len(scores) == 0:
            return
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


# ---------------------------------------------------------------------------
# Per-Track Evaluation
# ---------------------------------------------------------------------------
def evaluate_track(track, msp_thresh, n_bins):
    score_map_dir = SMIYC_RESULTS_DIR / track / "score_maps"
    output_dir    = SMIYC_RESULTS_DIR / track

    npz_files = sorted(score_map_dir.glob("*.npz"))
    if not npz_files:
        print(f"[{track}] [Error] Keine Score-Maps in {score_map_dir} — übersprungen.")
        return None

    n = len(npz_files)
    print(f"\n{'#' * 70}")
    print(f"#  {track}  ({n} Bilder)")
    print(f"{'#' * 70}")

    variant_names = ["A: Volles Bild", "B: Trapez", "C: Road",
                     "C: Road+SW",     "D: Neg. Filter"]
    method_names  = ["Energy", "DINOv2 kNN", "PixOOD", "RbA"]
    method_keys   = {
        "Energy":     "energy_map",
        "DINOv2 kNN": "knn_map",
        "PixOOD":     "pixood_map",
        "RbA":        "rba_map",
    }

    # === Score-Ranges aus Stichprobe (alle Bilder, da SMIYC klein) ===
    print(f"[{track}] Bestimme globale Score-Ranges ...")
    score_samples = {m: [] for m in method_names}
    for f in npz_files:
        d = np.load(f)
        for m in method_names:
            key = method_keys[m]
            if key in d.files:
                score_samples[m].append(d[key].astype(np.float32).flatten())

    score_ranges = {}
    for m in method_names:
        if not score_samples[m]:
            print(f"[{track}] [Warn] Keine Maps für {m} — übersprungen.")
            score_ranges[m] = None
            continue
        concat = np.concatenate(score_samples[m])
        margin = max(0.1, (concat.max() - concat.min()) * 0.05)
        score_ranges[m] = (float(concat.min()) - margin, float(concat.max()) + margin)
        print(f"[{track}] {m:12s} range: "
              f"[{score_ranges[m][0]:.4f}, {score_ranges[m][1]:.4f}]")
    del score_samples

    method_names = [m for m in method_names if score_ranges[m] is not None]

    aggs = {}
    for v in variant_names:
        aggs[v] = {}
        for m in method_names:
            smin, smax = score_ranges[m]
            aggs[v][m] = StreamingScoreAggregator(smin, smax, n_bins=n_bins)

    roi_stats    = {v: {"roi_pixels": 0, "ood_in_roi": 0} for v in variant_names}
    total_pixels = 0
    total_ood    = 0
    per_image_rows = []

    print(f"[{track}] Verarbeite {n} Bilder ...")
    for f in tqdm(npz_files, desc=f"{track}"):
        d = np.load(f)
        pred_class = d["pred_class"].astype(np.int32)
        msp_map    = d["msp_map"].astype(np.float32)
        ood_label  = d["ood_label"].astype(np.int32)
        H, W       = ood_label.shape

        total_pixels += H * W
        total_ood    += int(ood_label.sum())

        score_maps = {}
        for m in method_names:
            key = method_keys[m]
            if key in d.files:
                score_maps[m] = d[key].astype(np.float32)

        mask_a = np.ones((H, W), dtype=bool)
        mask_b = make_trapezoid_mask(H, W)
        mask_c_road    = (pred_class == 0)
        mask_c_road_sw = (pred_class == 0) | (pred_class == 1)

        bg_mask = np.zeros((H, W), dtype=bool)
        for tid in BG_TRAIN_IDS:
            bg_mask |= ((pred_class == tid) & (msp_map > msp_thresh))
        mask_d = ~bg_mask

        masks = {
            "A: Volles Bild": mask_a,
            "B: Trapez":      mask_b,
            "C: Road":        mask_c_road,
            "C: Road+SW":     mask_c_road_sw,
            "D: Neg. Filter": mask_d,
        }

        for vname, mask in masks.items():
            roi_stats[vname]["roi_pixels"] += int(mask.sum())
            roi_stats[vname]["ood_in_roi"] += int((ood_label[mask] == 1).sum())
            for mname, smap in score_maps.items():
                aggs[vname][mname].update(smap[mask], ood_label[mask])

        stem = f.stem
        for vname in ["A: Volles Bild", "D: Neg. Filter"]:
            mask = masks[vname]
            for mname, smap in score_maps.items():
                s = smap[mask]
                l = ood_label[mask]
                if l.sum() > 0 and (l == 0).sum() > 0:
                    img_auroc = roc_auc_score(l, s)
                else:
                    img_auroc = float("nan")
                per_image_rows.append({
                    "image":      stem,
                    "variant":    vname,
                    "method":     mname,
                    "auroc":      round(img_auroc, 4),
                    "ood_pixels": int(l.sum()),
                })

    # === Metriken ===
    print(f"\n{'=' * 90}")
    print(f"  {track} — ROI-Varianten ({n} Bilder)")
    print(f"{'=' * 90}")
    header = (f"{'Variante':<18} {'Methode':<14} {'AUROC':>7} {'AP':>8} "
              f"{'FPR95':>7} {'ROI %':>7} {'OoD-Ret.':>9}")
    print(header)
    print("-" * 90)

    result_rows = []
    for vname in variant_names:
        roi_pct = roi_stats[vname]["roi_pixels"] / total_pixels * 100
        ood_ret = roi_stats[vname]["ood_in_roi"] / max(total_ood, 1) * 100
        for mname in method_names:
            m = aggs[vname][mname].compute_metrics()
            print(f"  {vname:<18} {mname:<14} {m['auroc']:7.4f} {m['ap']:8.4f} "
                  f"{m['fpr95']:7.4f} {roi_pct:6.1f}% {ood_ret:7.1f}%")
            result_rows.append({
                "Variante":          vname,
                "Methode":           mname,
                "AUROC":             round(m["auroc"], 4),
                "AP":                round(m["ap"], 4),
                "FPR95":             round(m["fpr95"], 4),
                "ROI_pct":           round(roi_pct, 1),
                "OoD_Retention_pct": round(ood_ret, 1),
            })
        print()

    # === CSV ===
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "smiyc_results.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"[Saved] {csv_path}")

    # === LaTeX ===
    tex_path = output_dir / "smiyc_results.tex"
    with open(tex_path, "w") as ftex:
        ftex.write(f"% Auto-generated by evaluate_smiyc_variants.py ({track})\n")
        ftex.write("\\begin{table}[htbp]\n\\centering\n")
        ftex.write(f"\\caption{{OoD-Detektionsmetriken unter verschiedenen ROI-Varianten.\n")
        ftex.write(f"         SMIYC {track} (validation-Split, {n} Bilder, Vollbild-Basis).}}\n")
        ftex.write(f"\\label{{tab:smiyc_{track.lower()}}}\n")
        ftex.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        ftex.write("\\textbf{Variante} & \\textbf{Methode} & \\textbf{AUROC}$\\uparrow$")
        ftex.write(" & \\textbf{AP}$\\uparrow$ & \\textbf{FPR95}$\\downarrow$")
        ftex.write(" & \\textbf{ROI} & \\textbf{OoD-Ret.} \\\\\n\\midrule\n")
        prev_variant = None
        for row in result_rows:
            if prev_variant and row["Variante"] != prev_variant:
                ftex.write("\\midrule\n")
            prev_variant = row["Variante"]
            v = row["Variante"].replace("&", "\\&")
            ftex.write(f"{v} & {row['Methode']} & {row['AUROC']:.4f} & "
                       f"{row['AP']:.4f} & {row['FPR95']:.4f} & "
                       f"{row['ROI_pct']:.0f}\\,\\% & "
                       f"{row['OoD_Retention_pct']:.0f}\\,\\% \\\\\n")
        ftex.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[Saved] {tex_path}")

    # === Per-Image ===
    per_image_path = output_dir / "per_image_auroc.csv"
    with open(per_image_path, "w", newline="") as fpi:
        writer = csv.DictWriter(fpi, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"[Saved] {per_image_path}")

    print(f"\n=== {track}: Per-Image AUROC (A vs D) ===")
    if n <= 12:
        print(f"  [Hinweis] n={n} ist klein — Per-Image-Statistik nur grob interpretieren.")
    for mname in method_names:
        for vname in ["A: Volles Bild", "D: Neg. Filter"]:
            vals = [r["auroc"] for r in per_image_rows
                    if r["method"] == mname and r["variant"] == vname
                    and not np.isnan(r["auroc"])]
            if vals:
                arr = np.array(vals)
                print(f"  {mname:<14} {vname:<18} "
                      f"Mean={arr.mean():.4f}  Std={arr.std():.4f}  "
                      f"Min={arr.min():.4f}  Median={np.median(arr):.4f}  "
                      f"Max={arr.max():.4f}")

    return result_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msp_thresh", type=float, default=MSP_THRESHOLD)
    parser.add_argument("--n_bins", type=int, default=10_000)
    args = parser.parse_args()

    for track in TRACKS:
        evaluate_track(track, args.msp_thresh, args.n_bins)

    print(f"\n{'=' * 70}")
    print(f"Fertig. Ergebnisse in results/smiyc/<Track>/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
