"""
evaluate_roi_variants.py
------------------------
Wertet die ROI-Varianten (A-D) auf den vorab berechneten Score-Maps aus.
Methoden: Energy, DINOv2 kNN, PixOOD, RbA.

Voraussetzung: compute_score_maps.py wurde ausgefuehrt UND
               merge_rba_into_score_maps.py wurde ausgefuehrt.

Aufruf:
    python evaluate_roi_variants.py
    python evaluate_roi_variants.py --msp_thresh 0.95

Output:
    results/roi_variants/roi_results.csv
    results/roi_variants/roi_results.tex
    results/roi_variants/per_image_auroc.csv
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
from paths import ROI_VARIANTS_DIR, SCORE_MAPS_LAF, filter_noknown

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR    = ROI_VARIANTS_DIR
SCORE_MAP_DIR = SCORE_MAPS_LAF

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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msp_thresh", type=float, default=MSP_THRESHOLD)
    parser.add_argument("--n_bins", type=int, default=10_000)
    args = parser.parse_args()

    npz_files = sorted(SCORE_MAP_DIR.glob("*.npz"))
    npz_files, _n_noknown = filter_noknown(npz_files)
    if _n_noknown:
        print(f"[NoKnown] {_n_noknown} Bilder mit bekannten Klassen "
              f"(Kinder/Fahrraeder) aus der Auswertung entfernt.")
    if not npz_files:
        print(f"[Error] Keine Score-Maps in {SCORE_MAP_DIR} gefunden.")
        sys.exit(1)

    n = len(npz_files)
    print(f"[Score-Maps] {n} Dateien gefunden in {SCORE_MAP_DIR}")

    variant_names = ["A: Volles Bild", "B: Trapez", "C: Road",
                     "C: Road+SW",     "D: Neg. Filter"]
    method_names  = ["Energy", "DINOv2 kNN", "PixOOD", "RbA"]
    method_keys   = {
        "Energy":     "energy_map",
        "DINOv2 kNN": "knn_map",
        "PixOOD":     "pixood_map",
        "RbA":        "rba_map",
    }

    # === Score-Ranges aus Stichprobe ===
    print("[Init] Bestimme globale Score-Ranges aus Stichprobe ...")
    sample_size = min(50, n)
    score_samples = {m: [] for m in method_names}

    for f in npz_files[:sample_size]:
        d = np.load(f)
        for m in method_names:
            key = method_keys[m]
            if key in d.files:
                score_samples[m].append(d[key].astype(np.float32).flatten())

    score_ranges = {}
    for m in method_names:
        if not score_samples[m]:
            print(f"[Warn] Keine Score-Maps fuer {m} gefunden -- wird uebersprungen.")
            score_ranges[m] = None
            continue
        concat = np.concatenate(score_samples[m])
        margin = max(0.1, (concat.max() - concat.min()) * 0.05)
        score_ranges[m] = (float(concat.min()) - margin, float(concat.max()) + margin)
        print(f"[Init] {m:12s} range: [{score_ranges[m][0]:.4f}, {score_ranges[m][1]:.4f}]")

    method_names = [m for m in method_names if score_ranges[m] is not None]

    aggs = {}
    for v in variant_names:
        aggs[v] = {}
        for m in method_names:
            smin, smax = score_ranges[m]
            aggs[v][m] = StreamingScoreAggregator(smin, smax, n_bins=args.n_bins)

    roi_stats    = {v: {"roi_pixels": 0, "ood_in_roi": 0} for v in variant_names}
    total_pixels = 0
    total_ood    = 0
    per_image_rows = []

    # === Main loop ===
    print(f"\n[Eval] Verarbeite {n} Bilder ...")
    for f in tqdm(npz_files, desc="ROI-Auswertung"):
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
            bg_mask |= ((pred_class == tid) & (msp_map > args.msp_thresh))
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
    print("\n" + "=" * 90)
    print(f"  Kapitel 4 -- ROI-Varianten Ergebnisse ({n} Bilder)")
    print("=" * 90)
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
    csv_path = OUTPUT_DIR / "roi_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"[Saved] {csv_path}")

    # === LaTeX ===
    tex_path = OUTPUT_DIR / "roi_results.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by evaluate_roi_variants.py\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{OoD-Detektionsmetriken unter verschiedenen ROI-Varianten.\n")
        f.write("         Lost~\\&~Found test-split, " + str(n) + " Bilder.}\n")
        f.write("\\label{tab:roi_results}\n")
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("\\textbf{Variante} & \\textbf{Methode} & \\textbf{AUROC}$\\uparrow$")
        f.write(" & \\textbf{AP}$\\uparrow$ & \\textbf{FPR95}$\\downarrow$")
        f.write(" & \\textbf{ROI} & \\textbf{OoD-Ret.} \\\\\n\\midrule\n")

        prev_variant = None
        for row in result_rows:
            if prev_variant and row["Variante"] != prev_variant:
                f.write("\\midrule\n")
            prev_variant = row["Variante"]
            v = row["Variante"].replace("&", "\\&")
            f.write(f"{v} & {row['Methode']} & {row['AUROC']:.4f} & "
                    f"{row['AP']:.4f} & {row['FPR95']:.4f} & "
                    f"{row['ROI_pct']:.0f}\\,\\% & {row['OoD_Retention_pct']:.0f}\\,\\% \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[Saved] {tex_path}")

    # === Per-Image ===
    per_image_path = OUTPUT_DIR / "per_image_auroc.csv"
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"[Saved] {per_image_path}")

    print("\n=== Per-Image AUROC (Variante A vs D) ===")
    per_image_stats = []
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
                per_image_stats.append({
                    "Methode": mname,
                    "Variante": vname,
                    "Mean":   round(float(arr.mean()), 4),
                    "Std":    round(float(arr.std()), 4),
                    "Min":    round(float(arr.min()), 4),
                    "Median": round(float(np.median(arr)), 4),
                    "Max":    round(float(arr.max()), 4),
                    "N":      int(arr.size),
                })

    # === Tabelle 6: Verteilungsstatistik als Datei ===
    if per_image_stats:
        stats_csv = OUTPUT_DIR / "per_image_auroc_stats.csv"
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_stats[0].keys())
            writer.writeheader()
            writer.writerows(per_image_stats)
        print(f"[Saved] {stats_csv}")

        stats_tex = OUTPUT_DIR / "per_image_auroc_stats.tex"
        with open(stats_tex, "w", encoding="utf-8") as f:
            f.write("% Auto-generated by evaluate_roi_variants.py (Tabelle 6)\n")
            f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
            f.write("Methode & Variante & Mean & Std & Min & Median \\\\\n\\midrule\n")
            prev = None
            for r in per_image_stats:
                if prev is not None and r["Methode"] != prev:
                    f.write("\\midrule\n")
                prev = r["Methode"]
                f.write(f"{r['Methode']} & {r['Variante']} & {r['Mean']:.3f} & "
                        f"{r['Std']:.3f} & {r['Min']:.3f} & {r['Median']:.3f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"[Saved] {stats_tex}")

    print(f"\n{'=' * 60}")
    print(f"Fertig. Ergebnisse in: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
