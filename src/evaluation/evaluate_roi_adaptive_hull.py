"""
evaluate_roi_adaptive_hull.py
-----------------------------
Abschluss-Experiment: zusaetzliche ROI-Variante als kurze Alternative.

  Variante E -- "Road-adaptive Huelle" (gebogenes, road-folgendes Trapez)

Verbindet die Staerken von Variante B (Trapez: geschlossene, lochfreie Region)
und Variante C (Road-ROI: bildabhaengig, folgt Kurven). Basis ist hier fest
Road+Sidewalk (trainId 0+1); die Margins sind fest auf 0.08 (seitlich) und
0.05 (vertikal) gesetzt.

Konstruktion pro Bild:
  1. Road+Sidewalk-Maske aus der Modell-Segmentierung als Ausgangsbasis.
  2. Fuer jede Bildzeile den linken/rechten Rand bestimmen, um margin_x nach
     aussen erweitern und die Zeile durchgehend fuellen -> schliesst alle
     Loecher (z.B. Objekte auf der Strasse), anders als reines C.
  3. Vertikal um margin_y nach oben/unten erweitern.

Aendert keine bestehenden Tabellen. Liest dieselben Score-Maps,
schreibt nach results/roi_adaptive_hull/.

Aufruf (aus dem Repo-Root):
    python src/evaluation/evaluate_roi_adaptive_hull.py
"""

import csv
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR, SCORE_MAPS_LAF, filter_noknown

OUTPUT_DIR    = RESULTS_DIR / "roi_adaptive_hull"
SCORE_MAP_DIR = SCORE_MAPS_LAF

# Feste Parameter der Variante E
MARGIN_X = 0.08   # seitliche Erweiterung pro Zeile (Anteil der Bildbreite)
MARGIN_Y = 0.05   # vertikale Erweiterung (Anteil der Bildhoehe)
N_BINS   = 10_000

# Festes Trapez (identisch zu evaluate_roi_variants.py) -- nur fuer den Vergleich.
TRAP_TOP_Y, TRAP_BOT_Y = 0.28, 0.90
TRAP_TL_X,  TRAP_TR_X  = 0.38, 0.62
TRAP_BL_X,  TRAP_BR_X  = 0.05, 0.95


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


def make_adaptive_hull(road_mask, margin_x=MARGIN_X, margin_y=MARGIN_Y):
    """Road-folgende, lochfreie Huelle um die Road(+SW)-Maske."""
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
        hull[y, left:right] = True   # durchgehend fuellen -> keine Loecher
        rows.append(y)

    if rows:
        my = int(round(margin_y * H))
        y_top, y_bot = rows[0], rows[-1]
        for y in range(max(0, y_top - my), y_top):
            hull[y] = hull[y_top]
        for y in range(y_bot + 1, min(H, y_bot + 1 + my)):
            hull[y] = hull[y_bot]

    return hull


class StreamingScoreAggregator:
    """Identisch zu evaluate_roi_variants.py -- fuer vergleichbare Zahlen."""
    def __init__(self, score_min, score_max, n_bins=N_BINS):
        self.bin_edges = np.linspace(score_min, score_max, n_bins + 1)
        self.hist_pos  = np.zeros(n_bins, dtype=np.int64)
        self.hist_neg  = np.zeros(n_bins, dtype=np.int64)
        self.n_pos = 0
        self.n_neg = 0

    def update(self, scores, labels):
        if len(scores) == 0:
            return
        # Werte ausserhalb des Bin-Bereichs in die Randbins klemmen, damit
        # np.histogram keine Pixel stillschweigend verwirft (sonst kann die
        # TPR die 0.95-Schwelle verfehlen -> FPR95 faellt faelschlich auf 1.0).
        lo, hi = self.bin_edges[0], self.bin_edges[-1]
        scores = np.clip(scores, lo, hi)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.size:
            self.hist_pos += np.histogram(pos, bins=self.bin_edges)[0]
            self.n_pos    += pos.size
        if neg.size:
            self.hist_neg += np.histogram(neg, bins=self.bin_edges)[0]
            self.n_neg    += neg.size

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    method_names = ["Energy", "DINOv2 kNN", "PixOOD", "RbA"]
    key_of = {"Energy": "energy_map", "DINOv2 kNN": "knn_map",
              "PixOOD": "pixood_map", "RbA": "rba_map"}

    npz_files = sorted(SCORE_MAP_DIR.glob("*.npz"))
    npz_files, _n_noknown = filter_noknown(npz_files)
    if _n_noknown:
        print(f"[NoKnown] {_n_noknown} Bilder mit bekannten Klassen "
              f"(Kinder/Fahrraeder) aus der Auswertung entfernt.")
    if not npz_files:
        print(f"[Error] Keine Score-Maps in {SCORE_MAP_DIR}. Erst compute_score_maps.py "
              f"+ merge_rba_into_score_maps.py ausfuehren (oder Caches per "
              f"download_score_maps.py laden).")
        sys.exit(1)
    n = len(npz_files)
    print(f"[Score-Maps] {n} Dateien in {SCORE_MAP_DIR}")

    # --- Globale Score-Ranges aus Stichprobe (wie im Hauptskript) ---
    print("[Init] Bestimme globale Score-Ranges aus Stichprobe ...")
    score_samples = {m: [] for m in method_names}
    for f in npz_files[:min(50, n)]:
        d = np.load(f)
        for m in method_names:
            key = key_of[m]
            if key in d.files:
                score_samples[m].append(d[key].astype(np.float32).flatten())

    score_ranges = {}
    for m in method_names:
        if not score_samples[m]:
            print(f"[Init] {m:12s} -- keine Daten (uebersprungen, evtl. RbA nicht gemergt)")
            score_ranges[m] = None
            continue
        concat = np.concatenate(score_samples[m])
        margin = max(0.1, (concat.max() - concat.min()) * 0.05)
        score_ranges[m] = (float(concat.min()) - margin, float(concat.max()) + margin)
        print(f"[Init] {m:12s} range: [{score_ranges[m][0]:.4f}, {score_ranges[m][1]:.4f}]")
    method_names = [m for m in method_names if score_ranges[m] is not None]

    # --- Varianten: C: Road+SW (Basis), B (festes Trapez), E (adaptive Huelle) ---
    base_name = "C: Road+SW"
    variant_names = [base_name, "B: Trapez", "E: Adaptive Huelle"]
    aggs = {v: {m: StreamingScoreAggregator(*score_ranges[m], n_bins=N_BINS)
                for m in method_names} for v in variant_names}
    roi_stats = {v: {"roi_pixels": 0, "ood_in_roi": 0} for v in variant_names}
    total_pixels = 0
    total_ood    = 0

    print(f"\n[Eval] Verarbeite {n} Bilder "
          f"(Basis=Road+SW, margin_x={MARGIN_X}, margin_y={MARGIN_Y}) ...")
    for f in tqdm(npz_files, desc="Adaptive-Huelle"):
        d = np.load(f)
        pred_class = d["pred_class"].astype(np.int32)
        ood_label  = d["ood_label"].astype(np.int32)
        H, W       = ood_label.shape
        total_pixels += H * W
        total_ood    += int(ood_label.sum())

        score_maps = {m: d[key_of[m]].astype(np.float32) for m in method_names}

        road_mask = (pred_class == 0) | (pred_class == 1)   # Road + Sidewalk

        masks = {
            base_name:             road_mask,
            "B: Trapez":           make_trapezoid_mask(H, W),
            "E: Adaptive Huelle":  make_adaptive_hull(road_mask),
        }

        for vname, mask in masks.items():
            roi_stats[vname]["roi_pixels"] += int(mask.sum())
            roi_stats[vname]["ood_in_roi"] += int((ood_label[mask] == 1).sum())
            for mname, smap in score_maps.items():
                aggs[vname][mname].update(smap[mask], ood_label[mask])

    # --- Ausgabe ---
    print("\n" + "=" * 90)
    print(f"  Abschluss-Experiment -- ROI 'Adaptive Huelle' (Road+SW, {n} Bilder)")
    print("=" * 90)
    print(f"{'Variante':<22} {'Methode':<14} {'AUROC':>7} {'AP':>8} "
          f"{'FPR95':>7} {'ROI %':>7} {'OoD-Ret.':>9}")
    print("-" * 90)

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
                "AUROC": round(m["auroc"], 4), "AP": round(m["ap"], 4),
                "FPR95": round(m["fpr95"], 4),
                "ROI_pct": round(roi_pct, 1),
                "OoD_Retention_pct": round(ood_ret, 1),
            })
        print()

    csv_path = OUTPUT_DIR / "adaptive_hull_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        w.writeheader()
        w.writerows(result_rows)
    print(f"[Saved] {csv_path}")

    tex_path = OUTPUT_DIR / "adaptive_hull_results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by evaluate_roi_adaptive_hull.py\n")
        f.write(f"% Basis=Road+SW, margin_x={MARGIN_X}, margin_y={MARGIN_Y}\n")
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("\\textbf{Variante} & \\textbf{Methode} & \\textbf{AUROC}$\\uparrow$"
                " & \\textbf{AP}$\\uparrow$ & \\textbf{FPR95}$\\downarrow$"
                " & \\textbf{ROI} & \\textbf{OoD-Ret.} \\\\\n\\midrule\n")
        prev = None
        for r in result_rows:
            if prev and r["Variante"] != prev:
                f.write("\\midrule\n")
            prev = r["Variante"]
            v = r["Variante"].replace("&", "\\&")
            f.write(f"{v} & {r['Methode']} & {r['AUROC']:.4f} & {r['AP']:.4f} & "
                    f"{r['FPR95']:.4f} & {r['ROI_pct']:.0f}\\,\\% & "
                    f"{r['OoD_Retention_pct']:.0f}\\,\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[Saved] {tex_path}")

    print("\nHinweis: B und C: Road+SW dienen hier nur als Vergleich; ihre Werte sind\n"
          "identisch zu evaluate_roi_variants.py. Neu ist Variante E.")


if __name__ == "__main__":
    main()
