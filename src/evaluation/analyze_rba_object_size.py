"""
analyze_rba_object_size.py
--------------------------
Untersucht, warum RbA auf Lost & Found einbricht, auf SMIYC RoadObstacle21 aber
nicht -- obwohl beide kleine OoD-Objekte enthalten.

Pro Bild werden berechnet:
  - RbA Per-Image AUROC (auf dem vollen Bild, Variante A)
  - groesstes OoD-Objekt (Pixel, via Connected Components)
  - OoD-Pixel gesamt, Objektanzahl, Bildflaeche, OoD-Anteil

Ausgabe:
  - results/rba_analysis/rba_size_<dataset>.csv  (eine Zeile pro Bild)
  - results/rba_analysis/rba_size_comparison.png (Streudiagramm L&F vs RO21)
  - Konsolen-Zusammenfassung: AUROC nach Objektgroessen-Quartilen

Vergleicht bewusst Lost & Found mit RoadObstacle21 (RoadAnomaly21 wurde aus den
in Kapitel 2 genannten Gruenden ausgeschlossen).

Aufruf (aus dem Repo-Root):
    python src/evaluation/analyze_rba_object_size.py
"""

import sys
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RESULTS_DIR, SCORE_MAPS_LAF, SMIYC_RESULTS_DIR, filter_noknown

OUTPUT_DIR = RESULTS_DIR / "rba_analysis"

DATASETS = {
    "Lost & Found":   SCORE_MAPS_LAF,
    "RoadObstacle21": SMIYC_RESULTS_DIR / "RoadObstacle21" / "score_maps",
}


def per_image_stats(npz_dir, dataset_name):
    rows = []
    files = sorted(npz_dir.glob("*.npz"))
    if "Found" in dataset_name:           # NoKnown-Filter nur fuer Lost & Found
        files, n_nk = filter_noknown(files)
        if n_nk:
            print(f"[NoKnown] {n_nk} Bilder mit bekannten Klassen entfernt ({dataset_name}).")
    if not files:
        print(f"[Warn] Keine Score-Maps in {npz_dir} -- {dataset_name} uebersprungen.")
        return rows

    for f in tqdm(files, desc=dataset_name):
        d = np.load(f)
        if "rba_map" not in d.files:
            continue
        rba = d["rba_map"].astype(np.float32)
        ood = (d["ood_label"].astype(np.int32) == 1)
        H, W = ood.shape
        n_ood = int(ood.sum())
        if n_ood == 0 or (~ood).sum() == 0:
            continue   # kein OoD oder kein InD -> AUROC undefiniert

        # Per-Image RbA AUROC (volles Bild)
        try:
            auroc = float(roc_auc_score(ood.flatten(), rba.flatten()))
        except Exception:
            continue

        # Objektgroessen via Connected Components
        labeled, n_obj = ndimage.label(ood)
        if n_obj > 0:
            sizes = ndimage.sum(ood, labeled, range(1, n_obj + 1))
            largest = int(sizes.max())
            mean_obj = float(sizes.mean())
        else:
            largest = mean_obj = 0

        rows.append({
            "dataset":        dataset_name,
            "image":          f.stem,
            "rba_auroc":      round(auroc, 4),
            "ood_pixels":     n_ood,
            "n_objects":      int(n_obj),
            "largest_object": largest,
            "mean_object":    round(mean_obj, 1),
            "image_pixels":   H * W,
            "ood_fraction":   round(n_ood / (H * W), 6),
        })
    return rows


def summarize(rows, name):
    """AUROC nach Quartilen der groessten Objektgroesse."""
    if not rows:
        return
    arr = sorted(rows, key=lambda r: r["largest_object"])
    aurocs = np.array([r["rba_auroc"] for r in arr])
    sizes  = np.array([r["largest_object"] for r in arr])
    print(f"\n=== {name}  (n={len(arr)}) ===")
    print(f"  RbA AUROC gesamt:        Mean={aurocs.mean():.4f}  Median={np.median(aurocs):.4f}")
    print(f"  Groesstes OoD-Objekt:    Median={np.median(sizes):.0f} px  "
          f"Min={sizes.min():.0f}  Max={sizes.max():.0f}")
    # Quartile nach Objektgroesse
    q = np.quantile(sizes, [0.25, 0.5, 0.75])
    bins = [("Q1 (kleinste)", sizes <= q[0]),
            ("Q2",            (sizes > q[0]) & (sizes <= q[1])),
            ("Q3",            (sizes > q[1]) & (sizes <= q[2])),
            ("Q4 (groesste)",  sizes > q[2])]
    print(f"  RbA AUROC nach Objektgroessen-Quartil:")
    for label, mask in bins:
        if mask.any():
            print(f"    {label:<16} AUROC={aurocs[mask].mean():.4f}  "
                  f"(Objektgroesse {sizes[mask].min():.0f}-{sizes[mask].max():.0f} px)")
    # Korrelation Groesse <-> AUROC
    if len(arr) > 2:
        r = np.corrcoef(np.log10(sizes + 1), aurocs)[0, 1]
        print(f"  Korrelation log(Objektgroesse) <-> AUROC: r = {r:+.3f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = {}

    for name, npz_dir in DATASETS.items():
        rows = per_image_stats(npz_dir, name)
        all_rows[name] = rows
        if rows:
            key = "laf" if "Found" in name else "ro21"
            csv_path = OUTPUT_DIR / f"rba_size_{key}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"[Saved] {csv_path}")

    for name, rows in all_rows.items():
        summarize(rows, name)

    # --- Streudiagramm: Objektgroesse (log) vs RbA AUROC ---
    plt.figure(figsize=(9, 6))
    colors = {"Lost & Found": "#c0392b", "RoadObstacle21": "#2471a3"}
    for name, rows in all_rows.items():
        if not rows:
            continue
        x = [r["largest_object"] for r in rows]
        y = [r["rba_auroc"] for r in rows]
        plt.scatter(x, y, s=18, alpha=0.5, label=f"{name} (n={len(rows)})",
                    color=colors.get(name, "gray"), edgecolors="none")
    plt.xscale("log")
    plt.xlabel("Größtes OoD-Objekt pro Bild [Pixel, log]")
    plt.ylabel("RbA Per-Image AUROC")
    plt.title("RbA-Performance vs. OoD-Objektgröße")
    plt.axhline(0.5, color="gray", ls=":", lw=1, label="Zufall (AUROC=0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "rba_size_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {fig_path}")


if __name__ == "__main__":
    main()
