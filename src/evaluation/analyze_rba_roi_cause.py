"""
analyze_rba_roi_cause.py
------------------------
Beantwortet zwei Fragen zur RbA-Schwaeche auf Lost & Found:

  TEST 1  (Groessenkontrollierter Vergleich)
      Liegt der RbA-Unterschied L&F vs. RO21 NUR an der Objektgroesse
      (und die L&F-Mittelwerte werden von vielen kleinen Objekten gedrueckt),
      oder bleibt L&F auch bei GLEICHER absoluter Objektgroesse schlechter?
      -> Beide Datensaetze werden in gemeinsame absolute Pixel-Bins einsortiert,
         und die RbA-AUROC wird Bin fuer Bin gegenuebergestellt.

  TEST 2  (Fragmentierungs-Test)
      Bricht RbA auf L&F unter Road-ROI ein, weil der SegFormer-Road-ROI das
      OoD-OBJEKT selbst zerschneidet (nicht den Hintergrund)?
      -> Pro Bild: OoD-Retention unter Road-ROI, Fragmentanzahl, und der
         AUROC-Einbruch (Trapez B  minus  Road-ROI C) werden korreliert.

Datengrundlage: die vorhandenen Score-Maps (.npz) mit den keys
    rba_map, ood_label, pred_class, shape
Es wird NICHTS neu inferiert; alles laeuft auf dem Cache.

Aufruf (aus dem Repo-Root ODER aus src/evaluation/):
    python analyze_rba_roi_cause.py

Ausgabe:
    results/rba_analysis/test1_size_controlled.csv
    results/rba_analysis/test1_size_controlled.png
    results/rba_analysis/test2_fragmentation_laf.csv
    results/rba_analysis/test2_fragmentation_laf.png
    + Konsolen-Zusammenfassung beider Tests
"""

import sys
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# --- Pfade aus dem Repo holen (funktioniert aus Root und aus src/evaluation/) -
HERE = Path(__file__).resolve()
for cand in (HERE.parent / "src", HERE.parents[1] if len(HERE.parents) > 1 else HERE.parent):
    if (cand / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break
from paths import RESULTS_DIR, SCORE_MAPS_LAF, SMIYC_RESULTS_DIR  # noqa: E402

OUTPUT_DIR = RESULTS_DIR / "rba_analysis"

DATASETS = {
    "Lost & Found":   SCORE_MAPS_LAF,
    "RoadObstacle21": SMIYC_RESULTS_DIR / "RoadObstacle21" / "score_maps",
}

# trainId-Encoding in pred_class:  0 = road, 1 = sidewalk
ROAD_ID = 0
SIDEWALK_ID = 1

# --- Variante B: EXAKTES Trapez aus eval_config.yaml (wie make_roi_figure.py) ---
TRAPEZOID_REL = {
    "y_top": 0.28, "x_top_l": 0.38, "x_top_r": 0.62,
    "y_bot": 0.90, "x_bot_l": 0.05, "x_bot_r": 0.95,
}
# --- Variante D: Negativ-Filter (wie im Repo) ---
BACKGROUND_IDS = [2, 3, 4, 8, 10]   # building, wall, fence, vegetation, sky
MSP_THRESHOLD = 0.95


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================
def safe_auroc(scores, labels):
    """AUROC nur, wenn beide Klassen vorhanden sind, sonst NaN."""
    labels = labels.astype(np.int32)
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return float("nan")
    try:
        return float(roc_auc_score(labels.ravel(), scores.ravel()))
    except Exception:
        return float("nan")


def trapezoid_mask(H, W):
    """
    Variante B: EXAKTES festes Trapez aus eval_config.yaml, identisch zu
    make_roi_figure.py / visualize_roi_variants.py im Repo (cv2.fillPoly).
    """
    t = TRAPEZOID_REL
    pts = np.array([
        [int(t["x_top_l"] * W), int(t["y_top"] * H)],
        [int(t["x_top_r"] * W), int(t["y_top"] * H)],
        [int(t["x_bot_r"] * W), int(t["y_bot"] * H)],
        [int(t["x_bot_l"] * W), int(t["y_bot"] * H)],
    ], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    if HAVE_CV2:
        cv2.fillPoly(mask, [pts], 1)
    else:
        # Fallback ohne cv2: zeilenweise zwischen den interpolierten Kanten fuellen
        yt, yb = int(t["y_top"] * H), int(t["y_bot"] * H)
        for y in range(yt, min(yb, H)):
            f = (y - yt) / max(yb - yt, 1)
            xl = int((t["x_top_l"] + f * (t["x_bot_l"] - t["x_top_l"])) * W)
            xr = int((t["x_top_r"] + f * (t["x_bot_r"] - t["x_top_r"])) * W)
            mask[y, max(xl, 0):min(xr, W)] = 1
    return mask.astype(bool)


def negative_filter_mask(pred, msp):
    """Variante D: schliesst konfident klassifizierten Hintergrund aus."""
    bg = np.zeros(pred.shape, dtype=bool)
    for c in BACKGROUND_IDS:
        bg |= ((pred == c) & (msp > MSP_THRESHOLD))
    return ~bg


def write_csv_robust(path, rows):
    """CSV-Writer, der mit ueber Zeilen variierenden Spalten klarkommt."""
    if not rows:
        return
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ===========================================================================
# Daten einlesen: pro Bild alle benoetigten Groessen
# ===========================================================================
def load_per_image(npz_dir, dataset_name):
    rows = []
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        print(f"[Warn] Keine Score-Maps in {npz_dir} -- {dataset_name} uebersprungen.")
        return rows

    for f in tqdm(files, desc=dataset_name):
        d = np.load(f)
        if "rba_map" not in d.files or "ood_label" not in d.files:
            continue
        rba = d["rba_map"].astype(np.float32)
        ood = (d["ood_label"].astype(np.int32) == 1)
        pred = d["pred_class"].astype(np.int32) if "pred_class" in d.files else None
        msp = d["msp_map"].astype(np.float32) if "msp_map" in d.files else None
        H, W = ood.shape
        n_ood = int(ood.sum())
        if n_ood == 0 or (~ood).sum() == 0:
            continue

        # --- Objektgroessen (Connected Components auf dem OoD-Label) ---
        labeled, n_obj = ndimage.label(ood)
        sizes = ndimage.sum(ood, labeled, range(1, n_obj + 1)) if n_obj else np.array([])
        largest = int(sizes.max()) if sizes.size else 0

        # --- AUROC volles Bild (Variante A) ---
        auroc_full = safe_auroc(rba, ood)

        # --- AUROC unter festem Trapez (Variante B), modellunabhaengig ---
        mB = trapezoid_mask(H, W)
        auroc_B = safe_auroc(rba[mB], ood[mB])

        row = {
            "dataset":        dataset_name,
            "image":          f.stem,
            "rba_auroc_full": round(auroc_full, 4),
            "rba_auroc_trap": round(auroc_B, 4),
            "ood_pixels":     n_ood,
            "n_objects":      int(n_obj),
            "largest_object": largest,
            "image_pixels":   H * W,
        }

        # --- Road-ROI (Variante C) nur wenn pred_class vorhanden ---
        if pred is not None:
            road = (pred == ROAD_ID)
            roadsw = np.isin(pred, [ROAD_ID, SIDEWALK_ID])

            # AUROC unter Road-ROI
            auroc_C = safe_auroc(rba[road], ood[road])
            row["rba_auroc_road"] = round(auroc_C, 4)

            # OoD-Retention: welcher Anteil der OoD-Pixel ueberlebt den Road-ROI?
            ood_in_road = int((ood & road).sum())
            row["ood_retention_road"] = round(ood_in_road / max(n_ood, 1), 4)
            ood_in_roadsw = int((ood & roadsw).sum())
            row["ood_retention_roadsw"] = round(ood_in_roadsw / max(n_ood, 1), 4)

            # Fragmentierung: in wie viele Stuecke zerfaellt das OoD nach Road-ROI?
            ood_after = ood & road
            lab_after, n_frag = ndimage.label(ood_after)
            row["n_frag_after_road"] = int(n_frag)
            # Verhaeltnis Fragmente / urspruengliche Objekte (1.0 = keine Zersplitterung)
            row["frag_ratio"] = round(n_frag / max(n_obj, 1), 3)
            # AUROC-Einbruch Trapez -> Road (positiv = Road ist schlechter)
            if not np.isnan(auroc_B) and not np.isnan(auroc_C):
                row["auroc_drop_B_minus_C"] = round(auroc_B - auroc_C, 4)
            else:
                row["auroc_drop_B_minus_C"] = float("nan")

            # --- Negativ-Filter (Variante D), falls msp vorhanden ---
            if msp is not None:
                mD = negative_filter_mask(pred, msp)
                auroc_D = safe_auroc(rba[mD], ood[mD])
                row["rba_auroc_negfilter"] = round(auroc_D, 4)
                row["ood_retention_negfilter"] = round(int((ood & mD).sum()) / max(n_ood, 1), 4)
                if not np.isnan(auroc_B) and not np.isnan(auroc_D):
                    row["auroc_drop_B_minus_D"] = round(auroc_B - auroc_D, 4)
                else:
                    row["auroc_drop_B_minus_D"] = float("nan")

        rows.append(row)
    return rows


# ===========================================================================
# TEST 1: Groessenkontrollierter Vergleich
# ===========================================================================
def test1_size_controlled(all_rows):
    print("\n" + "=" * 78)
    print("  TEST 1 — Groessenkontrollierter Vergleich (gemeinsame absolute Bins)")
    print("=" * 78)

    laf = [r for r in all_rows["Lost & Found"]]
    ro21 = [r for r in all_rows["RoadObstacle21"]]
    if not laf or not ro21:
        print("  [Abbruch] Mindestens ein Datensatz ist leer.")
        return []

    laf_sizes = np.array([r["largest_object"] for r in laf])
    ro21_sizes = np.array([r["largest_object"] for r in ro21])

    # Gemeinsame Bins auf Basis des UEBERLAPPENDEN Groessenbereichs.
    lo = max(laf_sizes.min(), ro21_sizes.min())
    hi = min(laf_sizes.max(), ro21_sizes.max())
    print(f"  L&F Groessen:  min={laf_sizes.min():.0f}  median={np.median(laf_sizes):.0f}  max={laf_sizes.max():.0f} px")
    print(f"  RO21 Groessen: min={ro21_sizes.min():.0f}  median={np.median(ro21_sizes):.0f}  max={ro21_sizes.max():.0f} px")
    print(f"  Ueberlappender Bereich: {lo:.0f} .. {hi:.0f} px")
    if hi <= lo:
        print("  [Warn] Kein ueberlappender Groessenbereich -> Bin-Vergleich nicht moeglich.")
        print("         (Die Kontrollzeile unten funktioniert trotzdem.)")
        out_rows = []
    else:
        # Bin-Kanten: bevorzugt an gemeinsamen Quantilen im Ueberlappungsbereich,
        # damit die Bins auch bei wenigen Punkten gefuellt sind.
        both = np.concatenate([laf_sizes[(laf_sizes >= lo) & (laf_sizes <= hi)],
                               ro21_sizes[(ro21_sizes >= lo) & (ro21_sizes <= hi)]])
        n_bins = min(4, max(2, len(both) // 6))   # 2..4 Bins je nach Datenmenge
        edges = np.quantile(both, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)                  # doppelte Kanten entfernen
        out_rows = []
        print(f"\n  {'Bin (px)':<22}{'L&F AUROC (n)':<22}{'RO21 AUROC (n)':<22}{'Diff':<8}")
        print("  " + "-" * 70)
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i + 1]
            last = (i == len(edges) - 2)
            lmask = (laf_sizes >= a) & ((laf_sizes <= b) if last else (laf_sizes < b))
            rmask = (ro21_sizes >= a) & ((ro21_sizes <= b) if last else (ro21_sizes < b))
            l_vals = [laf[j]["rba_auroc_full"] for j in np.where(lmask)[0]]
            r_vals = [ro21[j]["rba_auroc_full"] for j in np.where(rmask)[0]]
            l_aur = np.nanmean(l_vals) if len(l_vals) else float("nan")
            r_aur = np.nanmean(r_vals) if len(r_vals) else float("nan")
            diff = (l_aur - r_aur) if (not np.isnan(l_aur) and not np.isnan(r_aur)) else float("nan")
            binlabel = f"{a:.0f}-{b:.0f}"
            print(f"  {binlabel:<22}"
                  f"{(f'{l_aur:.3f} (n={lmask.sum()})' if not np.isnan(l_aur) else f'-- (n={lmask.sum()})'):<22}"
                  f"{(f'{r_aur:.3f} (n={rmask.sum()})' if not np.isnan(r_aur) else f'-- (n={rmask.sum()})'):<22}"
                  f"{(f'{diff:+.3f}' if not np.isnan(diff) else 'n/a'):<8}")
            out_rows.append({
                "bin_lo_px": int(a), "bin_hi_px": int(b),
                "laf_auroc": round(l_aur, 4) if not np.isnan(l_aur) else "",
                "laf_n": int(lmask.sum()),
                "ro21_auroc": round(r_aur, 4) if not np.isnan(r_aur) else "",
                "ro21_n": int(rmask.sum()),
                "diff": round(diff, 4) if not np.isnan(diff) else "",
            })
        # Interpretationshilfe
        diffs = [r["diff"] for r in out_rows if r["diff"] != ""]
        if diffs:
            md = float(np.mean(diffs))
            print(f"\n  -> Mittlere Differenz (L&F - RO21) ueber gefuellte Bins: {md:+.3f}")
            if md < -0.03:
                print("     L&F bleibt bei GLEICHER Groesse schlechter -> Groesse allein erklaert es NICHT.")
            elif abs(md) <= 0.03:
                print("     Bei gleicher Groesse aehnlich -> Unterschied ist GROESSENGETRIEBEN.")

    # zusaetzlich: die in der Arbeit genannte 2088-px-Schwelle
    thr = float(np.median(ro21_sizes))  # RO21-Median als faire Schwelle
    l_big = [r["rba_auroc_full"] for r in laf if r["largest_object"] > thr]
    r_all = [r["rba_auroc_full"] for r in ro21]
    print("\n  Kontrollzeile (groessenkontrolliert ueber RO21-Median):")
    print(f"    Schwelle = RO21-Median = {thr:.0f} px")
    if l_big:
        print(f"    L&F, groesstes Objekt > {thr:.0f} px:  "
              f"AUROC mean = {np.nanmean(l_big):.3f}  (n={len(l_big)})")
    print(f"    RO21 (alle, Median-Groesse {thr:.0f} px): "
          f"AUROC mean = {np.nanmean(r_all):.3f}  (n={len(r_all)})")

    # --- Plot: AUROC vs. Groesse, beide Datensaetze, gleiche x-Achse ---
    plt.figure(figsize=(9, 6))
    for name, rows, color in [("Lost & Found", laf, "#c0392b"),
                              ("RoadObstacle21", ro21, "#2471a3")]:
        x = [r["largest_object"] for r in rows]
        y = [r["rba_auroc_full"] for r in rows]
        plt.scatter(x, y, s=18, alpha=0.45, color=color, edgecolors="none",
                    label=f"{name} (n={len(rows)})")
    plt.axvspan(lo, hi, color="green", alpha=0.06, label="overlap range")
    plt.xscale("log")
    plt.xlabel("Largest OoD object per image [px, log]")
    plt.ylabel("RbA per-image AUROC")
    plt.title("Test 1: RbA AUROC vs. object size (common absolute scale)")
    plt.axhline(0.5, color="gray", ls=":", lw=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUTPUT_DIR / "test1_size_controlled.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {p}")
    return out_rows


# ===========================================================================
# TEST 2: Fragmentierungs-Test (nur Lost & Found)
# ===========================================================================
def test2_fragmentation(laf_rows):
    print("\n" + "=" * 78)
    print("  TEST 2 — Fragmentierungs-Test (Lost & Found, Road-ROI vs. Trapez)")
    print("=" * 78)

    rows = [r for r in laf_rows if "auroc_drop_B_minus_C" in r
            and not (isinstance(r["auroc_drop_B_minus_C"], float)
                     and np.isnan(r["auroc_drop_B_minus_C"]))]
    if not rows:
        print("  [Abbruch] Keine verwertbaren Zeilen (pred_class fehlt?).")
        return

    retention = np.array([r["ood_retention_road"] for r in rows])
    frag = np.array([r["n_frag_after_road"] for r in rows])
    drop = np.array([r["auroc_drop_B_minus_C"] for r in rows])
    sizes = np.array([r["largest_object"] for r in rows])

    print(f"  Bilder mit gueltigem B- und C-AUROC: n = {len(rows)}")
    print(f"  OoD-Retention unter Road-ROI:   Median = {np.median(retention):.2f}  "
          f"Mean = {retention.mean():.2f}")
    print(f"  AUROC-Einbruch (B - C):         Median = {np.median(drop):+.3f}  "
          f"Mean = {drop.mean():+.3f}")

    # Korrelationen: Was sagt den AUROC-Einbruch am besten vorher?
    def corr(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)      # NaN-Paare rauswerfen
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def fmt_r(r, expectation):
        if np.isnan(r):
            return f"r = n/a  ({expectation}; zu wenig Varianz in den Daten)"
        return f"r = {r:+.3f}  ({expectation})"

    print("\n  Korrelationen mit dem AUROC-Einbruch (B - C):")
    print(f"    OoD-Retention      vs. Einbruch:  "
          f"{fmt_r(corr(retention, drop), 'erwartet NEGATIV: weniger Retention -> groesserer Einbruch')}")
    print(f"    Fragmentanzahl     vs. Einbruch:  "
          f"{fmt_r(corr(frag, drop), 'erwartet POSITIV: mehr Fragmente -> groesserer Einbruch')}")
    print(f"    log(Objektgroesse) vs. Einbruch:  "
          f"{fmt_r(corr(np.log10(sizes + 1), drop), 'erwartet NEGATIV: groessere Objekte -> kleinerer Einbruch')}")

    # Aufgeschluesselt nach Retention-Terzilen
    print("\n  AUROC-Einbruch nach OoD-Retention (Terzile):")
    order = np.argsort(retention)
    thirds = np.array_split(order, 3)
    labels = ["niedrige Retention (Objekt stark zerschnitten)",
              "mittlere Retention",
              "hohe Retention (Objekt fast intakt)"]
    for lab, idx in zip(labels, thirds):
        if len(idx):
            print(f"    {lab:<48} mean(B-C) = {drop[idx].mean():+.3f}  "
                  f"(Retention {retention[idx].min():.2f}-{retention[idx].max():.2f}, "
                  f"n={len(idx)})")

    # CSV (robuste Header: Vereinigung aller keys ueber alle Zeilen)
    p_csv = OUTPUT_DIR / "test2_fragmentation_laf.csv"
    write_csv_robust(p_csv, rows)
    print(f"\n  [Saved] {p_csv}")

    # Plot: Retention vs. AUROC-Einbruch, eingefaerbt nach Objektgroesse
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(retention, drop, c=np.log10(sizes + 1),
                     cmap="viridis", s=22, alpha=0.7, edgecolors="none")
    cb = plt.colorbar(sc)
    cb.set_label("log10(largest OoD object [px])")
    plt.axhline(0, color="gray", ls=":", lw=1)
    plt.xlabel("OoD retention under Road-ROI  (1.0 = object fully kept)")
    plt.ylabel("AUROC drop  (Trapezoid B  −  Road-ROI C)")
    plt.title("Test 2: does Road-ROI fragmenting the object cause the RbA drop?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p_png = OUTPUT_DIR / "test2_fragmentation_laf.png"
    plt.savefig(p_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p_png}")


# ===========================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = {name: load_per_image(folder, name) for name, folder in DATASETS.items()}

    # Optional: die kompletten Per-Bild-Tabellen mitspeichern (fuer eigene Auswertung)
    for name, rows in all_rows.items():
        if rows:
            key = "laf" if "Found" in name else "ro21"
            write_csv_robust(OUTPUT_DIR / f"per_image_{key}.csv", rows)

    # Test 1
    t1 = test1_size_controlled(all_rows)
    if t1:
        write_csv_robust(OUTPUT_DIR / "test1_size_controlled.csv", t1)
        print(f"  [Saved] {OUTPUT_DIR / 'test1_size_controlled.csv'}")

    # Test 2 (nur Lost & Found)
    test2_fragmentation(all_rows["Lost & Found"])

    print("\nFertig. Schick mir die Konsolenausgabe + die beiden PNGs, "
          "dann interpretiere ich die Ergebnisse.")


if __name__ == "__main__":
    main()
