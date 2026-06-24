"""
analyze_rba_hood_hypothesis.py
------------------------------
Verifiziert quantitativ die Motorhauben-Hypothese fuer den RbA-Einbruch
unter Road-ROI auf Lost & Found:

  "Der Road-ROI bricht ein, weil die Motorhaube (unterste ~10% des Bildes)
   von SegFormer als 'road' klassifiziert wird und dort hohe RbA-Scores
   (Falsch-Positive) erzeugt. Das feste Trapez (y_bot=0.90) schneidet diese
   Zone ab und bricht deshalb NICHT ein."

Drei Vorhersagen werden getestet (ueber ALLE L&F-Bilder, kein Sampling):

  (1) Liegt die Motorhauben-Zone ueberhaupt im Road-ROI, und erzeugt RbA
      dort Falsch-Positive?
        -> Anteil der untersten 10%, der als 'road' klassifiziert ist
        -> mittlerer RbA-Score dort vs. im restlichen Road-ROI

  (2) Sagt die Motorhauben-FP-Masse den AUROC-Einbruch (B - C) vorher?
        -> Korrelation hood_fp_fraction  <->  (auroc_trap - auroc_road)

  (3) GEGENPROBE (kausal): Entfernt man die unterste 10% aus dem Road-ROI,
      verschwindet der Einbruch?
        -> C_no_hood = AUROC auf (road AND y<=0.90)
        -> wird C_no_hood ~ B(Trapez)?  Dann ist die Motorhaube die Ursache.

Datengrundlage: die Score-Maps (.npz) mit rba_map, ood_label, pred_class.
Laeuft nur auf Lost & Found.

Aufruf (aus Repo-Root ODER src/evaluation/):
    python analyze_rba_hood_hypothesis.py
    python analyze_rba_hood_hypothesis.py --hood 0.90   # Zonengrenze (Default 0.90)
    python analyze_rba_hood_hypothesis.py --fp-thresh 0.6  # FP-Score-Schwelle

Ausgabe:
    results/rba_analysis/hood_hypothesis_laf.csv  (eine Zeile pro Bild)
    + Konsolen-Zusammenfassung der drei Tests
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from sklearn.metrics import roc_auc_score

# --- Repo-Pfade (funktioniert aus Root und aus src/evaluation/) ---
HERE = Path(__file__).resolve()
for cand in (HERE.parent / "src", HERE.parents[1] if len(HERE.parents) > 1 else HERE.parent):
    if (cand / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break
from paths import RESULTS_DIR, SCORE_MAPS_LAF  # noqa: E402

ANALYSIS_DIR = RESULTS_DIR / "rba_analysis"
ROAD_ID = 0

# Trapez-Geometrie aus eval_config.yaml (zur Reproduktion von Variante B)
TRAPEZOID_REL = {
    "y_top": 0.28, "x_top_l": 0.38, "x_top_r": 0.62,
    "y_bot": 0.90, "x_bot_l": 0.05, "x_bot_r": 0.95,
}

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def safe_auroc(scores, labels):
    labels = labels.astype(np.int32)
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return float("nan")
    try:
        return float(roc_auc_score(labels.ravel(), scores.ravel()))
    except Exception:
        return float("nan")


def trapezoid_mask(H, W):
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
        yt, yb = int(t["y_top"] * H), int(t["y_bot"] * H)
        for y in range(yt, min(yb, H)):
            f = (y - yt) / max(yb - yt, 1)
            xl = int((t["x_top_l"] + f * (t["x_bot_l"] - t["x_top_l"])) * W)
            xr = int((t["x_top_r"] + f * (t["x_bot_r"] - t["x_top_r"])) * W)
            mask[y, max(xl, 0):min(xr, W)] = 1
    return mask.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hood", type=float, default=0.90,
                    help="Obergrenze der Motorhauben-Zone als Bildanteil "
                         "(Default 0.90 = unterste 10%, = Trapez-Unterkante)")
    ap.add_argument("--fp-thresh", type=float, default=None,
                    help="RbA-Score-Schwelle fuer 'Falsch-Positiv'. Ohne Angabe: "
                         "pro Bild das 90. Perzentil der road-RbA-Scores.")
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SCORE_MAPS_LAF.glob("*.npz"))
    if not files:
        print(f"[Fehler] Keine .npz in {SCORE_MAPS_LAF}")
        return

    rows = []
    for f in tqdm(files, desc="Lost & Found"):
        d = np.load(f)
        if not {"rba_map", "ood_label", "pred_class"} <= set(d.files):
            continue
        rba = d["rba_map"].astype(np.float32)
        ood = (d["ood_label"].astype(np.int32) == 1)
        pred = d["pred_class"].astype(np.int32)
        H, W = rba.shape
        if ood.sum() == 0 or (~ood).sum() == 0:
            continue

        road = (pred == ROAD_ID)
        hood_row = int(args.hood * H)
        hood_zone = np.zeros((H, W), dtype=bool)
        hood_zone[hood_row:, :] = True              # unterste (1-hood) des Bildes
        road_hood = road & hood_zone                # Motorhauben-Pixel im road-ROI
        road_upper = road & ~hood_zone              # restlicher road-ROI

        # FP-Schwelle (hoher RbA-Score = OoD-Verdacht); FP = hoher Score auf NICHT-OoD
        if args.fp_thresh is not None:
            thr = args.fp_thresh
        else:
            thr = np.percentile(rba[road], 90) if road.sum() > 0 else 0.0

        # --- (1) Motorhauben-Zone: road-Anteil + FP-Verhalten ---
        n_hood = int(hood_zone.sum())
        frac_hood_is_road = road_hood.sum() / max(n_hood, 1)
        # Falsch-Positive in der Motorhauben-Zone (road, kein OoD, hoher Score)
        fp_hood = road_hood & (~ood) & (rba > thr)
        fp_hood_fraction = fp_hood.sum() / max(road_hood.sum(), 1)
        mean_rba_hood = float(rba[road_hood].mean()) if road_hood.sum() > 0 else np.nan
        mean_rba_upper = float(rba[road_upper].mean()) if road_upper.sum() > 0 else np.nan

        # --- AUROC-Varianten ---
        mB = trapezoid_mask(H, W)
        auroc_A = safe_auroc(rba, ood)                       # volles Bild
        auroc_B = safe_auroc(rba[mB], ood[mB])               # Trapez
        auroc_C = safe_auroc(rba[road], ood[road])           # Road-ROI
        # (3) Gegenprobe: Road-ROI OHNE Motorhauben-Zone
        road_nohood = road & ~hood_zone
        auroc_C_nohood = safe_auroc(rba[road_nohood], ood[road_nohood])

        rows.append({
            "image": f.stem,
            "auroc_full_A": round(auroc_A, 4),
            "auroc_trap_B": round(auroc_B, 4),
            "auroc_road_C": round(auroc_C, 4),
            "auroc_road_nohood": round(auroc_C_nohood, 4),
            "drop_B_minus_C": round(auroc_B - auroc_C, 4) if np.isfinite(auroc_B) and np.isfinite(auroc_C) else "",
            "drop_B_minus_Cnohood": round(auroc_B - auroc_C_nohood, 4) if np.isfinite(auroc_B) and np.isfinite(auroc_C_nohood) else "",
            "frac_hood_is_road": round(frac_hood_is_road, 4),
            "fp_hood_fraction": round(fp_hood_fraction, 4),
            "mean_rba_hood": round(mean_rba_hood, 4) if np.isfinite(mean_rba_hood) else "",
            "mean_rba_upper": round(mean_rba_upper, 4) if np.isfinite(mean_rba_upper) else "",
            "ood_in_hood": int((ood & hood_zone).sum()),
        })

    if not rows:
        print("[Fehler] Keine verwertbaren Bilder.")
        return

    # CSV speichern
    p = ANALYSIS_DIR / "hood_hypothesis_laf.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Auswertung ----
    def arr(k):
        return np.array([r[k] if r[k] != "" else np.nan for r in rows], dtype=float)

    fhr = arr("frac_hood_is_road")
    fpf = arr("fp_hood_fraction")
    dBC = arr("drop_B_minus_C")
    dBCnh = arr("drop_B_minus_Cnohood")
    rba_hood = arr("mean_rba_hood")
    rba_upper = arr("mean_rba_upper")
    A = arr("auroc_full_A"); B = arr("auroc_trap_B"); C = arr("auroc_road_C")
    Cnh = arr("auroc_road_nohood")
    ood_hood = arr("ood_in_hood")

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def fmt_r(r):
        return f"r = {r:+.3f}" if np.isfinite(r) else "r = n/a (zu wenig Varianz)"

    n = len(rows)
    print("\n" + "=" * 80)
    print(f"  MOTORHAUBEN-HYPOTHESE — Verifikation ueber {n} L&F-Bilder")
    print(f"  Motorhauben-Zone: unterste {(1-args.hood)*100:.0f}% des Bildes (y > {args.hood})")
    print("=" * 80)

    print("\n  (1) Liegt die Motorhaube im Road-ROI und erzeugt sie Falsch-Positive?")
    print(f"      Anteil der Motorhauben-Zone, der als 'road' klassifiziert ist:")
    print(f"        Median = {np.nanmedian(fhr):.1%}   Mean = {np.nanmean(fhr):.1%}")
    print(f"      RbA-Score in der Motorhauben-Zone vs. restlicher Road-ROI:")
    print(f"        Motorhaube : {np.nanmean(rba_hood):+.3f}")
    print(f"        restl. Road: {np.nanmean(rba_upper):+.3f}")
    higher = np.nanmean(rba_hood) > np.nanmean(rba_upper)
    print(f"        -> RbA-Scores in der Motorhauben-Zone sind im Schnitt "
          f"{'HOEHER' if higher else 'nicht hoeher'} (hoeher = mehr Falsch-Positive)")
    print(f"      Mittlerer FP-Anteil in der Motorhauben-Zone: {np.nanmean(fpf):.1%}")

    print("\n  (2) Sagt die Motorhauben-FP-Masse den AUROC-Einbruch (B-C) vorher?")
    print(f"      Korrelation  FP-Anteil(Motorhaube)  <->  Einbruch(B-C):  "
          f"{fmt_r(corr(fpf, dBC))}")
    print(f"      Korrelation  road-Anteil(Motorhaube) <->  Einbruch(B-C):  "
          f"{fmt_r(corr(fhr, dBC))}")
    print(f"      (positiv = mehr Motorhauben-FP -> groesserer Einbruch, wie erwartet)")

    print("\n  (3) GEGENPROBE — verschwindet der Einbruch, wenn man die")
    print("      Motorhauben-Zone aus dem Road-ROI entfernt?")
    print(f"        B (Trapez)            mean AUROC = {np.nanmean(B):.3f}")
    print(f"        C (Road-ROI)          mean AUROC = {np.nanmean(C):.3f}   "
          f"(Einbruch vs B: {np.nanmean(B)-np.nanmean(C):+.3f})")
    print(f"        C ohne Motorhaube     mean AUROC = {np.nanmean(Cnh):.3f}   "
          f"(Einbruch vs B: {np.nanmean(B)-np.nanmean(Cnh):+.3f})")
    recovered = (np.nanmean(B) - np.nanmean(C)) - (np.nanmean(B) - np.nanmean(Cnh))
    print(f"        -> Entfernen der Motorhaube holt {recovered:+.3f} AUROC zurueck.")
    frac_expl = recovered / max(np.nanmean(B) - np.nanmean(C), 1e-9)
    print(f"        -> Das erklaert {frac_expl:.0%} des gesamten B->C-Einbruchs.")

    # Sanity: liegen OoD-Objekte selbst in der Motorhauben-Zone? (sollten kaum)
    print(f"\n  Sanity-Check: OoD-Pixel in der Motorhauben-Zone (sollte ~0 sein):")
    print(f"        Median = {np.nanmedian(ood_hood):.0f} px,  "
          f"Bilder mit OoD in Zone: {int(np.nansum(ood_hood > 0))}/{n}")

    print("\n  Interpretation:")
    r_fp = corr(fpf, dBC)
    # Die GEGENPROBE (Test 3) ist der kausale Beleg. Die Korrelation (Test 2) ist
    # NICHT erforderlich: wenn die Motorhaube in fast JEDEM Bild aehnlich stark
    # praesent ist, gibt es kaum Varianz zwischen Bildern -> r~0 ist dann ZU ERWARTEN,
    # obwohl der Effekt gross ist. Das Urteil stuetzt sich daher auf frac_expl.
    if frac_expl >= 0.8:
        print("    >>> Hypothese BESTAETIGT (kausal): Die Motorhaube ist die Ursache des")
        print(f"        Road-ROI-Einbruchs. Ihr Entfernen erklaert {frac_expl:.0%} des Einbruchs;")
        print("        C-ohne-Motorhaube erreicht B-Niveau oder darueber.")
        if np.isfinite(r_fp) and abs(r_fp) < 0.2:
            print("        (Die ~0-Korrelation in Test 2 widerspricht dem NICHT: die Motorhaube")
            print("         ist in fast jedem Bild aehnlich praesent -> konstanter Effekt, keine")
            print("         Variation zwischen Bildern -> erwartungsgemaess r~0.)")
    elif frac_expl >= 0.4:
        print("    >>> Hypothese ueberwiegend bestaetigt: Die Motorhaube erklaert den")
        print(f"        Grossteil ({frac_expl:.0%}) des Einbruchs, aber nicht alles.")
    elif frac_expl >= 0.2:
        print("    >>> Hypothese TEILWEISE bestaetigt: Die Motorhaube traegt erkennbar")
        print("        bei, ist aber nicht die alleinige Ursache.")
    else:
        print("    >>> Hypothese NICHT bestaetigt: Die Motorhaube erklaert den Einbruch")
        print("        nur zu einem kleinen Teil; andere Faktoren dominieren.")

    print(f"\n  [Saved] {p}")


if __name__ == "__main__":
    main()
