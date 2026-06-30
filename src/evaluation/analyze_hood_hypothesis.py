"""
analyze_hood_hypothesis.py
--------------------------
Verallgemeinerte Version von analyze_rba_hood_hypothesis.py: prueft die
Motorhauben-Hypothese fuer BELIEBIGE Score-Maps -- standardmaessig fuer RbA
UND PixOOD, da beide unter den modellabhaengigen Varianten C/D auf
Lost & Found einbrechen.

Hypothese (fuer jede getestete Methode):
  "Der Einbruch unter Road-ROI entsteht, weil die Motorhaube (unterste ~10%
   des Bildes) von SegFormer als 'road' klassifiziert wird und der Score dort
   Falsch-Positive erzeugt. Das feste Trapez (y_bot=0.90) schneidet diese Zone
   ab und bricht deshalb NICHT ein."

Pro Methode werden drei Vorhersagen getestet (ueber ALLE L&F-Bilder):
  (1) Liegt die Motorhauben-Zone im Road-ROI, und ist der Score dort
      auffaellig hoch (Falsch-Positive)?
  (2) Korreliert die Motorhauben-FP-Masse mit dem AUROC-Einbruch (B - C)?
      (Bei einem konstant praesenten Confounder ist r~0 zu ERWARTEN.)
  (3) GEGENPROBE (kausal): Entfernt man die unterste 10% aus dem Road-ROI,
      verschwindet der Einbruch? -> C_no_hood ~ B(Trapez)?

WICHTIG zur Score-Konvention:
  Fuer beide Methoden gilt in den .npz: HOEHERER Wert = staerkerer OoD-Hinweis
  (rba_map in [-1.1, 0], pixood_map in [0, 1]). "Hoher Score auf Nicht-OoD-Pixel"
  = Falsch-Positiv. Das Skript bestimmt die FP-Schwelle pro Bild als Perzentil
  der Road-Scores und ist daher skaleninvariant.

Aufruf (aus Repo-Root ODER src/evaluation/):
    python analyze_hood_hypothesis.py                      # RbA und PixOOD + Beispiel-Heatmap
    python analyze_hood_hypothesis.py --methods rba        # nur RbA
    python analyze_hood_hypothesis.py --methods pixood     # nur PixOOD
    python analyze_hood_hypothesis.py --hood 0.90 --fp-pct 90
    python analyze_hood_hypothesis.py --no-heatmap         # Tabellen-Analyse ohne Abbildung
    python analyze_hood_hypothesis.py --heatmap-img <stem> # andere Beispiel-Szene

Ausgabe je Methode:
    results/rba_analysis/hood_hypothesis_<method>_laf.csv
    + Konsolen-Zusammenfassung
Zusaetzlich (sofern nicht --no-heatmap):
    results/figures/chapter5/hood_<method>_heatmap.png
    (Score-Heatmap ueber der Road-ROI fuer eine Beispiel-Szene; zeigt den
     Falsch-Positiv-Block auf der Ego-Motorhaube -- die Abbildung der Arbeit.)
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve()
for cand in (HERE.parent / "src", HERE.parents[1] if len(HERE.parents) > 1 else HERE.parent):
    if (cand / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break
from paths import RESULTS_DIR, SCORE_MAPS_LAF  # noqa: E402
try:
    from paths import FIGURES_DIR  # noqa: E402
except ImportError:
    FIGURES_DIR = RESULTS_DIR / "figures"

ANALYSIS_DIR = RESULTS_DIR / "rba_analysis"
ROAD_ID = 0

TRAPEZOID_REL = {
    "y_top": 0.28, "x_top_l": 0.38, "x_top_r": 0.62,
    "y_bot": 0.90, "x_bot_l": 0.05, "x_bot_r": 0.95,
}

# Welche .npz-Keys gehoeren zu welcher Methode (hoeher = mehr OoD).
SCORE_KEYS = {
    "rba": "rba_map",
    "pixood": "pixood_map",
    "energy": "energy_map",
    "knn": "knn_map",
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


def analyze_method(method, files, hood_frac, fp_pct):
    score_key = SCORE_KEYS[method]
    rows = []
    skipped = 0
    for f in tqdm(files, desc=f"{method.upper():8s}"):
        d = np.load(f)
        if score_key not in d.files or "ood_label" not in d.files or "pred_class" not in d.files:
            skipped += 1
            continue
        score = d[score_key].astype(np.float32)
        ood = (d["ood_label"].astype(np.int32) == 1)
        pred = d["pred_class"].astype(np.int32)
        H, W = score.shape
        if ood.sum() == 0 or (~ood).sum() == 0:
            continue

        road = (pred == ROAD_ID)
        hood_row = int(hood_frac * H)
        hood_zone = np.zeros((H, W), dtype=bool)
        hood_zone[hood_row:, :] = True
        road_hood = road & hood_zone
        road_upper = road & ~hood_zone

        thr = np.percentile(score[road], fp_pct) if road.sum() > 0 else 0.0
        fp_hood = road_hood & (~ood) & (score > thr)
        fp_hood_fraction = fp_hood.sum() / max(road_hood.sum(), 1)
        mean_hood = float(score[road_hood].mean()) if road_hood.sum() > 0 else np.nan
        mean_upper = float(score[road_upper].mean()) if road_upper.sum() > 0 else np.nan

        mB = trapezoid_mask(H, W)
        a_A = safe_auroc(score, ood)
        a_B = safe_auroc(score[mB], ood[mB])
        a_C = safe_auroc(score[road], ood[road])
        road_nohood = road & ~hood_zone
        a_C_nohood = safe_auroc(score[road_nohood], ood[road_nohood])

        rows.append({
            "image": f.stem,
            "auroc_full_A": round(a_A, 4),
            "auroc_trap_B": round(a_B, 4),
            "auroc_road_C": round(a_C, 4),
            "auroc_road_nohood": round(a_C_nohood, 4),
            "drop_B_minus_C": round(a_B - a_C, 4) if np.isfinite(a_B) and np.isfinite(a_C) else "",
            "frac_hood_is_road": round(road_hood.sum() / max(hood_zone.sum(), 1), 4),
            "fp_hood_fraction": round(fp_hood_fraction, 4),
            "mean_score_hood": round(mean_hood, 4) if np.isfinite(mean_hood) else "",
            "mean_score_upper": round(mean_upper, 4) if np.isfinite(mean_upper) else "",
            "ood_in_hood": int((ood & hood_zone).sum()),
        })
    return rows, skipped


def report(method, rows, hood_frac):
    if not rows:
        print(f"\n[{method.upper()}] Keine verwertbaren Bilder (Score-Key fehlt im Cache?).")
        return

    def arr(k):
        return np.array([r[k] if r[k] != "" else np.nan for r in rows], dtype=float)

    fhr = arr("frac_hood_is_road"); fpf = arr("fp_hood_fraction")
    dBC = arr("drop_B_minus_C")
    s_hood = arr("mean_score_hood"); s_upper = arr("mean_score_upper")
    A = arr("auroc_full_A"); B = arr("auroc_trap_B"); C = arr("auroc_road_C")
    Cnh = arr("auroc_road_nohood"); ood_hood = arr("ood_in_hood")

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def fmt_r(r):
        return f"r = {r:+.3f}" if np.isfinite(r) else "r = n/a (zu wenig Varianz)"

    n = len(rows)
    print("\n" + "=" * 80)
    print(f"  MOTORHAUBEN-HYPOTHESE [{method.upper()}] -- {n} L&F-Bilder")
    print(f"  Motorhauben-Zone: unterste {(1-hood_frac)*100:.0f}% (y > {hood_frac})")
    print("=" * 80)

    print("\n  (1) Motorhaube im Road-ROI und Score dort auffaellig hoch?")
    print(f"      Anteil der Motorhauben-Zone als 'road': "
          f"Median {np.nanmedian(fhr):.1%}, Mean {np.nanmean(fhr):.1%}")
    print(f"      Mittlerer Score  Motorhaube={np.nanmean(s_hood):+.3f}  "
          f"vs. restl. Road={np.nanmean(s_upper):+.3f}  "
          f"({'HOEHER' if np.nanmean(s_hood) > np.nanmean(s_upper) else 'nicht hoeher'} "
          f"= mehr Falsch-Positive)")
    print(f"      Mittlerer FP-Anteil in der Motorhauben-Zone: {np.nanmean(fpf):.1%}")

    print("\n  (2) Korrelation mit dem AUROC-Einbruch (B-C):")
    print(f"      FP-Anteil(Motorhaube)  <-> Einbruch:  {fmt_r(corr(fpf, dBC))}")
    print(f"      (Bei konstant praesenter Motorhaube ist r~0 ZU ERWARTEN.)")

    print("\n  (3) GEGENPROBE -- Motorhauben-Zone aus Road-ROI entfernen:")
    print(f"      B (Trapez)         mean AUROC = {np.nanmean(B):.3f}")
    print(f"      C (Road-ROI)       mean AUROC = {np.nanmean(C):.3f}   "
          f"(Einbruch vs B: {np.nanmean(B)-np.nanmean(C):+.3f})")
    print(f"      C ohne Motorhaube  mean AUROC = {np.nanmean(Cnh):.3f}   "
          f"(Einbruch vs B: {np.nanmean(B)-np.nanmean(Cnh):+.3f})")
    recovered = (np.nanmean(B) - np.nanmean(C)) - (np.nanmean(B) - np.nanmean(Cnh))
    frac_expl = recovered / max(np.nanmean(B) - np.nanmean(C), 1e-9)
    print(f"      -> Entfernen holt {recovered:+.3f} AUROC zurueck "
          f"({frac_expl:.0%} des B->C-Einbruchs).")

    print(f"\n  Sanity: OoD-Pixel in der Motorhauben-Zone -- "
          f"Median {np.nanmedian(ood_hood):.0f} px, "
          f"Bilder mit OoD dort: {int(np.nansum(ood_hood > 0))}/{n}")

    # Urteil
    print("\n  Fazit:", end=" ")
    if np.nanmean(B) - np.nanmean(C) < 0.01:
        print(">>> Auf dieser Methode gibt es kaum einen B->C-Einbruch; "
              "der Test ist hier wenig aussagekraeftig.")
    elif frac_expl >= 0.6:
        print(f">>> Motorhaube BESTAETIGT als Hauptursache "
              f"({frac_expl:.0%} des Einbruchs erklaert).")
    elif frac_expl >= 0.3:
        print(f">>> Motorhaube traegt ueberwiegend bei ({frac_expl:.0%}).")
    else:
        print(f">>> Motorhaube erklaert nur {frac_expl:.0%} -- andere Faktoren dominieren.")

    p = ANALYSIS_DIR / f"hood_hypothesis_{method}_laf.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [Saved] {p}")


# Standard-Beispielszene fuer die Heatmap-Abbildung (Abbildung in Kapitel 5).
HEATMAP_DEFAULT_STEM = "02_Hanns_Klemm_Str_44_000010_000080"


def save_hood_heatmap(method, stem, hood_frac):
    """Rendert die Score-Heatmap einer Methode ueber der Road-ROI fuer EINE
    Beispielszene und hebt den Falsch-Positiv-Block auf der Motorhaube hervor.
    Erzeugt results/figures/chapter5/hood_<method>_heatmap.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score_key = SCORE_KEYS[method]
    # passende .npz-Datei finden (Teilstring-Match, robust gegen _leftImg8bit)
    target = stem.replace("_leftImg8bit", "")
    match = None
    for f in sorted(SCORE_MAPS_LAF.glob("*.npz")):
        if target in f.stem:
            match = f
            break
    if match is None:
        print(f"  [Heatmap] Szene '{stem}' nicht im Cache gefunden -- uebersprungen.")
        return
    d = np.load(match)
    if score_key not in d.files:
        print(f"  [Heatmap] '{score_key}' fehlt in {match.name} -- uebersprungen.")
        return

    score = d[score_key].astype(np.float32)
    ood = (d["ood_label"].astype(np.int32) == 1)
    pred = d["pred_class"].astype(np.int32)
    H, W = score.shape
    road = (pred == ROAD_ID)

    # Score nur auf Road-ROI, Rest abgedunkelt; auf [0,1] normiert fuer Anzeige
    s = score.copy()
    lo, hi = np.percentile(s[road], 1), np.percentile(s[road], 99) if road.sum() else (s.min(), s.max())
    s_norm = np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)
    disp = np.zeros((H, W))
    disp[road] = s_norm[road]

    a_C = safe_auroc(score[road], ood[road]) if road.sum() else float("nan")
    road_nohood = road.copy()
    road_nohood[int(hood_frac * H):, :] = False
    a_nohood = safe_auroc(score[road_nohood], ood[road_nohood]) if road_nohood.sum() else float("nan")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bg = np.zeros((H, W, 3))
    ax.imshow(bg)
    im = ax.imshow(np.ma.masked_where(~road, disp), cmap="magma", vmin=0, vmax=1)
    # GT-Kontur gruen
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
        if ood.sum() > 0:
            edge = binary_dilation(ood, iterations=2) & ~binary_erosion(ood, iterations=1)
            ov = np.zeros((H, W, 4)); ov[edge] = [0, 1, 0, 1]
            ax.imshow(ov)
    except Exception:
        if ood.sum() > 0:
            ax.contour(ood, levels=[0.5], colors="lime", linewidths=1.2)
    # Motorhauben-Grenze markieren
    ax.axhline(int(hood_frac * H), color="white", lw=1.0, ls="--", alpha=0.8)
    ax.text(W * 0.01, int(hood_frac * H) - 8, "Motorhauben-Zone",
            color="white", fontsize=9, alpha=0.9)
    ax.set_title(
        f"{method.upper()}-Score auf Road-ROI  |  "
        f"AUROC(C)={a_C:.3f}  \u2192  ohne Motorhaube={a_nohood:.3f}",
        fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{method.upper()}-Score (normiert)")
    plt.tight_layout()

    out_dir = FIGURES_DIR / "chapter5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"hood_{method}_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Heatmap] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["rba", "pixood"],
                    choices=list(SCORE_KEYS.keys()),
                    help="Welche Methoden testen (Default: rba pixood)")
    ap.add_argument("--hood", type=float, default=0.90,
                    help="Obergrenze der Motorhauben-Zone (Default 0.90)")
    ap.add_argument("--fp-pct", type=float, default=90,
                    help="Perzentil der Road-Scores als FP-Schwelle (Default 90)")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="Nur Tabellen-Analyse, keine Beispiel-Heatmap erzeugen")
    ap.add_argument("--heatmap-img", type=str, default=HEATMAP_DEFAULT_STEM,
                    help=f"Beispielszene fuer die Heatmap (Default: {HEATMAP_DEFAULT_STEM})")
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SCORE_MAPS_LAF.glob("*.npz"))
    if not files:
        print(f"[Fehler] Keine .npz in {SCORE_MAPS_LAF}")
        return

    for method in args.methods:
        rows, skipped = analyze_method(method, files, args.hood, args.fp_pct)
        if skipped:
            print(f"  [{method.upper()}] {skipped} Bilder ohne '{SCORE_KEYS[method]}' uebersprungen.")
        report(method, rows, args.hood)
        if not args.no_heatmap:
            save_hood_heatmap(method, args.heatmap_img, args.hood)

    print("\nFertig.")


if __name__ == "__main__":
    main()
