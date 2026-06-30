"""
find_variant_c_failure_case.py
------------------------------
Sucht in den vorhandenen Lost-&-Found-Score-Maps ein gutes Ersatzbild fuer den
VERSAGENSFALL VON VARIANTE C (Road-ROI), nachdem der alte Kandidat (Kind auf
Bobby-Car) durch die test-NoKnown-Filterung entfallen ist.

Gesucht wird ein Bild, bei dem:
  - die OoD-Retention unter Road-ROI NIEDRIG ist
    (das OoD-Objekt wird von SegFormer NICHT als "road" klassifiziert und
     faellt damit aus der Variante-C-ROI heraus),
  - die Retention unter Trapez (B) und Negativfilter (D) HOCH bleibt
    (so entsteht der lehrreiche Kontrast "C versagt, B und D nicht"),
  - das Objekt GENUG OoD-Pixel hat, damit es auf der Abbildung sichtbar ist,
  - das Objekt UEBERWIEGEND auf der Fahrbahn liegt (sonst ist der Ausfall aus
    der Road-ROI trivial und didaktisch wertlos).

Es wird NICHTS neu inferiert; alles laeuft auf dem .npz-Cache.

Aufruf (aus Repo-Root ODER src/evaluation/):
    python find_variant_c_failure_case.py
    python find_variant_c_failure_case.py --min-ood 800 --top 15

Ausgabe:
    Konsolen-Ranking der besten Kandidaten (mit allen Kennzahlen)
    results/rba_analysis/variant_c_failure_candidates.csv
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# --- Repo-Pfade (funktioniert aus Root und aus src/evaluation/) ---
HERE = Path(__file__).resolve()
for cand in (HERE.parent / "src", HERE.parents[1] if len(HERE.parents) > 1 else HERE.parent):
    if (cand / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break
from paths import RESULTS_DIR, SCORE_MAPS_LAF  # noqa: E402

ANALYSIS_DIR = RESULTS_DIR / "rba_analysis"
ROAD_ID = 0
SIDEWALK_ID = 1

# Trapez (Variante B) exakt wie eval_config.yaml / make_roi_figure.py
TRAPEZOID_REL = {
    "y_top": 0.28, "x_top_l": 0.38, "x_top_r": 0.62,
    "y_bot": 0.90, "x_bot_l": 0.05, "x_bot_r": 0.95,
}
# Negativfilter (Variante D)
BACKGROUND_IDS = [2, 3, 4, 8, 10]   # building, wall, fence, vegetation, sky
MSP_THRESHOLD = 0.95


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


def negative_filter_mask(pred, msp):
    bg = np.zeros(pred.shape, dtype=bool)
    for c in BACKGROUND_IDS:
        bg |= ((pred == c) & (msp > MSP_THRESHOLD))
    return ~bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ood", type=int, default=500,
                    help="Mindestzahl OoD-Pixel, damit das Objekt sichtbar ist "
                         "(Default 500)")
    ap.add_argument("--max-road-ret", type=float, default=0.20,
                    help="Maximale Road-ROI-Retention fuer einen Versagensfall "
                         "(Default 0.20 = hoechstens 20%% des Objekts bleiben)")
    ap.add_argument("--min-other-ret", type=float, default=0.90,
                    help="Mindest-Retention unter Trapez UND Negativfilter "
                         "(Default 0.90)")
    ap.add_argument("--top", type=int, default=15,
                    help="Anzahl der angezeigten Top-Kandidaten (Default 15)")
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SCORE_MAPS_LAF.glob("*.npz"))
    if not files:
        print(f"[Fehler] Keine .npz in {SCORE_MAPS_LAF}")
        return

    rows = []
    for f in tqdm(files, desc="Lost & Found"):
        d = np.load(f)
        if not {"ood_label", "pred_class"} <= set(d.files):
            continue
        ood = (d["ood_label"].astype(np.int32) == 1)
        pred = d["pred_class"].astype(np.int32)
        msp = d["msp_map"].astype(np.float32) if "msp_map" in d.files else None
        H, W = ood.shape
        n_ood = int(ood.sum())
        if n_ood < args.min_ood:
            continue

        road = (pred == ROAD_ID)
        roadsw = np.isin(pred, [ROAD_ID, SIDEWALK_ID])
        mB = trapezoid_mask(H, W)

        ret_road = int((ood & road).sum()) / n_ood
        ret_roadsw = int((ood & roadsw).sum()) / n_ood
        ret_trap = int((ood & mB).sum()) / n_ood
        ret_negf = (int((ood & negative_filter_mask(pred, msp)).sum()) / n_ood
                    if msp is not None else float("nan"))

        # Wie viel des Objekts liegt im "Fahrbahn-Bildbereich" (untere Bildhaelfte
        # ohne die unterste Motorhauben-Zone)? Dient als Plausibilitaet, dass das
        # Objekt wirklich auf der Strasse ist und nicht am oberen Bildrand.
        ood_ys = np.where(ood.any(axis=1))[0]
        obj_center_y = float(np.mean(np.where(ood)[0])) / H if n_ood else 0.0

        rows.append({
            "image": f.stem,
            "ood_pixels": n_ood,
            "ret_road": round(ret_road, 3),
            "ret_roadsw": round(ret_roadsw, 3),
            "ret_trapezoid": round(ret_trap, 3),
            "ret_negfilter": round(ret_negf, 3) if not np.isnan(ret_negf) else "",
            "obj_center_y_rel": round(obj_center_y, 3),
        })

    if not rows:
        print("[Fehler] Keine Bilder mit genug OoD-Pixeln gefunden.")
        return

    # --- Kandidaten filtern: C versagt, B und D behalten ---
    def passes(r):
        if r["ret_road"] > args.max_road_ret:
            return False
        if r["ret_trapezoid"] < args.min_other_ret:
            return False
        if r["ret_negfilter"] != "" and r["ret_negfilter"] < args.min_other_ret:
            return False
        return True

    cands = [r for r in rows if passes(r)]

    # Score: je niedriger Road-Retention und je hoeher Trapez/Negativ, desto besser;
    # leichte Bevorzugung von mehr OoD-Pixeln (sichtbarer) und Objekten weiter unten.
    def score(r):
        negf = r["ret_negfilter"] if r["ret_negfilter"] != "" else 1.0
        return (
            (1.0 - r["ret_road"]) * 2.0          # Hauptkriterium: niedrige Road-Ret
            + r["ret_trapezoid"]                 # Trapez behaelt
            + negf                               # Negativfilter behaelt
            + min(r["ood_pixels"] / 5000, 1.0)   # sichtbar (gedeckelt)
            + r["obj_center_y_rel"]              # Objekt eher unten im Bild
        )
    cands.sort(key=score, reverse=True)

    print("\n" + "=" * 92)
    print(f"  VERSAGENSFALL-KANDIDATEN fuer Variante C  "
          f"(Road-Ret \u2264 {args.max_road_ret:.0%}, Trapez & Neg.-Filter \u2265 "
          f"{args.min_other_ret:.0%}, \u2265 {args.min_ood} OoD-Px)")
    print("=" * 92)
    print(f"  Gefundene Kandidaten: {len(cands)} von {len(rows)} geprueften Bildern\n")
    if not cands:
        print("  Keine Kandidaten unter diesen Kriterien. Versuche z.B.:")
        print("    python find_variant_c_failure_case.py --max-road-ret 0.35 --min-other-ret 0.85")
        # trotzdem die Bilder mit der niedrigsten Road-Retention zeigen
        rows.sort(key=lambda r: r["ret_road"])
        print("\n  Stattdessen: die 10 Bilder mit der niedrigsten Road-Retention:")
        cands = rows[:10]

    print(f"  {'#':<3}{'Bild':<48}{'OoD-Px':>7}{'Road':>7}{'R+SW':>7}"
          f"{'Trap':>7}{'NegF':>7}{'y_obj':>7}")
    print("  " + "-" * 88)
    for i, r in enumerate(cands[:args.top], 1):
        negf = f"{r['ret_negfilter']:.2f}" if r["ret_negfilter"] != "" else "  -- "
        print(f"  {i:<3}{r['image']:<48}{r['ood_pixels']:>7}"
              f"{r['ret_road']:>7.2f}{r['ret_roadsw']:>7.2f}"
              f"{r['ret_trapezoid']:>7.2f}{negf:>7}{r['obj_center_y_rel']:>7.2f}")

    # CSV (alle Kandidaten, sortiert)
    p = ANALYSIS_DIR / "variant_c_failure_candidates.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cands[0].keys()))
        w.writeheader()
        w.writerows(cands)
    print(f"\n  [Saved] {p}")
    print("\n  Tipp: Nimm einen Kandidaten aus den oberen Plaetzen und visualisiere ihn mit")
    print("        deinem vorhandenen Skript, z.B.:")
    if cands:
        print(f'        python src/visualization/visualize_roi_variants.py '
              f'--img "{cands[0]["image"]}"')


if __name__ == "__main__":
    main()
