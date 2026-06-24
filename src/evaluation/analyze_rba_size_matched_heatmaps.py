"""
analyze_rba_size_matched_heatmaps.py
------------------------------------
Zweiteilig:

  TEIL 1 — Eng groessen-gematchter Vergleich auf der Road-ROI
      Vergleicht RbA-AUROC zwischen L&F und RO21 in MEHREREN ENGEN
      Pixelfenstern (Standard: je 200 px breit), getrennt fuer
      Variante A (volles Bild) und Variante C (Road-ROI).
      So sieht man bei NAHEZU GLEICHER Objektgroesse, ob die Road-ROI
      die AUROC gegenueber dem vollen Bild senkt -- fair, ohne breite Bins.
      n pro Fenster wird klar ausgewiesen (bei RO21 oft sehr klein!).

  TEIL 2 — RbA-Heatmaps NUR auf der ausgewerteten Road-ROI
      Fuer bis zu 5 L&F- und 5 RO21-Bilder (aus den Fenstern) wird die
      RbA-Score-Heatmap gezeichnet, wobei alles AUSSERHALB der Road-ROI
      abgedunkelt ist. OoD-Konturen (gruen) sind ueberlagert.
      So wird sichtbar, ob RbA auf den Strassenpixeln selbst
      Falsch-Positive erzeugt.

Datengrundlage:
  - results/rba_analysis/per_image_laf.csv  und  per_image_ro21.csv
    (aus analyze_rba_roi_cause.py; werden hier nur gelesen -> schnell)
  - die Score-Maps (.npz) nur fuer die ausgewaehlten Heatmap-Bilder

Aufruf (aus Repo-Root ODER src/evaluation/):
    python analyze_rba_size_matched_heatmaps.py
    python analyze_rba_size_matched_heatmaps.py --windows 150-350 700-900 2000-2200
    python analyze_rba_size_matched_heatmaps.py --width 200 --n-heatmaps 5

Ausgabe:
    results/rba_analysis/size_matched_windows.csv
    results/rba_analysis/heatmaps_roadroi/laf_<stem>.png   (max 5)
    results/rba_analysis/heatmaps_roadroi/ro21_<stem>.png  (max 5)
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

# --- Repo-Pfade (funktioniert aus Root und aus src/evaluation/) ---
HERE = Path(__file__).resolve()
for cand in (HERE.parent / "src", HERE.parents[1] if len(HERE.parents) > 1 else HERE.parent):
    if (cand / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break
from paths import RESULTS_DIR, SCORE_MAPS_LAF, SMIYC_RESULTS_DIR  # noqa: E402

ANALYSIS_DIR = RESULTS_DIR / "rba_analysis"
HEATMAP_DIR = ANALYSIS_DIR / "heatmaps_roadroi"
ROAD_ID = 0   # trainId 0 = road

NPZ_DIRS = {
    "laf":  SCORE_MAPS_LAF,
    "ro21": SMIYC_RESULTS_DIR / "RoadObstacle21" / "score_maps",
}


# ===========================================================================
# Per-Bild-CSV laden (aus dem letzten Lauf von analyze_rba_roi_cause.py)
# ===========================================================================
def load_per_image_csv(key):
    p = ANALYSIS_DIR / f"per_image_{key}.csv"
    if not p.exists():
        print(f"[Fehler] {p} fehlt. Bitte zuerst analyze_rba_roi_cause.py laufen lassen,")
        print(f"         das schreibt per_image_laf.csv und per_image_ro21.csv.")
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = []
    for r in rows:
        def g(k):
            v = r.get(k, "")
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan
        out.append({
            "image": r["image"],
            "size": g("largest_object"),
            "auroc_full": g("rba_auroc_full"),
            "auroc_road": g("rba_auroc_road"),
            "retention_road": g("ood_retention_road"),
        })
    return out


# ===========================================================================
# TEIL 1: Vergleich in festen, engen Fenstern
# ===========================================================================
def windows_from_args(args, laf, ro21):
    """Liefert eine Liste (lo, hi) von Fenstern."""
    if args.windows:
        out = []
        for w in args.windows:
            lo, hi = w.split("-")
            out.append((float(lo), float(hi)))
        return out
    # Automatik: waehle Fenster rund um die RO21-Objektgroessen (weil RO21 der
    # knappe Datensatz ist). Nimm RO21-Quartile als Zentren, je +-width/2.
    ro21_sizes = np.array([r["size"] for r in ro21 if np.isfinite(r["size"])])
    centers = np.quantile(ro21_sizes, [0.25, 0.5, 0.75])
    half = args.width / 2
    return [(max(c - half, 1), c + half) for c in centers]


def fmt(x):
    return f"{x:.3f}" if np.isfinite(x) else "  -- "


def fmt_signed(x):
    return f"{x:+.3f}" if np.isfinite(x) else "  n/a "


def compare_windows(laf, ro21, windows):
    print("\n" + "=" * 84)
    print("  TEIL 1 — Eng groessen-gematchter Vergleich (A: volles Bild  vs.  C: Road-ROI)")
    print("=" * 84)
    print("  ACHTUNG: n pro Fenster beachten! Bei n=1-2 (v.a. RO21) ist die")
    print("           AUROC-Differenz anekdotisch, nicht statistisch belastbar.\n")

    laf_size = np.array([r["size"] for r in laf])
    ro21_size = np.array([r["size"] for r in ro21])

    out_rows = []
    for lo, hi in windows:
        lm = (laf_size >= lo) & (laf_size < hi)
        rm = (ro21_size >= lo) & (ro21_size < hi)

        def mean(rows, mask, key):
            vals = [rows[i][key] for i in np.where(mask)[0] if np.isfinite(rows[i][key])]
            return (np.mean(vals) if vals else np.nan), len(vals)

        l_A, l_nA = mean(laf, lm, "auroc_full")
        l_C, l_nC = mean(laf, lm, "auroc_road")
        r_A, r_nA = mean(ro21, rm, "auroc_full")
        r_C, r_nC = mean(ro21, rm, "auroc_road")

        print(f"  Fenster {int(lo)}-{int(hi)} px   "
              f"(L&F n={lm.sum()},  RO21 n={rm.sum()})")
        print(f"    L&F :  A(full)={fmt(l_A)}  C(road)={fmt(l_C)}  "
              f"->  C-A = {fmt_signed(l_C - l_A)}")
        print(f"    RO21:  A(full)={fmt(r_A)}  C(road)={fmt(r_C)}  "
              f"->  C-A = {fmt_signed(r_C - r_A)}")
        # gematchter Datensatz-Unterschied auf der Road-ROI
        print(f"    Road-ROI Unterschied (L&F - RO21) bei gleicher Groesse: "
              f"{fmt_signed(l_C - r_C)}")
        print()

        out_rows.append({
            "win_lo": int(lo), "win_hi": int(hi),
            "laf_n": int(lm.sum()), "ro21_n": int(rm.sum()),
            "laf_auroc_full": round(l_A, 4) if np.isfinite(l_A) else "",
            "laf_auroc_road": round(l_C, 4) if np.isfinite(l_C) else "",
            "laf_C_minus_A": round(l_C - l_A, 4) if np.isfinite(l_C - l_A) else "",
            "ro21_auroc_full": round(r_A, 4) if np.isfinite(r_A) else "",
            "ro21_auroc_road": round(r_C, 4) if np.isfinite(r_C) else "",
            "ro21_C_minus_A": round(r_C - r_A, 4) if np.isfinite(r_C - r_A) else "",
            "road_diff_laf_minus_ro21": round(l_C - r_C, 4) if np.isfinite(l_C - r_C) else "",
        })
    return out_rows


def suggest_windows(laf, ro21, width):
    """Findet die Fenster mit der besten GEMEINSAMEN Besetzung (min(L&F,RO21))."""
    laf_size = np.array([r["size"] for r in laf if np.isfinite(r["size"])])
    ro21_size = np.array([r["size"] for r in ro21 if np.isfinite(r["size"])])
    # Kandidaten-Zentren = jede RO21-Objektgroesse (RO21 ist der knappe Datensatz)
    best = []
    for c in sorted(ro21_size):
        lo, hi = c - width / 2, c + width / 2
        n_l = int(((laf_size >= lo) & (laf_size < hi)).sum())
        n_r = int(((ro21_size >= lo) & (ro21_size < hi)).sum())
        best.append((min(n_l, n_r), n_l, n_r, lo, hi))
    best.sort(reverse=True)
    print("\n  Tipp: am besten GEMEINSAM besetzte 200px-Fenster (min(L&F,RO21)):")
    seen = set()
    shown = 0
    for mn, n_l, n_r, lo, hi in best:
        key = (round(lo / 50) * 50)   # nahe Fenster zusammenfassen
        if key in seen:
            continue
        seen.add(key)
        print(f"    {int(lo)}-{int(hi)} px   L&F n={n_l},  RO21 n={n_r}   (min={mn})")
        shown += 1
        if shown >= 6:
            break
    print("    -> Du kannst diese gezielt waehlen, z.B.:")
    # eindeutige Top-Fenster fuer den Beispielbefehl (keine Duplikate)
    uniq_cmd = []
    seen_cmd = set()
    for _, _, _, lo, hi in best:
        tag = f"{int(lo)}-{int(hi)}"
        k = round(lo / 50) * 50
        if k in seen_cmd:
            continue
        seen_cmd.add(k)
        uniq_cmd.append(tag)
        if len(uniq_cmd) >= 3:
            break
    print(f"       python {Path(__file__).name} --windows {' '.join(uniq_cmd)}\n")


# ===========================================================================
# TEIL 2: Heatmaps -- RbA nur auf der Road-ROI
# ===========================================================================
def pick_heatmap_images(rows, windows, n_max):
    """
    Waehle bis zu n_max Bilder, die in IRGENDEIN Fenster fallen.
    Bevorzugt Bilder, deren Objekt im Road-ROI liegt (auswertbare road-AUROC),
    damit die OoD-Kontur auf dem SICHTBAREN (road) Bereich liegt.
    """
    size = np.array([r["size"] for r in rows])
    in_any = np.zeros(len(rows), dtype=bool)
    for lo, hi in windows:
        in_any |= (size >= lo) & (size < hi)
    cand = [rows[i] for i in np.where(in_any)[0]]
    if not cand:
        return []
    # Bevorzugung: Bilder mit road-Retention > 0 zuerst (Objekt im road-ROI),
    # damit man auf der sichtbaren Flaeche etwas sieht.
    cand.sort(key=lambda r: (r.get("retention_road", 0) <= 0, r["size"]))
    # gleichmaessig ueber den (sortierten) Bereich samplen
    if len(cand) <= n_max:
        return cand
    idx = np.linspace(0, len(cand) - 1, n_max).round().astype(int)
    return [cand[i] for i in idx]


def find_npz(key, stem):
    """Finde die .npz-Datei zu einem image-stem."""
    folder = NPZ_DIRS[key]
    # exakter Treffer
    p = folder / f"{stem}.npz"
    if p.exists():
        return p
    # sonst per glob (stem koennte Teilstring sein)
    hits = list(folder.glob(f"{stem}*.npz")) or list(folder.glob(f"*{stem}*.npz"))
    return hits[0] if hits else None


def draw_heatmap(key, stem, out_path):
    npz = find_npz(key, stem)
    if npz is None:
        print(f"    [Warn] keine .npz fuer {stem} gefunden -- uebersprungen")
        return False
    d = np.load(npz)
    rba = d["rba_map"].astype(np.float32)
    ood = (d["ood_label"].astype(np.int32) == 1)
    pred = d["pred_class"].astype(np.int32)
    road = (pred == ROAD_ID)

    H, W = rba.shape
    n_ood = int(ood.sum())
    ood_in_road = int((ood & road).sum())
    retention = ood_in_road / max(n_ood, 1)

    # AUROC auf voller Flaeche und auf Road-ROI (zur Anzeige im Titel)
    from sklearn.metrics import roc_auc_score
    def auroc(mask):
        l = ood[mask].astype(int); s = rba[mask]
        if l.sum() == 0 or (l == 0).sum() == 0:
            return float("nan")
        return roc_auc_score(l, s)
    a_full = auroc(np.ones_like(ood, dtype=bool))
    a_road = auroc(road)

    # --- Visualisierung: RbA-Heatmap, ausserhalb Road-ROI abgedunkelt ---
    # rba normalisieren NUR ueber die Road-ROI (das ist der ausgewertete Bereich)
    if road.sum() > 0:
        lo, hi = np.percentile(rba[road], [2, 98])
    else:
        lo, hi = rba.min(), rba.max()
    norm = np.clip((rba - lo) / max(hi - lo, 1e-6), 0, 1)

    cmap = plt.cm.inferno
    rgb = cmap(norm)[..., :3]
    # ausserhalb der Road-ROI: stark abdunkeln
    rgb[~road] = rgb[~road] * 0.18

    # OoD-Konturen (gruen) ueberlagern
    if n_ood > 0:
        contour = ndimage.binary_dilation(ood, iterations=2) & ~ndimage.binary_erosion(ood, iterations=1)
        rgb[contour] = [0.0, 1.0, 0.0]

    fig, ax = plt.subplots(figsize=(W / 200, H / 200), dpi=150)
    ax.imshow(rgb)
    ax.axis("off")
    title = (f"{key.upper()}  {stem}\n"
             f"OoD={n_ood}px  largest≈{_largest(ood)}px  "
             f"Road-ROI retention={retention:.0%}\n"
             f"AUROC  full={_f(a_full)}  road-ROI={_f(a_road)}")
    ax.set_title(title, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("RbA score (normalised over road-ROI)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _largest(ood):
    lab, n = ndimage.label(ood)
    if n == 0:
        return 0
    return int(ndimage.sum(ood, lab, range(1, n + 1)).max())


def _f(x):
    return f"{x:.3f}" if np.isfinite(x) else "n/a"


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="*", default=None,
                    help="Feste Fenster wie 150-350 700-900 2000-2200 (px). "
                         "Ohne Angabe: automatisch 3 Fenster um RO21-Quartile.")
    ap.add_argument("--width", type=float, default=200,
                    help="Fensterbreite fuer die Automatik (Default 200 px)")
    ap.add_argument("--n-heatmaps", type=int, default=5,
                    help="max. Heatmaps pro Datensatz (Default 5)")
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    laf = load_per_image_csv("laf")
    ro21 = load_per_image_csv("ro21")
    if laf is None or ro21 is None:
        return

    windows = windows_from_args(args, laf, ro21)

    # TEIL 1
    rows = compare_windows(laf, ro21, windows)
    # Hilfe: am besten gemeinsam besetzte Fenster vorschlagen
    suggest_windows(laf, ro21, args.width)
    p = ANALYSIS_DIR / "size_matched_windows.csv"
    if rows:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  [Saved] {p}")

    # TEIL 2: Heatmaps
    print("\n" + "=" * 84)
    print("  TEIL 2 — RbA-Heatmaps (nur Road-ROI sichtbar, OoD-Kontur gruen)")
    print("=" * 84)
    for key, rows_ds in [("laf", laf), ("ro21", ro21)]:
        picks = pick_heatmap_images(rows_ds, windows, args.n_heatmaps)
        print(f"  {key.upper()}: {len(picks)} Bilder ausgewaehlt")
        for r in picks:
            out = HEATMAP_DIR / f"{key}_{r['image']}.png"
            ok = draw_heatmap(key, r["image"], out)
            if ok:
                print(f"    [Saved] {out.name}   (size≈{int(r['size'])}px)")

    print("\nFertig. Schick mir die Konsolenausgabe + die Heatmaps, dann interpretiere ich.")


if __name__ == "__main__":
    main()
