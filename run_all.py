"""
run_all.py
----------
Master pipeline: reproduces every experiment and figure of the thesis.
All commands run from the repo root — scripts resolve their paths via src/paths.py.

    python run_all.py                  # everything, in order
    python run_all.py --list           # show all stages
    python run_all.py --only laf_eval  # run a single stage
    python run_all.py --skip baseline gallery laf_scores smiyc_scores
    python run_all.py --from smiyc_eval                         # resume from a stage
    python run_all.py --continue-on-error

⚠️  RUNTIME WARNING
    Without precomputed score-map caches the full pipeline takes SEVERAL HOURS
    up to a full day, depending on your GPU:
      - DINOv2 gallery + L&F kNN evaluation   : ~3 h   (1 096 images)
      - L&F score maps (Energy/kNN/MSP/PixOOD): ~2-3 h
      - Chapter-2 baseline                    : ~0.5-1 h
      - SMIYC score maps                      : ~10-20 min (40 images)
    Strongly recommended: fetch the caches first
        python scripts/download_score_maps.py
    and then run
        python run_all.py --skip gallery laf_scores rba_merge smiyc_scores
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY   = sys.executable

def S(*parts):  # path to a script, as string
    return str(ROOT.joinpath(*parts))

# ---------------------------------------------------------------------------
# Stage definitions: (name, description, [commands])
# ---------------------------------------------------------------------------
STAGES = [
    ("baseline",
     "Chapter 2: SegFormer-B2 baseline (MSP/Entropy/Energy) → results/baseline/",
     [[PY, S("main.py")]]),

    ("gallery",
     "DINOv2 gallery build (cache/dinov2_gallery.pt) + kNN PoC on full L&F (~3 h)",
     [[PY, S("src", "scoring", "dinov2_knn_ood.py")]]),

    ("laf_scores",
     "L&F score-map cache: Energy, kNN, MSP, PixOOD (~2-3 h, resumable) → results/roi_variants/score_maps/",
     [[PY, S("src", "scoring", "compute_score_maps.py"), "--skip_existing"]]),

    ("smiyc_scores",
     "SMIYC score-map caches (both tracks) → results/smiyc/<Track>/score_maps/",
     [[PY, S("src", "scoring", "compute_score_maps_smiyc.py"), "--track", "RoadAnomaly21", "--skip_existing"],
      [PY, S("src", "scoring", "compute_score_maps_smiyc.py"), "--track", "RoadObstacle21", "--skip_existing"]]),

    ("rba_merge",
     "Merge RbA maps (rba_score_maps/, from Colab) into the caches — skip if caches were downloaded",
     [[PY, S("src", "scoring", "merge_rba_into_score_maps.py")],
      [PY, S("src", "scoring", "merge_rba_into_score_maps_smiyc.py")]]),

    ("laf_eval",
     "ROI variants L&F (Tab. 5+6) + closing ablations (Tab. 7) + SegFormer IoU (Tab. 8)",
     [[PY, S("src", "evaluation", "evaluate_roi_variants.py")],
      [PY, S("src", "evaluation", "evaluate_roi_closing.py")],
      [PY, S("src", "evaluation", "evaluate_roi_closing_sw.py")],
      [PY, S("src", "evaluation", "measure_segformer_iou.py")]]),

    ("smiyc_eval",
     "ROI-variant evaluation on SMIYC (both tracks) → results/smiyc/<Track>/",
     [[PY, S("src", "evaluation", "evaluate_smiyc_variants.py")]]),

    ("figures",
     "All thesis figures → results/figures/ and results/smiyc/<Track>/heatmaps/",
     [[PY, S("src", "visualization", "visualize.py"), "--imgs",
       "02_Hanns_Klemm_Str_44_000002_000180", "02_Hanns_Klemm_Str_44_000006_000180"],
      [PY, S("src", "visualization", "single_image_analysis.py"), "--img", "02_Hanns_Klemm_Str_44_000006_000180"],
      [PY, S("src", "visualization", "single_image_analysis.py"), "--img", "04_Maurener_Weg_8_000004_000100"],
      [PY, S("src", "visualization", "visualize_roi_variants.py"), "--img", "02_Hanns_Klemm_Str_44_000006_000180_leftImg8bit"],
      [PY, S("src", "visualization", "visualize_roi_variants.py"), "--img", "04_Maurener_Weg_8_000004_000100_leftImg8bit"],
      [PY, S("src", "visualization", "visualize_roi_variants.py"), "--img", "15_Rechbergstr_Deckenpfronn_000004_000210_leftImg8bit"],
      [PY, S("src", "visualization", "visualize_smiyc_heatmaps.py")]]),
]

STAGE_NAMES = [s[0] for s in STAGES]


def run_stage(name, desc, commands, continue_on_error):
    print("\n" + "=" * 78)
    print(f"STAGE [{name}] — {desc}")
    print("=" * 78)
    for cmd in commands:
        printable = " ".join(str(c) for c in cmd)
        print(f"\n>>> {printable}")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"!!! Command failed (exit {result.returncode}) after {dt/60:.1f} min")
            if not continue_on_error:
                print(f"Aborting. Re-run with:  python run_all.py --from {name}")
                sys.exit(result.returncode)
        else:
            print(f"--- done in {dt/60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Run the full thesis pipeline.")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--only", nargs="+", choices=STAGE_NAMES, help="run only these stages")
    ap.add_argument("--skip", nargs="+", choices=STAGE_NAMES, default=[], help="skip stages")
    ap.add_argument("--from", dest="from_stage", choices=STAGE_NAMES,
                    help="resume from this stage")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true", help="skip runtime confirmation")
    args = ap.parse_args()

    if args.list:
        for name, desc, cmds in STAGES:
            print(f"{name:<14} {desc}  [{len(cmds)} command(s)]")
        return

    selected = STAGES
    if args.only:
        selected = [s for s in STAGES if s[0] in args.only]
    if args.from_stage:
        idx = STAGE_NAMES.index(args.from_stage)
        selected = [s for s in selected if STAGE_NAMES.index(s[0]) >= idx]
    selected = [s for s in selected if s[0] not in args.skip]

    heavy = {"baseline", "gallery", "laf_scores"} & {s[0] for s in selected}
    if heavy and not args.yes:
        print("⚠️  The selected stages include expensive GPU stages:", ", ".join(sorted(heavy)))
        print("    Expect SEVERAL HOURS of runtime depending on your GPU.")
        print("    Tip: python scripts/download_score_maps.py")
        print("         python run_all.py --skip gallery laf_scores rba_merge smiyc_scores")
        if input("Continue? [y/N] ").strip().lower() != "y":
            sys.exit(0)

    t0 = time.time()
    for name, desc, cmds in selected:
        run_stage(name, desc, cmds, args.continue_on_error)
    print(f"\nAll selected stages finished in {(time.time()-t0)/60:.1f} min.")
    print("Results: results/   |   Figures: results/figures/ and results/smiyc/<Track>/heatmaps/")


if __name__ == "__main__":
    main()
