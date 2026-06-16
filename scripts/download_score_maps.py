"""
download_score_maps.py
----------------------
Downloads the precomputed score-map caches (Energy, DINOv2-kNN, MSP, PixOOD, RbA)
from Zenodo and unpacks them into the locations the evaluation scripts expect:

    results/roi_variants/score_maps/*.npz      (Lost & Found, 1 096 images)
    results/smiyc/RoadAnomaly21/score_maps/*.npz
    results/smiyc/RoadObstacle21/score_maps/*.npz

Using these caches skips ~5-6 h of GPU inference and is the ONLY way to evaluate
RbA without recomputing its maps on Google Colab (see colab/README.md).

Usage:
    python scripts/download_score_maps.py
    python scripts/download_score_maps.py --dataset laf      # only Lost & Found
    python scripts/download_score_maps.py --dataset smiyc    # only SMIYC
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# TODO(Gerd): nach dem Zenodo-Upload die echten Download-URLs eintragen.
# Zenodo-Dateien haben URLs der Form:
#   https://zenodo.org/records/<RECORD_ID>/files/<FILENAME>?download=1
# ---------------------------------------------------------------------------
ARCHIVES = {
    "laf": {
        "url": "https://zenodo.org/records/20722623/files/score_maps_laf.zip?download=1",
        "target": ROOT / "results" / "roi_variants" / "score_maps",
    },
    "smiyc_ra21": {
        "url": "https://zenodo.org/records/20722623/files/score_maps_smiyc_RoadAnomaly21.zip?download=1",
        "target": ROOT / "results" / "smiyc" / "RoadAnomaly21" / "score_maps",
    },
    "smiyc_ro21": {
        "url": "https://zenodo.org/records/20722623/files/score_maps_smiyc_RoadObstacle21.zip?download=1",
        "target": ROOT / "results" / "smiyc" / "RoadObstacle21" / "score_maps",
    },
}


def download_and_extract(name: str, url: str, target: Path):
    if "XXXXXXX" in url:
        print(f"[{name}] URL noch nicht konfiguriert — bitte Zenodo-Record-ID in "
              f"scripts/download_score_maps.py eintragen.")
        return False

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target.parent / f"{name}_download.zip"

    print(f"[{name}] Downloading {url}")

    def _progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r[{name}] {pct:3d}%  ({blocks*block_size/1e9:.2f} GB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    print(f"\n[{name}] Extracting to {target}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    zip_path.unlink()
    n = len(list(target.glob("*.npz")))
    print(f"[{name}] Done — {n} .npz files in {target}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["all", "laf", "smiyc"], default="all")
    args = ap.parse_args()

    selected = list(ARCHIVES.keys())
    if args.dataset == "laf":
        selected = ["laf"]
    elif args.dataset == "smiyc":
        selected = ["smiyc_ra21", "smiyc_ro21"]

    ok = all(download_and_extract(k, ARCHIVES[k]["url"], ARCHIVES[k]["target"])
             for k in selected)
    if ok:
        print("\nAlle Caches bereit. Weiter mit:")
        print("  python run_all.py --skip gallery laf_scores smiyc_scores")


if __name__ == "__main__":
    main()
