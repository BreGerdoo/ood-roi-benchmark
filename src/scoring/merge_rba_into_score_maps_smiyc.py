"""
merge_rba_into_score_maps_smiyc.py
----------------------------------
Mischt die auf Colab berechneten RbA-Score-Maps in die lokalen SMIYC-
Score-Maps (Energy/DINOv2/PixOOD/MSP) ein — analog zu
merge_rba_into_score_maps.py, aber für SMIYC.

Unterschiede zum L&F-Merge:
  - Pro Track getrennt (RoadAnomaly21, RoadObstacle21)
  - SMIYC-Dateinamen matchen DIREKT: <stem>_rba.npz -> <stem>.npz
    (kein _leftImg8bit-Suffix wie bei L&F)

Erwartete Eingabe (von Colab heruntergeladen + entpackt):
    <rba_dir>/RoadAnomaly21/validation0000_rba.npz, ...
    <rba_dir>/RoadObstacle21/validation_1_rba.npz, ...

Ziel (lokale SMIYC-Score-Maps):
    results/smiyc/RoadAnomaly21/score_maps/validation0000.npz, ...
    results/smiyc/RoadObstacle21/score_maps/validation_1.npz, ...

Aufruf:
    python merge_rba_into_score_maps_smiyc.py --rba_dir rba_score_maps/smiyc
    python merge_rba_into_score_maps_smiyc.py --rba_dir "..." --dry_run
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RBA_UPLOAD_DIR, SMIYC_RESULTS_DIR

TRACKS = ["RoadAnomaly21", "RoadObstacle21"]


def merge_track(track, rba_dir, dry_run):
    rba_track_dir = Path(rba_dir) / track
    target_dir    = SMIYC_RESULTS_DIR / track / "score_maps"

    if not rba_track_dir.exists():
        print(f"[{track}] [Warn] RbA-Ordner nicht gefunden: {rba_track_dir} — übersprungen.")
        return
    if not target_dir.exists():
        print(f"[{track}] [Warn] Ziel-Ordner nicht gefunden: {target_dir} — übersprungen.")
        return

    rba_files = sorted(rba_track_dir.glob("*_rba.npz"))
    target_map = {p.stem: p for p in target_dir.glob("*.npz")}

    print(f"\n[{track}]")
    print(f"  RbA-Files:        {len(rba_files)}")
    print(f"  Bestehende Files: {len(target_map)}")

    # Matching: <stem>_rba -> <stem>
    matched   = []
    unmatched = []
    for rba_path in rba_files:
        stem = rba_path.stem.replace("_rba", "")
        if stem in target_map:
            matched.append((rba_path, target_map[stem]))
        else:
            unmatched.append(rba_path)

    print(f"  Matching:   {len(matched)}/{len(rba_files)}")
    print(f"  Unmatched:  {len(unmatched)}")
    if unmatched:
        for u in unmatched[:5]:
            print(f"     - {u.name}")

    if dry_run:
        print(f"  [Dry-Run] Würde {len(matched)} Dateien mergen.")
        return

    n_merged = 0
    n_errors = 0
    for rba_path, target_path in tqdm(matched, desc=f"Merge {track}"):
        try:
            # RbA-Map laden, Handle sofort schließen
            with np.load(rba_path) as rba_data:
                rba_map = np.array(rba_data["rba_map"])

            # Ziel laden, alle Felder in den Speicher kopieren
            with np.load(target_path) as target_data:
                existing_fields = list(target_data.files)
                new_dict = {k: np.array(target_data[k]) for k in existing_fields}

            # Shape-Check
            if "ood_label" in existing_fields:
                target_shape = new_dict["ood_label"].shape
                if target_shape != rba_map.shape:
                    print(f"\n[Warn] Shape-Mismatch bei {target_path.name}: "
                          f"target={target_shape} vs rba={rba_map.shape}")
                    n_errors += 1
                    continue

            new_dict["rba_map"] = rba_map

            # Atomar speichern (numpy hängt .npz an -> ohne Suffix speichern, dann rename)
            tmp_base = target_path.with_name(target_path.stem + "_tmp")
            np.savez_compressed(tmp_base, **new_dict)
            tmp_path = tmp_base.with_suffix(".npz")
            if target_path.exists():
                target_path.unlink()
            tmp_path.replace(target_path)

            n_merged += 1
        except Exception as e:
            print(f"\n[Err] {target_path.name}: {e}")
            n_errors += 1

    print(f"  Gemergt: {n_merged}, Fehler: {n_errors}")

    # Verify an einer Beispiel-Datei
    if matched and n_merged > 0:
        _, sample_target = matched[0]
        d = np.load(sample_target)
        print(f"  [Verify] {sample_target.name}: Felder = {sorted(d.files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rba_dir", default=str(RBA_UPLOAD_DIR / "smiyc"),
                        help="Ordner der die RbA-Track-Unterordner enthält "
                             "mit Unterordnern RoadAnomaly21/ und RoadObstacle21/")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"[Input] RbA-Basis-Ordner: {args.rba_dir}")
    for track in TRACKS:
        merge_track(track, args.rba_dir, args.dry_run)

    print("\n=== Fertig ===")


if __name__ == "__main__":
    main()
