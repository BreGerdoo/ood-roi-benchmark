"""
merge_rba_into_score_maps.py
----------------------------
Merged die von Colab berechneten RbA Score-Maps in die bestehenden
score_maps/*.npz Dateien.

Eingabe:  rba_score_maps/<stem>_rba.npz mit field "rba_map"
Bestehend: results/roi_variants/score_maps/<stem>.npz
Output:    Selbe Datei mit zusätzlichem field "rba_map"

Aufruf:
    python merge_rba_into_score_maps.py
    python merge_rba_into_score_maps.py --dry_run     # nur prüfen
    python merge_rba_into_score_maps.py --rba_dir <path>
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from paths import RBA_UPLOAD_DIR, SCORE_MAPS_LAF

# Defaults
DEFAULT_RBA_DIR    = RBA_UPLOAD_DIR / "laf"
DEFAULT_TARGET_DIR = SCORE_MAPS_LAF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rba_dir", type=str, default=str(DEFAULT_RBA_DIR),
                        help=f"Ordner mit den <stem>_rba.npz Dateien (default: {DEFAULT_RBA_DIR})")
    parser.add_argument("--target_dir", type=str, default=str(DEFAULT_TARGET_DIR),
                        help=f"Bestehender score_maps-Ordner (default: {DEFAULT_TARGET_DIR})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Nur prüfen welche Dateien gemergt würden, nichts schreiben")
    args = parser.parse_args()

    rba_dir    = Path(args.rba_dir)
    target_dir = Path(args.target_dir)

    if not rba_dir.exists():
        print(f"[Error] RbA-Ordner nicht gefunden: {rba_dir}")
        sys.exit(1)
    if not target_dir.exists():
        print(f"[Error] Score-Maps-Ordner nicht gefunden: {target_dir}")
        sys.exit(1)

    # RbA-Dateien einlesen
    rba_files = sorted(rba_dir.glob("*_rba.npz"))
    print(f"[Input]  RbA-Files:        {len(rba_files)}")

    target_files = sorted(target_dir.glob("*.npz"))
    print(f"[Input]  Bestehende Files: {len(target_files)}")

    # Map: stem -> target_path
    target_map = {f.stem: f for f in target_files}

    # Matching prüfen
    # RbA-Stem:    "02_Hanns_Klemm_Str_44_000000_000020_rba"
    # Target-Stem: "02_Hanns_Klemm_Str_44_000000_000020_leftImg8bit"
    # → Wir matchen über den gemeinsamen Präfix (alles vor _rba bzw. _leftImg8bit)
    matched   = []
    unmatched = []
    for rba_path in rba_files:
        # "..._rba" entfernen → reiner Bild-Stem
        rba_base = rba_path.stem.replace("_rba", "")
        # Im Target: stem hat "_leftImg8bit" Suffix
        target_candidate = rba_base + "_leftImg8bit"
        if target_candidate in target_map:
            matched.append((rba_path, target_map[target_candidate]))
        elif rba_base in target_map:
            # Fallback: ohne _leftImg8bit Suffix
            matched.append((rba_path, target_map[rba_base]))
        else:
            unmatched.append(rba_path)

    print(f"\n[Match]  Matching:   {len(matched)}/{len(rba_files)}")
    print(f"[Match]  Unmatched:  {len(unmatched)}")

    if unmatched:
        print("\nUnmatched RbA-Files (erste 5):")
        for u in unmatched[:5]:
            print(f"  {u.name}")
        print("Bitte prüfen ob die Dateinamen-Konvention passt.")

    if args.dry_run:
        print("\n[DryRun] Beende ohne Schreiben.")
        return

    # === Merging ===
    print(f"\n[Merge]  Starte Merging von {len(matched)} Dateien ...")
    n_merged       = 0
    n_already_has  = 0
    n_errors       = 0

    for rba_path, target_path in tqdm(matched, desc="Merge"):
        try:
            # Beide laden — alle Felder sofort in den Speicher kopieren und
            # File-Handles schließen, sonst hält Windows die target-Datei
            # während des replace() noch fest
            with np.load(rba_path) as rba_data:
                rba_map = np.array(rba_data["rba_map"])

            with np.load(target_path) as target_data:
                existing_fields = list(target_data.files)
                new_dict = {k: np.array(target_data[k]) for k in existing_fields}

            # Shape-Konsistenzprüfung
            if "ood_label" in existing_fields:
                target_shape = new_dict["ood_label"].shape
                if target_shape != rba_map.shape:
                    print(f"\n[Warn] Shape-Mismatch bei {target_path.name}: "
                          f"target={target_shape} vs rba={rba_map.shape}")
                    n_errors += 1
                    continue

            already_has = "rba_map" in existing_fields
            if already_has:
                n_already_has += 1

            new_dict["rba_map"] = rba_map

            # Atomar speichern (erst neue Datei, dann umbenennen)
            # WICHTIG: np.savez_compressed hängt automatisch .npz an, daher
            # speichern wir ohne Suffix und benennen die fertige .npz Datei um
            tmp_base = target_path.with_name(target_path.stem + "_tmp")
            np.savez_compressed(tmp_base, **new_dict)
            tmp_path = tmp_base.with_suffix(".npz")
            # Vorhandene Zieldatei löschen vor dem rename (Windows-Problem)
            if target_path.exists():
                target_path.unlink()
            tmp_path.replace(target_path)

            n_merged += 1

        except Exception as e:
            print(f"\n[Err] {target_path.name}: {e}")
            n_errors += 1

    print(f"\n=== Done ===")
    print(f"Gemergt:                  {n_merged}")
    print(f"Davon hatten schon rba_map: {n_already_has}")
    print(f"Fehler:                   {n_errors}")

    # Verifikation: alle Felder einer zufälligen Datei
    if n_merged > 0:
        sample = matched[0][1]
        d = np.load(sample)
        print(f"\n[Verify] Beispiel-Datei: {sample.name}")
        print(f"[Verify] Felder: {sorted(d.files)}")
        for k in sorted(d.files):
            v = d[k]
            print(f"  {k:12s}  shape={v.shape}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
