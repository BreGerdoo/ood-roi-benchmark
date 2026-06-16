"""
compute_score_maps_smiyc.py
---------------------------
Wie compute_score_maps.py, aber für die SegmentMeIfYouCan-Tracks
(RoadAnomaly21, RoadObstacle21) statt Lost & Found.

Berechnet pro Bild die lokalen Score-Maps (Energy, DINOv2 kNN, MSP, Pred-
Klassen, PixOOD) und speichert sie als .npz. RbA wird später separat über
Colab berechnet und via merge_rba_into_score_maps_smiyc.py reingemischt.

Wichtig:
  - Nutzt den SMIYC-Loader (smiyc_datasets.py), der ood_label bereits
    korrekt als 0/1 liefert (1=OoD, 255 wird zu InD=0). Der full_image-Modus
    setzt valid_mask überall auf 1 (identische Basis für alle ROI-Varianten).
  - Eigener Output-Ordner PRO Track, damit sich die beiden Tracks nicht
    überschreiben (Dateinamen wie validation0000 wären sonst kollidierend
    nicht — aber getrennte Ordner sind sauberer und eindeutig).

Aufruf:
    python compute_score_maps_smiyc.py --track RoadAnomaly21
    python compute_score_maps_smiyc.py --track RoadObstacle21
    python compute_score_maps_smiyc.py --track RoadAnomaly21 --skip_existing
    python compute_score_maps_smiyc.py --track RoadObstacle21 --skip_pixood

Output:
    results/smiyc/<Track>/score_maps/<image_stem>.npz
        - energy_map:  float16 [H, W]
        - knn_map:     float16 [H, W]
        - msp_map:     float16 [H, W]
        - pixood_map:  float16 [H, W]
        - pred_class:  int8    [H, W]
        - ood_label:   int8    [H, W]   (1=OoD, 0=InD)
        - shape:       (H, W)
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from dataloaders.smiyc_datasets import build_smiyc_dataset
from paths import DATA_SMIYC, GALLERY_PATH, SMIYC_RESULTS_DIR, PIXOOD_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Track -> Daten-Ordner. Passe die Pfade an deine Ablage an, falls anders.
SMIYC_ROOTS = DATA_SMIYC

# PixOOD-Repo liegt als Schwester-Ordner neben dem Repo-Root (siehe paths.py)

SEGFORMER_ID    = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
DINOV2_MODEL    = "dinov2_vitb14"
DINO_INPUT_SIZE = 518
NUM_PATCHES_1D  = DINO_INPUT_SIZE // 14   # 37
KNN_K           = 10


# ---------------------------------------------------------------------------
# PixOOD-Loader (identisch zum L&F-Skript)
# ---------------------------------------------------------------------------
def load_pixood():
    if not PIXOOD_DIR.exists():
        raise FileNotFoundError(
            f"PixOOD-Ordner nicht gefunden: {PIXOOD_DIR}\n"
            "Bitte das Repo dorthin klonen oder PIXOOD_DIR anpassen."
        )
    original_cwd = os.getcwd()
    os.chdir(PIXOOD_DIR)
    sys.path.insert(0, str(PIXOOD_DIR))
    from pixood import PixOOD
    pixood = PixOOD(".", eval_labels=[])
    os.chdir(original_cwd)
    return pixood


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True,
                        choices=["RoadAnomaly21", "RoadObstacle21"],
                        help="Welcher SMIYC-Track")
    parser.add_argument("--root", type=str, default=None,
                        help="Override Daten-Ordner (sonst Default aus SMIYC_ROOTS)")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_pixood", action="store_true")
    args = parser.parse_args()

    # Daten-Ordner bestimmen
    root = Path(args.root) if args.root else SMIYC_ROOTS[args.track]
    if not root.exists():
        print(f"[Error] Daten-Ordner nicht gefunden: {root}")
        print(f"        Passe --root an oder lege die Daten dort ab.")
        sys.exit(1)

    output_dir = SMIYC_RESULTS_DIR / args.track / "score_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset ===
    # full_image=True -> valid_mask überall 1, identische Basis für alle Varianten
    ds = build_smiyc_dataset(args.track, str(root), size=None,
                             min_ood_pixels=1, full_image=True)
    n = len(ds) if args.max_images < 0 else min(args.max_images, len(ds))
    print(f"[Dataset] {args.track}: {n} Bilder")

    # === SegFormer ===
    print(f"[SegFormer] Lade {SEGFORMER_ID} ...")
    from transformers import SegformerForSemanticSegmentation
    segformer = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
    segformer.to(device).eval()

    # === DINOv2 ===
    print(f"[DINOv2] Lade {DINOV2_MODEL} ...")
    dino = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL)
    dino.to(device).eval()

    # === Gallery ===
    print(f"[Gallery] Lade {GALLERY_PATH} ...")
    gallery = torch.load(GALLERY_PATH, map_location=device).float()
    gallery = F.normalize(gallery, dim=1)

    # === PixOOD ===
    pixood = None
    if not args.skip_pixood:
        print(f"[PixOOD] Lade aus {PIXOOD_DIR} ...")
        pixood = load_pixood()
        print("[PixOOD] Bereit.")

    # === Main loop ===
    print(f"\n[Compute] Berechne Score-Maps für {n} Bilder ({args.track}) ...")
    n_skipped   = 0
    n_processed = 0

    for idx in tqdm(range(n), desc=f"Score-Maps {args.track}"):
        sample   = ds[idx]
        stem     = Path(sample["path"]).stem
        out_path = output_dir / f"{stem}.npz"

        if args.skip_existing and out_path.exists():
            try:
                existing = np.load(out_path)
                required = {"energy_map", "knn_map", "msp_map", "pred_class", "ood_label"}
                if not args.skip_pixood:
                    required.add("pixood_map")
                if required.issubset(set(existing.files)):
                    n_skipped += 1
                    continue
            except Exception:
                pass

        img_tensor = sample["image"]               # [3, H, W]
        H, W       = sample["ood_label"].shape

        # SMIYC-Loader liefert ood_label bereits als 0/1.
        # Der >1->0 Cleanup ist harmlos (greift bei SMIYC nicht), wird aus
        # Konsistenz mit dem L&F-Skript beibehalten.
        ood_label = np.where(
            sample["ood_label"] > 1, 0, sample["ood_label"]
        ).astype(np.int8)

        # --- SegFormer Inferenz ---
        img_input = F.interpolate(
            img_tensor.unsqueeze(0).to(device),
            size=(512, 1024), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            outputs = segformer(pixel_values=img_input)
            logits  = F.interpolate(outputs.logits, size=(H, W),
                                    mode="bilinear", align_corners=False)

        logits_sq  = logits.squeeze(0)                # [19, H, W]
        softmax    = torch.softmax(logits_sq, dim=0)
        pred_class = softmax.argmax(dim=0).cpu().numpy().astype(np.int8)
        msp_map    = softmax.max(dim=0).values.cpu().numpy().astype(np.float16)

        energy_map = (-torch.logsumexp(logits_sq, dim=0)).cpu().numpy().astype(np.float16)

        del logits, logits_sq, softmax, outputs, img_input

        # --- DINOv2 Inferenz ---
        dino_input = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
            mode="bilinear", align_corners=False
        ).to(device)

        with torch.no_grad():
            dino_out = dino.get_intermediate_layers(
                dino_input, n=1, return_class_token=False
            )[0].squeeze(0)   # [1369, 768]

        dino_out_norm = F.normalize(dino_out, dim=1)
        sims          = torch.mm(dino_out_norm, gallery.T)
        topk_sims, _  = sims.topk(KNN_K, dim=1)
        patch_scores  = (1.0 - topk_sims).mean(dim=1)

        knn_map = F.interpolate(
            patch_scores.reshape(1, 1, NUM_PATCHES_1D, NUM_PATCHES_1D).float(),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy().astype(np.float16)

        del dino_input, dino_out, dino_out_norm, sims, topk_sims, patch_scores

        # --- Speichern vorbereiten ---
        save_dict = {
            "energy_map": energy_map,
            "knn_map":    knn_map,
            "msp_map":    msp_map,
            "pred_class": pred_class,
            "ood_label":  ood_label,
            "shape":      np.array([H, W], dtype=np.int32),
        }

        # --- PixOOD Inferenz ---
        if pixood is not None:
            from PIL import Image
            img_pil = Image.open(sample["path"]).convert("RGB")
            with torch.no_grad():
                pixood_score = pixood.evaluate(img_pil)
            if isinstance(pixood_score, torch.Tensor):
                pixood_map = pixood_score.cpu().numpy()
            else:
                pixood_map = np.asarray(pixood_score)

            if pixood_map.shape != (H, W):
                pmap_t = torch.from_numpy(pixood_map).float()[None, None]
                pixood_map = F.interpolate(
                    pmap_t, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze().numpy()

            save_dict["pixood_map"] = pixood_map.astype(np.float16)

        np.savez_compressed(out_path, **save_dict)
        n_processed += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[Done] {n_processed} verarbeitet, {n_skipped} übersprungen")
    print(f"[Done] Score-Maps in: {output_dir}/")


if __name__ == "__main__":
    main()
