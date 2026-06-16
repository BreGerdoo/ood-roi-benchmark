"""
compute_score_maps.py
---------------------
Berechnet pro Bild die Score-Maps (Energy, DINOv2 kNN, MSP, Pred-Klassen,
PixOOD) und speichert sie als kompakte .npz-Dateien auf der Festplatte.

Dieser Schritt ist teuer (~2-3h für 1096 Bilder), läuft aber nur einmal.
Danach können beliebig viele ROI-Varianten und Metriken in Sekunden
ausgewertet werden — ohne Modell-Inferenz.

Aufruf:
    python compute_score_maps.py
    python compute_score_maps.py --max_images 50    # Schnelltest
    python compute_score_maps.py --skip_existing    # nur fehlende Bilder
    python compute_score_maps.py --skip_pixood     # ohne PixOOD

Output:
    results/roi_variants/score_maps/<image_stem>.npz
        - energy_map:  float16 [H, W]
        - knn_map:     float16 [H, W]
        - msp_map:     float16 [H, W]
        - pixood_map:  float16 [H, W]   (NEU)
        - pred_class:  int8    [H, W]
        - ood_label:   int8    [H, W]   (255 → 0)
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

from dataloaders.laf_datasets import LostAndFoundOoDDataset
from paths import DATA_LAF, GALLERY_PATH, SCORE_MAPS_LAF, PIXOOD_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAF_ROOT   = str(DATA_LAF)
OUTPUT_DIR = SCORE_MAPS_LAF

# PixOOD-Repo liegt als Schwester-Ordner neben dem Repo-Root (siehe paths.py)

SEGFORMER_ID    = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
DINOV2_MODEL    = "dinov2_vitb14"
DINO_INPUT_SIZE = 518
NUM_PATCHES_1D  = DINO_INPUT_SIZE // 14   # 37
KNN_K           = 10


# ---------------------------------------------------------------------------
# PixOOD-Loader
# ---------------------------------------------------------------------------
def load_pixood():
    """
    Lädt PixOOD aus dem Schwester-Ordner.
    PixOOD muss aus seinem eigenen Verzeichnis heraus initialisiert werden,
    da es relative Pfade für Checkpoints verwendet.
    """
    if not PIXOOD_DIR.exists():
        raise FileNotFoundError(
            f"PixOOD-Ordner nicht gefunden: {PIXOOD_DIR}\n"
            "Bitte das Repo dorthin klonen oder PIXOOD_DIR anpassen."
        )

    # cwd temporär auf PixOOD-Ordner umstellen
    original_cwd = os.getcwd()
    os.chdir(PIXOOD_DIR)

    sys.path.insert(0, str(PIXOOD_DIR))
    from pixood import PixOOD
    pixood = PixOOD(".", eval_labels=[])

    # cwd zurücksetzen
    os.chdir(original_cwd)
    return pixood


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=-1,
                        help="-1 = alle Bilder")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Bilder mit vorhandener .npz-Datei überspringen (nur wenn alle Felder vorhanden)")
    parser.add_argument("--skip_pixood", action="store_true",
                        help="PixOOD nicht laden/berechnen (nur Energy + kNN + MSP)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset ===
    ds = LostAndFoundOoDDataset(root=LAF_ROOT, split="test", size=None,
                                min_ood_pixels=100)
    n = len(ds) if args.max_images < 0 else min(args.max_images, len(ds))
    print(f"[Dataset] {n} Bilder")

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
    print(f"\n[Compute] Berechne Score-Maps für {n} Bilder ...")
    n_skipped   = 0
    n_processed = 0

    for idx in tqdm(range(n), desc="Score-Maps"):
        sample   = ds[idx]
        stem     = Path(sample["path"]).stem
        out_path = OUTPUT_DIR / f"{stem}.npz"

        # Prüfen ob existierende Datei alle benötigten Felder hat
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
                pass  # bei Lesefehler einfach neu berechnen

        img_tensor = sample["image"]               # [3, H, W]
        H, W       = sample["ood_label"].shape

        # OoD-Label: 255 → 0
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

        # GPU-Speicher freigeben
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

        # --- PixOOD Inferenz ---
        save_dict = {
            "energy_map": energy_map,
            "knn_map":    knn_map,
            "msp_map":    msp_map,
            "pred_class": pred_class,
            "ood_label":  ood_label,
            "shape":      np.array([H, W], dtype=np.int32),
        }

        if pixood is not None:
            from PIL import Image
            img_pil = Image.open(sample["path"]).convert("RGB")
            with torch.no_grad():
                pixood_score = pixood.evaluate(img_pil)
            if isinstance(pixood_score, torch.Tensor):
                pixood_map = pixood_score.cpu().numpy()
            else:
                pixood_map = np.asarray(pixood_score)

            # Auf Bildauflösung resizen falls nötig
            if pixood_map.shape != (H, W):
                pmap_t = torch.from_numpy(pixood_map).float()[None, None]
                pixood_map = F.interpolate(
                    pmap_t, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze().numpy()

            save_dict["pixood_map"] = pixood_map.astype(np.float16)

        # --- Speichern ---
        np.savez_compressed(out_path, **save_dict)
        n_processed += 1

        # GPU-Cache leeren
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[Done] {n_processed} verarbeitet, {n_skipped} übersprungen")
    print(f"[Done] Score-Maps in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
