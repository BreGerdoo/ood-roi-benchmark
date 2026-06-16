"""
measure_segformer_iou.py
------------------------
Berechnet die per-Klasse IoU für SegFormer-B2 auf Cityscapes val.
Ergebnis wird für Kapitel 4 der Bachelorarbeit benötigt.

Aufruf:
    python measure_segformer_iou.py

Ausgabe: Tabelle mit IoU pro Klasse + mIoU.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from dataloaders.laf_datasets import CityscapesInDDataset
from paths import DATA_CITYSCAPES, RESULTS_DIR

SEGFORMER_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

CS_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]
NUM_CLASSES = 19

# Cityscapes labelId → trainId mapping
LABEL_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}


def labelid_to_trainid(label_img):
    """Convert Cityscapes labelIds to trainIds (255 = ignore)."""
    out = np.full_like(label_img, 255, dtype=np.uint8)
    for lid, tid in LABEL_TO_TRAIN.items():
        out[label_img == lid] = tid
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cs_root = str(DATA_CITYSCAPES)

    print(f"[Model] Lade {SEGFORMER_ID} ...")
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
    model.to(device).eval()

    ds = CityscapesInDDataset(root=cs_root, split="val", size=None)

    # Confusion matrix
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for i in tqdm(range(len(ds)), desc="Evaluating"):
        sample = ds[i]
        img = sample["image"].unsqueeze(0).to(device)

        # Build label path from image path
        img_path = Path(sample["path"])
        stem = img_path.stem.replace("_leftImg8bit", "")
        city = img_path.parent.name
        label_path = Path(cs_root) / "gtFine" / "val" / city / f"{stem}_gtFine_labelIds.png"
        gt_raw = np.array(Image.open(label_path))
        gt_train = labelid_to_trainid(gt_raw)

        H, W = gt_train.shape

        # Inference at 512x1024 for speed, upsample back
        img_input = F.interpolate(img, size=(512, 1024), mode="bilinear",
                                  align_corners=False)
        with torch.no_grad():
            logits = model(pixel_values=img_input).logits
            logits_full = F.interpolate(logits, size=(H, W), mode="bilinear",
                                        align_corners=False)
        pred = logits_full.argmax(dim=1)[0].cpu().numpy()

        # Only count valid pixels (trainId != 255)
        valid = gt_train != 255
        pred_v = pred[valid]
        gt_v = gt_train[valid]

        for p, g in zip(pred_v, gt_v):
            conf[g, p] += 1

    # Compute per-class IoU
    print("\n" + "=" * 50)
    print(f"  SegFormer-B2 — Per-Klasse IoU (Cityscapes val)")
    print("=" * 50)

    ious = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        iou = tp / max(tp + fp + fn, 1)
        ious.append(iou)
        marker = "  ◄" if c == 0 else ""
        print(f"  {CS_CLASSES[c]:<16} {iou * 100:6.2f}%{marker}")

    miou = np.mean(ious)
    print(f"\n  {'mIoU':<16} {miou * 100:6.2f}%")
    print("=" * 50)
    print(f"\n  Road IoU: {ious[0] * 100:.1f}%")
    print(f"  Diesen Wert in Kapitel 4 verwenden.")

    # CSV fuer die Arbeit (Tabelle 8)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "segformer_iou.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class,iou\n")
        for c in range(NUM_CLASSES):
            f.write(f"{CS_CLASSES[c]},{ious[c]:.4f}\n")
        f.write(f"mIoU,{miou:.4f}\n")
    print(f"  [Saved] {csv_path}")


if __name__ == "__main__":
    main()
