"""
datasets.py
-----------
Dataset loaders for Cityscapes (InD) and Lost & Found (OoD).

Cityscapes structure expected:
    root/
      leftImg8bit-normal/val/<city>/<city>_<seq>_<frame>_leftImg8bit.png
      gtFine/val/<city>/<city>_<seq>_<frame>_gtFine_labelIds.png

Lost & Found structure expected:
    root/
      leftImg8bit-Objekte/test/<sequence>/<n>_leftImg8bit.png
      gtCoarse/test/<sequence>/<n>_gtCoarse_labelIds.png

Lost & Found label mapping (from official documentation):
    labelId=0        → background / unlabeled / ego vehicle / out of roi (ignore)
    labelId=1        → free road (InD)
    labelId=2–30,32,35,40–43  → obstacles / hazards (OoD, trainId=2)
    labelId=31,33,34,36,37,38,39 → non-hazards (trainId=0, treated as InD)
    labelId=255      → ignore
"""

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch

# Cityscapes label IDs 0-6 and 255 are void/ignore
CITYSCAPES_IGNORE_IDS = set(range(0, 7)) | {255}

# Lost & Found: non-hazard labelIds (trainId=0) — treated as InD, not OoD
# '30'=31, '32'=33, '33'=34, '35'=36, '36'=37, '37'=38, '38'=39
LAF_NON_HAZARD_IDS = {31, 33, 34, 36, 37, 38, 39}

# ImageNet normalisation (used by all torchvision / HuggingFace pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _load_image(path: Path, size: tuple | None = None) -> torch.Tensor:
    """Load RGB image as normalised float tensor [3, H, W]."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)  # size = (H, W)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor


def _laf_ood_mask(label_raw: np.ndarray) -> np.ndarray:
    """
    Return binary OoD mask for a Lost & Found label image.
    True  = obstacle pixel (trainId=2, labelId in 2-30,32,35,40-43)
    False = free road, background, non-hazard, or ignore
    """
    return ((label_raw >= 2) & ~np.isin(label_raw, list(LAF_NON_HAZARD_IDS))).astype(np.uint8)


class CityscapesInDDataset(Dataset):
    """
    Cityscapes validation set — all pixels are InD (ood_label = 0).

    Supports folder names 'leftImg8bit' and 'leftImg8bit-normal'.
    """

    def __init__(self, root: str, split: str = "val", size: tuple | None = None):
        self.size = size
        root = Path(root)

        img_base  = "leftImg8bit-normal" if (root / "leftImg8bit-normal").exists() else "leftImg8bit"
        img_dir   = root / img_base / split
        label_dir = root / "gtFine" / split

        self.samples = []
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem       = img_path.stem.replace("_leftImg8bit", "")
                label_path = label_dir / city_dir.name / f"{stem}_gtFine_labelIds.png"
                if label_path.exists():
                    self.samples.append((img_path, label_path))

        print(f"[CityscapesInD] {len(self.samples)} images (split='{split}')")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image     = _load_image(img_path, self.size)
        label_raw = np.array(Image.open(label_path))

        if self.size:
            label_raw = np.array(
                Image.fromarray(label_raw).resize((self.size[1], self.size[0]), Image.NEAREST)
            )

        valid_mask = np.ones_like(label_raw, dtype=np.uint8)
        for vid in CITYSCAPES_IGNORE_IDS:
            valid_mask[label_raw == vid] = 0

        return {
            "image":      image,
            "ood_label":  np.zeros_like(label_raw, dtype=np.uint8),
            "valid_mask": valid_mask,
            "path":       str(img_path),
        }


class LostAndFoundOoDDataset(Dataset):
    """
    Lost & Found dataset — obstacles are OoD objects.

    Handles nested structure:
        leftImg8bit-Objekte/{split}/{sequence}/{name}_leftImg8bit.png
        gtCoarse/{split}/{sequence}/{name}_gtCoarse_labelIds.png

    OoD label mapping (official Lost & Found label table):
        labelId >= 2, excl. {31,33,34,36,37,38,39} → OoD obstacle (trainId=2)
        labelId == 1                                 → free road (InD)
        labelId == 0                                 → background/ignore
        labelId in {31,33,34,36,37,38,39}            → non-hazard (trainId=0, InD)
        labelId == 255                               → ignore
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        size: tuple | None = None,
        min_ood_pixels: int = 100,
    ):
        self.size = size
        root  = Path(root)

        img_base  = "leftImg8bit-Objekte" if (root / "leftImg8bit-Objekte").exists() else "leftImg8bit"
        img_dir   = root / img_base / split
        label_dir = root / "gtCoarse" / split

        all_img_paths = sorted(img_dir.rglob("*_leftImg8bit.png"))

        self.samples = []
        missing = 0
        for img_path in all_img_paths:
            rel_path   = img_path.relative_to(img_dir)
            stem       = img_path.stem.replace("_leftImg8bit", "")
            label_path = label_dir / rel_path.parent / f"{stem}_gtCoarse_labelIds.png"

            if label_path.exists():
                self.samples.append((img_path, label_path))
            else:
                missing += 1

        print(f"[LostAndFound] Found {len(self.samples)} images "
              f"(split='{split}', {missing} missing labels)")

        if min_ood_pixels > 0:
            filtered = []
            for img_path, label_path in self.samples:
                label = np.array(Image.open(label_path))
                if int(_laf_ood_mask(label).sum()) >= min_ood_pixels:
                    filtered.append((img_path, label_path))
            print(f"[LostAndFound] {len(filtered)} kept (>= {min_ood_pixels} OoD pixels)")
            self.samples = filtered

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image     = _load_image(img_path, self.size)
        label_raw = np.array(Image.open(label_path))

        if self.size:
            label_raw = np.array(
                Image.fromarray(label_raw).resize((self.size[1], self.size[0]), Image.NEAREST)
            )

        return {
            "image":      image,
            "ood_label":  _laf_ood_mask(label_raw),
            # Volles Bild auswerten: die L&F-eigene "valid_mask" (nur annotierter
            # Fahrbahn-Bereich, label_raw>=1) wird BEWUSST ignoriert. Eingeschraenkt
            # wird ausschliesslich durch die eigenen ROI-Varianten A-D (in der
            # Score-Map-/ROI-Pipeline), nie hier im Loader.
            "valid_mask": np.ones_like(label_raw, dtype=np.uint8),
            "path":       str(img_path),
        }
