"""
smiyc_datasets.py
-----------------
Dataset loaders für den SegmentMeIfYouCan (SMIYC) Benchmark:
  - RoadAnomaly21  (semantische Anomalien, ganzes Bild als ROI)
  - RoadObstacle21 (Hindernisse, Straße als offizielle ROI)

Diese Klassen liefern exakt dasselbe Interface wie CityscapesInDDataset /
LostAndFoundOoDDataset in datasets.py:

    {
        "image":      torch.Tensor [3, H, W]  (ImageNet-normalisiert),
        "ood_label":  np.ndarray   [H, W]  uint8   (1 = OoD, 0 = InD),
        "valid_mask": np.ndarray   [H, W]  uint8   (1 = auswerten, 0 = ignore),
        "path":       str,
    }

────────────────────────────────────────────────────────────────────────────
WICHTIG — SMIYC-Label-Encoding (NICHT wie Lost & Found!)
────────────────────────────────────────────────────────────────────────────
Die Ground-Truth liegt in `labels_masks/<name>_labels_semantic.png`:

    0   = in-distribution  (bekannt)
    1   = anomaly/obstacle (OoD)            <-- positiv
    255 = void / ignore

Bei RoadObstacle21 markiert 255 zusätzlich alles AUSSERHALB der Straßen-ROI.
Für den ROI-VARIANTENVERGLEICH (Standard full_image=True) wird diese offizielle
ROI bewusst ignoriert: valid_mask = volles Bild bei BEIDEN Tracks. GT==255 zaehlt
dann als InD (Negativ), nie als OoD. So ist die Auswertungsbasis fuer beide
Datensaetze und alle eigenen ROI-Varianten exakt identisch.

Das unterscheidet sich fundamental vom L&F-Loader (>=2 -> OoD, ==1 -> road).
Den L&F-Loader hier zu verwenden würde die Masken zerstören.

────────────────────────────────────────────────────────────────────────────
Erwartete Ordnerstruktur (offizieller SMIYC-Download)
────────────────────────────────────────────────────────────────────────────
    <root>/
        images/
            <name>.jpg          (oder .png / .webp)
        labels_masks/
            <name>_labels_semantic.png

Beide Tracks haben dieselbe Struktur, nur ein anderer root.
Lege die Datensätze analog zu deinem data/-Ordner ab, z. B.:
    data/smiyc/RoadAnomaly21/
    data/smiyc/RoadObstacle21/
"""

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch

# ImageNet-Normalisierung — identisch zu datasets.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# SMIYC GT-Pixelwerte
SMIYC_INLIER  = 0
SMIYC_OOD     = 1
SMIYC_IGNORE  = 255

# Mögliche Bild-Endungen (SMIYC verteilt je nach Track .jpg/.png/.webp)
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _load_image(path: Path, size: tuple | None = None) -> torch.Tensor:
    """RGB-Bild als normalisierter Float-Tensor [3, H, W]. Identisch zu datasets.py."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)  # size = (H, W)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor


def _find_label(label_dir: Path, stem: str) -> Path | None:
    """
    Finde die GT-Maske zu einem Bild-Stem.
    Offiziell: <stem>_labels_semantic.png
    Fallback : <stem>.png (manche Mirror-Distributionen lassen das Suffix weg)
    """
    cand = label_dir / f"{stem}_labels_semantic.png"
    if cand.exists():
        return cand
    cand = label_dir / f"{stem}.png"
    if cand.exists():
        return cand
    return None


class _SMIYCBase(Dataset):
    """
    Gemeinsame Basis für RoadAnomaly21 und RoadObstacle21.

    Parameters
    ----------
    root : str
        Pfad zum Track-Ordner (enthält images/ und labels_masks/).
    size : tuple (H, W) | None
        Wenn None: Originalauflösung beibehalten (empfohlen, wie bei L&F).
        Score-Maps werden später auf Originalauflösung zurückgerechnet.
    min_ood_pixels : int
        Bilder mit weniger OoD-Pixeln werden verworfen (alle SMIYC-Bilder
        enthalten OoD, daher meist nur ein Sicherheitsnetz).
    full_image : bool
        True  (Standard, EMPFOHLEN für ROI-Variantenvergleich)
              -> valid_mask = komplettes Vollbild (alles 1), bei BEIDEN Tracks
                 identisch. Die offizielle Track-ROI wird IGNORIERT, damit die
                 Basis für alle ROI-Varianten exakt gleich ist.
                 GT==255-Pixel werden als InD (Negativ) behandelt, NIE als OoD.
        False -> valid_mask = (GT != 255), d. h. offizielle ROI des Tracks
                 (bei RO21 = Straße). Nur nutzen, wenn du den offiziellen
                 Benchmark reproduzieren willst, nicht für deinen Variantenvergleich.

    Hinweis zur Fairness:
        Im full_image-Modus ist die Anzahl/Fläche der ausgewerteten Pixel pro
        Bild über beide Datensätze und alle Methoden identisch. OoD bleibt
        ausschließlich GT==1. Damit ist jeder Methoden- und ROI-Vergleich auf
        exakt derselben Grundmenge — keine datensatzseitige ROI kontaminiert A.
    """

    LABEL_SUBDIR = "labels_masks"
    IMAGE_SUBDIR = "images"

    def __init__(
        self,
        root: str,
        size: tuple | None = None,
        min_ood_pixels: int = 1,
        full_image: bool = True,
    ):
        self.size = size
        self.full_image = full_image
        root = Path(root)

        img_dir   = root / self.IMAGE_SUBDIR
        label_dir = root / self.LABEL_SUBDIR

        if not img_dir.exists():
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] images/ nicht gefunden unter {img_dir}. "
                f"Erwartet wird <root>/images/ und <root>/labels_masks/."
            )
        if not label_dir.exists():
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] labels_masks/ nicht gefunden unter {label_dir}."
            )

        # Alle Bilder einsammeln (beliebige Endung)
        all_imgs = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )

        self.samples = []
        missing = 0
        for img_path in all_imgs:
            label_path = _find_label(label_dir, img_path.stem)
            if label_path is not None:
                self.samples.append((img_path, label_path))
            else:
                missing += 1

        print(f"[{self.__class__.__name__}] {len(self.samples)} Bilder gefunden "
              f"({missing} ohne Label).")

        # Optionaler OoD-Pixel-Filter (Konsistenz mit L&F-Loader)
        if min_ood_pixels > 0:
            filtered = []
            for img_path, label_path in self.samples:
                lab = np.array(Image.open(label_path))
                if int((lab == SMIYC_OOD).sum()) >= min_ood_pixels:
                    filtered.append((img_path, label_path))
            print(f"[{self.__class__.__name__}] {len(filtered)} behalten "
                  f"(>= {min_ood_pixels} OoD-Pixel).")
            self.samples = filtered

        roi_desc = ("VOLLES BILD (alles 1; GT==255 zaehlt als InD)"
                    if full_image else "offizielle ROI (GT!=255)")
        print(f"[{self.__class__.__name__}] valid_mask = {roi_desc}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image     = _load_image(img_path, self.size)
        label_raw = np.array(Image.open(label_path))

        # Falls die Maske 3-kanalig gespeichert ist, ersten Kanal nehmen
        if label_raw.ndim == 3:
            label_raw = label_raw[..., 0]

        if self.size:
            label_raw = np.array(
                Image.fromarray(label_raw).resize(
                    (self.size[1], self.size[0]), Image.NEAREST
                )
            )

        ood_label   = (label_raw == SMIYC_OOD).astype(np.uint8)
        ignore_mask = (label_raw == SMIYC_IGNORE)

        if self.full_image:
            # VOLLES BILD: jede Position wird ausgewertet. Identische Basis
            # fuer beide Tracks und alle ROI-Varianten. GT==255 ist KEINE
            # Auswertungssperre mehr, wird aber wegen ood_label==0 dort
            # automatisch als InD (Negativ) behandelt — nie als OoD.
            valid_mask = np.ones_like(ood_label, dtype=np.uint8)
        else:
            # Offizielle Track-ROI (nur fuer Benchmark-Reproduktion).
            valid_mask = (~ignore_mask).astype(np.uint8)

        return {
            "image":      image,
            "ood_label":  ood_label,
            "valid_mask": valid_mask,
            "path":       str(img_path),
        }


class RoadAnomaly21Dataset(_SMIYCBase):
    """
    RoadAnomaly21 — semantische Anomalien. Standard full_image=True:
    valid_mask = volles Bild. GT==255 (unklare Objektraender) zaehlt als InD.
    Identische Auswertungsbasis wie RoadObstacle21 im full_image-Modus.
    """

    def __init__(self, root, size=None, min_ood_pixels=1, full_image=True):
        super().__init__(root, size, min_ood_pixels, full_image)


class RoadObstacle21Dataset(_SMIYCBase):
    """
    RoadObstacle21 — Hindernisse. Im Standard (full_image=True) wird die
    offizielle Straßen-ROI BEWUSST ignoriert: valid_mask = volles Bild, exakt
    wie bei RoadAnomaly21. So ist die Basis fuer deinen ROI-Variantenvergleich
    bei beiden Tracks identisch. GT==255 (abseits der Strasse) zaehlt als InD.

    Nur fuer offizielle Benchmark-Reproduktion: full_image=False.
    """

    def __init__(self, root, size=None, min_ood_pixels=1, full_image=True):
        super().__init__(root, size, min_ood_pixels, full_image)


# Bequemer Dispatch nach Track-Name
def build_smiyc_dataset(track: str, root: str, size=None,
                        min_ood_pixels=1, full_image=True):
    track = track.lower()
    if track in ("roadanomaly21", "anomaly", "ra21"):
        return RoadAnomaly21Dataset(root, size, min_ood_pixels, full_image)
    if track in ("roadobstacle21", "obstacle", "ro21"):
        return RoadObstacle21Dataset(root, size, min_ood_pixels, full_image)
    raise ValueError(f"Unbekannter Track '{track}'. "
                     f"Erlaubt: RoadAnomaly21 | RoadObstacle21")
