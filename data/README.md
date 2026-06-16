# Datasets

The datasets are **not** part of this repository. Place them under `data/` exactly as
shown below — the loaders (`src/dataloaders/`) and `src/paths.py` expect this structure.

## 1. Cityscapes (InD)  →  `data/id/`

Register and download from https://www.cityscapes-dataset.com/downloads/
(`leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`):

```
data/id/
├── leftImg8bit-normal/          # or "leftImg8bit" — both folder names are detected
│   ├── train/<city>/<city>_<seq>_<frame>_leftImg8bit.png
│   └── val/<city>/...
└── gtFine/
    ├── train/<city>/...
    └── val/<city>/<city>_<seq>_<frame>_gtFine_labelIds.png
```

## 2. Lost & Found (OoD)  →  `data/ood/`

Download from http://www.6d-vision.com/lostandfounddataset
(`leftImg8bit` + `gtCoarse`, test split):

```
data/ood/
├── leftImg8bit-Objekte/         # or "leftImg8bit" — both are detected
│   └── test/<sequence>/<name>_leftImg8bit.png
└── gtCoarse/
    └── test/<sequence>/<name>_gtCoarse_labelIds.png
```

Label convention (handled by the loader): labelId ≥ 2 = obstacle (OoD), excluding the
seven non-hazard ids {31, 33, 34, 36, 37, 38, 39}; labelId 1 = free road; 0/255 = ignore.

## 3. SegmentMeIfYouCan (SMIYC)  →  `data/smiyc/`

Validation sets with public labels:
- RoadAnomaly21:  https://zenodo.org/record/5270237
- RoadObstacle21: https://zenodo.org/record/5281633

Keep the official archive folder names:

```
data/smiyc/
├── dataset_AnomalyTrack/            # RoadAnomaly21
│   ├── images/<name>.jpg
│   └── labels_masks/<name>_labels_semantic.png
└── dataset_ObstacleTrack/           # RoadObstacle21
    ├── images/<name>.webp
    └── labels_masks/<name>_labels_semantic.png
```

GT encoding: `0 = InD`, `1 = anomaly/obstacle (OoD)`, `255 = void/ignore`.
For the ROI-variant comparison the official SMIYC ROI is deliberately ignored
(`valid_mask` = all ones); GT=255 counts as InD/negative. See README §7.

## 4. Precomputed score-map caches (optional, recommended)

```
python scripts/download_score_maps.py
```
unpacks the Zenodo archive into:

```
results/roi_variants/score_maps/<image_stem>.npz                # Lost & Found
results/smiyc/RoadAnomaly21/score_maps/<image_stem>.npz
results/smiyc/RoadObstacle21/score_maps/<image_stem>.npz
```

Each `.npz` contains `energy_map`, `knn_map`, `msp_map`, `pixood_map`, `rba_map`
(float16, [H, W]), `pred_class`, `ood_label` and `shape`.
