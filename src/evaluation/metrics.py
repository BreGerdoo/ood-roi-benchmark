"""
metrics.py
----------
Pixel-level OoD detection metrics.

Standard metrics used in anomaly segmentation literature
(e.g. SegmentMeIfYouCan benchmark, Fishyscapes benchmark):

  - AUROC   : Area Under ROC Curve
  - AP      : Average Precision (area under P-R curve)
  - FPR95   : False Positive Rate at 95% True Positive Rate

Note on class imbalance:
  Lost & Found has very few OoD pixels per image (~1-3% of total pixels).
  AP is therefore expected to be low even for good detectors.
  FPR95 is the primary metric in the anomaly-segmentation community because
  it is more robust to extreme positive/negative imbalance.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_metrics(
    ood_scores: np.ndarray,   # 1-D array of per-pixel scores (higher = more OoD)
    labels: np.ndarray,       # 1-D binary array: 1 = OoD pixel, 0 = InD pixel
) -> dict:
    """
    Compute AUROC, AP, and FPR@95TPR.

    Parameters
    ----------
    ood_scores : np.ndarray, shape [N]
        Per-pixel OoD score (higher values indicate OoD).
    labels : np.ndarray, shape [N], dtype int
        Binary ground-truth: 1 = OoD, 0 = InD.

    Returns
    -------
    dict with keys: auroc, ap, fpr95
    """
    assert ood_scores.shape == labels.shape, "Shape mismatch between scores and labels."
    assert labels.ndim == 1, "Labels must be 1-D."

    n_pos = labels.sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0:
        raise ValueError("No positive (OoD) pixels found in labels.")
    if n_neg == 0:
        raise ValueError("No negative (InD) pixels found in labels.")

    auroc = roc_auc_score(labels, ood_scores)
    ap    = average_precision_score(labels, ood_scores)
    fpr95 = _fpr_at_tpr(labels, ood_scores, tpr_threshold=0.95)

    return {"auroc": auroc, "ap": ap, "fpr95": fpr95}


def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, tpr_threshold: float = 0.95) -> float:
    """Return FPR at the score threshold that achieves >= tpr_threshold TPR."""
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Find the smallest FPR where TPR >= tpr_threshold
    idx = np.searchsorted(tpr_arr, tpr_threshold)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])


def aggregate_image_results(image_results: list[dict]) -> dict:
    """
    Pool per-image score/label arrays, then compute global metrics.

    This is the standard 'pool-then-evaluate' protocol used in
    Fishyscapes and SegmentMeIfYouCan.

    Parameters
    ----------
    image_results : list of dicts, each with keys 'scores' and 'labels'

    Returns
    -------
    dict with keys: auroc, ap, fpr95, n_images, n_ood_pixels, n_ind_pixels
    """
    all_scores = np.concatenate([r["scores"].ravel() for r in image_results])
    all_labels = np.concatenate([r["labels"].ravel() for r in image_results])

    metrics = compute_metrics(all_scores, all_labels)
    metrics["n_images"]     = len(image_results)
    metrics["n_ood_pixels"] = int(all_labels.sum())
    metrics["n_ind_pixels"] = int((all_labels == 0).sum())
    metrics["imbalance_ratio"] = metrics["n_ood_pixels"] / max(metrics["n_ind_pixels"], 1)
    return metrics
