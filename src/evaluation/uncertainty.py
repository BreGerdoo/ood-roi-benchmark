"""
uncertainty.py
--------------
Pixel-wise OoD uncertainty scores for semantic segmentation.
All functions take raw logits (torch.Tensor, shape [C, H, W]) and
return a 2D numpy array of shape [H, W] where HIGHER = more OoD.
"""

import numpy as np
import torch
import torch.nn.functional as F


def msp_score(logits: torch.Tensor) -> np.ndarray:
    """
    Maximum Softmax Probability (MSP) score.
    Hendrycks & Gimpel (2017), ICLR.

    OoD score = 1 - max_c softmax(logits)_c
    Higher means more uncertain / more OoD.
    """
    probs = F.softmax(logits, dim=0)          # [C, H, W]
    max_prob, _ = probs.max(dim=0)            # [H, W]
    return (1.0 - max_prob).cpu().numpy()


def entropy_score(logits: torch.Tensor) -> np.ndarray:
    """
    Predictive entropy of the softmax distribution.

    OoD score = -sum_c p_c * log(p_c + eps)
    Higher means more uncertain / more OoD.
    """
    probs = F.softmax(logits, dim=0)          # [C, H, W]
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=0) # [H, W]
    return entropy.cpu().numpy()


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Energy-based OoD score.
    Liu et al. (2020), NeurIPS.

    OoD score = -T * log sum_c exp(logits_c / T)
    Negated so that HIGHER = more OoD (consistent with MSP/entropy).
    """
    energy = -temperature * torch.logsumexp(logits / temperature, dim=0)  # [H, W]
    return energy.cpu().numpy()


SCORE_FUNCTIONS = {
    "msp":     msp_score,
    "entropy": entropy_score,
    "energy":  energy_score,
}
