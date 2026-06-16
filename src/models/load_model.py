"""
load_model.py
-------------
Cityscapes-pretrained segmentation model for OoD uncertainty evaluation.

Primary:  nvidia/segformer-b2-finetuned-cityscapes-1024-1024
          → requires free HuggingFace account: huggingface-cli login

Fallback: facebook/mask2former-swin-small-cityscapes-semantic
          → fully public, no login required

Both models are trained on Cityscapes (19 classes).

For thesis citation:
  SegFormer: Xie et al., NeurIPS 2021
  Mask2Former: Cheng et al., CVPR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SEGFORMER_ID   = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
MASK2FORMER_ID = "facebook/mask2former-swin-small-cityscapes-semantic"

_loaded_model_type = None   # "segformer" or "mask2former"


def load_deeplabv3_cityscapes(device: str = "cuda") -> tuple:
    """
    Load a Cityscapes-pretrained segmentation model.
    Tries SegFormer-B2 first, falls back to Mask2Former if not authenticated.
    """
    global _loaded_model_type
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Try 1: SegFormer-B2 (requires HuggingFace login)
    # ------------------------------------------------------------------
    try:
        from transformers import SegformerForSemanticSegmentation
        print(f"[Model] Loading SegFormer-B2 (Cityscapes, 19 classes) on: {device}")
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
        _loaded_model_type = "segformer"
        print(f"[Model] Ready — SegFormer-B2 ({SEGFORMER_ID})")

    # ------------------------------------------------------------------
    # Try 2: Mask2Former (fully public, no login needed)
    # ------------------------------------------------------------------
    except Exception as e1:
        print(f"[Model] SegFormer failed ({type(e1).__name__}). Trying Mask2Former...")
        try:
            from transformers import Mask2FormerForUniversalSegmentation
            print(f"[Model] Loading Mask2Former (Cityscapes, 19 classes) on: {device}")
            model = Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER_ID)
            _loaded_model_type = "mask2former"
            print(f"[Model] Ready — Mask2Former ({MASK2FORMER_ID})")
            print("[Model] NOTE for thesis: Using Mask2Former (Cheng et al., CVPR 2022).")
            print("[Model] For SegFormer: run 'huggingface-cli login' with a free HF account.")
        except Exception as e2:
            raise RuntimeError(
                f"Could not load any model.\n"
                f"  SegFormer error:   {e1}\n"
                f"  Mask2Former error: {e2}\n\n"
                f"Fix: pip install huggingface_hub && huggingface-cli login"
            )

    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return model, device


def forward_logits(model: nn.Module, image: torch.Tensor, device) -> torch.Tensor:
    """
    Run image through model, return per-pixel logits [C, H, W].
    Handles SegFormer (upsamples from 1/4 res) and Mask2Former output formats.
    """
    image = image.unsqueeze(0).to(device)   # [1, 3, H, W]
    H, W  = image.shape[2], image.shape[3]

    with torch.no_grad():
        output = model(pixel_values=image)

    # SegFormer: output.logits = [1, C, H/4, W/4]
    if hasattr(output, "logits") and output.logits.dim() == 4:
        logits_small = output.logits[0]     # [C, H/4, W/4]
        logits = F.interpolate(
            logits_small.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0]                                # [C, H, W]
        return logits

    # Mask2Former: use class_queries_logits + masks_queries_logits
    # Combine into dense per-pixel logits via softmax over class queries
    if hasattr(output, "masks_queries_logits"):
        # masks: [1, Q, H/4, W/4], class_logits: [1, Q, C+1]
        mask_logits  = output.masks_queries_logits[0]       # [Q, H/4, W/4]
        class_logits = output.class_queries_logits[0]       # [Q, C+1]

        # Upsample masks to full resolution
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        )[0]                                                 # [Q, H, W]

        # Combine: for each pixel, weighted sum over queries
        # Remove void class (last column)
        class_probs = class_logits[:, :-1].softmax(dim=-1)  # [Q, C]
        mask_probs  = mask_logits.sigmoid()                  # [Q, H, W]

        # Dense logits: [C, H, W]
        logits = torch.einsum("qc,qhw->chw", class_probs, mask_probs)
        return logits

    raise TypeError(f"Unknown model output format: {type(output)}")
