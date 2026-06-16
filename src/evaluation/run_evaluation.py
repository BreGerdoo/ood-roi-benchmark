"""
run_evaluation.py
-----------------
Reproducible evaluation pipeline for Chapter 2.3.

Can be called two ways:
  1. Via main.py:  from evaluation.run_evaluation import run
  2. Directly:     python evaluation/run_evaluation.py (uses configs/eval_config.yaml)
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add project root to sys.path so all sibling packages are importable
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from evaluation.uncertainty import SCORE_FUNCTIONS
from evaluation.metrics import aggregate_image_results
from dataloaders.laf_datasets import CityscapesInDDataset, LostAndFoundOoDDataset
from models.load_model import load_deeplabv3_cityscapes, forward_logits

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset, model, device, score_names, max_images=-1, desc="Evaluating"):
    """Run all uncertainty scores over a dataset."""
    results = {name: [] for name in score_names}
    n = len(dataset) if max_images < 0 else min(max_images, len(dataset))

    for idx in tqdm(range(n), desc=desc):
        sample     = dataset[idx]
        image      = sample["image"]
        ood_label  = sample["ood_label"]
        valid_mask = sample["valid_mask"]

        logits = forward_logits(model, image, device)   # [C, H, W]

        for score_name in score_names:
            score_map = SCORE_FUNCTIONS[score_name](logits)   # [H, W] numpy
            valid = valid_mask.astype(bool)
            results[score_name].append({
                "scores": score_map[valid].astype(np.float32),
                "labels": ood_label[valid].astype(np.int32),
                "path":   sample["path"],
            })

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_distributions(ind_results, ood_results, score_names, output_path):
    fig, axes = plt.subplots(1, len(score_names), figsize=(6 * len(score_names), 4))
    if len(score_names) == 1:
        axes = [axes]

    for ax, score_name in zip(axes, score_names):
        ind_scores = np.concatenate([r["scores"] for r in ind_results[score_name]])
        ood_obstacle = np.concatenate([
            r["scores"][r["labels"] == 1] for r in ood_results[score_name]
            if (r["labels"] == 1).sum() > 0
        ])
        ood_background = np.concatenate([
            r["scores"][r["labels"] == 0] for r in ood_results[score_name]
        ])

        bins = 80
        ax.hist(ind_scores,     bins=bins, alpha=0.5, density=True, label="Cityscapes InD",       color="steelblue")
        ax.hist(ood_background, bins=bins, alpha=0.5, density=True, label="LAF background (InD)", color="orange")
        ax.hist(ood_obstacle,   bins=bins, alpha=0.7, density=True, label="LAF obstacle (OoD)",   color="crimson")
        ax.set_title(score_name.upper())
        ax.set_xlabel("OoD Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Score Distributions: Cityscapes vs Lost & Found", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


# ---------------------------------------------------------------------------
# run() — called from main.py
# ---------------------------------------------------------------------------

def run(config: dict):
    set_seed(config.get("seed", SEED))

    # Relative Pfade (output + Daten) gegen den Repo-Root aufloesen,
    # damit der Aufruf aus jedem Arbeitsverzeichnis funktioniert.
    output_dir = config.get("output_dir", "results/baseline")
    if not Path(output_dir).is_absolute():
        output_dir = str(PROJECT_ROOT / output_dir)
    for key in ("laf_root", "cityscapes_root"):
        if key in config["data"] and not Path(config["data"][key]).is_absolute():
            config["data"][key] = str(PROJECT_ROOT / config["data"][key])
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved to {output_dir}/config.json")

    data_cfg   = config["data"]
    size       = tuple(data_cfg["img_size"]) if data_cfg.get("img_size") else None
    max_images = data_cfg.get("max_images", 200)
    score_names = config.get("scores", ["msp", "entropy", "energy"])

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, device = load_deeplabv3_cityscapes(config.get("device", "cpu"))

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    cityscapes_ds = CityscapesInDDataset(
        root=data_cfg["cityscapes_root"],
        split=data_cfg.get("split", "val"),
        size=size,
    )
    laf_ds = LostAndFoundOoDDataset(
        root=data_cfg["laf_root"],
        size=size,
        min_ood_pixels=100,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n=== Evaluating Cityscapes (InD baseline) ===")
    ind_results = evaluate_dataset(
        cityscapes_ds, model, device, score_names,
        max_images=max_images, desc="Cityscapes"
    )

    print("\n=== Evaluating Lost & Found (OoD benchmark) ===")
    ood_results = evaluate_dataset(
        laf_ds, model, device, score_names,
        max_images=max_images, desc="Lost & Found"
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\n=== Computing Metrics ===")
    rows = []
    for score_name in score_names:
        agg = aggregate_image_results(ood_results[score_name])
        rows.append({
            "Score":           score_name.upper(),
            "AUROC":           round(agg["auroc"], 4),
            "AP":              round(agg["ap"],    4),
            "FPR95":           round(agg["fpr95"], 4),
            "N Images":        agg["n_images"],
            "OoD Pixels":      agg["n_ood_pixels"],
            "InD Pixels":      agg["n_ind_pixels"],
            "Imbalance Ratio": f"1 : {1/agg['imbalance_ratio']:.0f}",
        })
        print(
            f"  {score_name.upper():8s} | AUROC={agg['auroc']:.4f} | "
            f"AP={agg['ap']:.4f} | FPR95={agg['fpr95']:.4f} | "
            f"OoD pixels: {agg['n_ood_pixels']:,}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "metrics_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Results] Saved: {csv_path}")

    tex_cols = ["Score", "AUROC", "AP", "FPR95", "Imbalance Ratio"]
    tex_path = os.path.join(output_dir, "metrics_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by run_evaluation.py\n")
        f.write("% Dataset: Cityscapes (InD) vs. Lost & Found (OoD)\n")
        f.write("% Model: SegFormer-B2 (Cityscapes)\n\n")
        f.write(df[tex_cols].to_latex(index=False, float_format="%.4f"))
    print(f"[Results] Saved: {tex_path}")

    plot_score_distributions(
        ind_results, ood_results, score_names,
        output_path=os.path.join(output_dir, "score_distributions.png"),
    )

    # Summary table (analogue to Table 1 in thesis)
    summary_rows = []
    for score_name in score_names:
        ind_vals = np.concatenate([r["scores"] for r in ind_results[score_name]])
        ood_bg   = np.concatenate([r["scores"][r["labels"] == 0] for r in ood_results[score_name]])
        ood_obj  = np.concatenate([
            r["scores"][r["labels"] == 1] for r in ood_results[score_name]
            if (r["labels"] == 1).sum() > 0
        ])
        summary_rows.append({
            "Score":           score_name.upper(),
            "Mean (InD CS)":   round(float(ind_vals.mean()), 4),
            "Top5% (InD CS)":  round(float(np.percentile(ind_vals, 95)), 4),
            "Mean (LAF bg)":   round(float(ood_bg.mean()), 4),
            "Mean (LAF OoD)":  round(float(ood_obj.mean()), 4),
            "Top5% (LAF OoD)": round(float(np.percentile(ood_obj, 95)), 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    with open(os.path.join(output_dir, "summary_table.tex"), "w") as f:
        f.write("% Auto-generated summary table\n\n")
        f.write(df_summary.to_latex(index=False, float_format="%.4f"))

    print(f"[Results] Saved summary tables to {output_dir}/")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Direct execution fallback
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    config_path = PROJECT_ROOT / "configs" / "eval_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    run(config)
