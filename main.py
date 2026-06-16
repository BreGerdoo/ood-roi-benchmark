"""
main.py
-------
Entry point for the Chapter-2 baseline evaluation (SegFormer MSP/Entropy/Energy).
Run from anywhere:
    python main.py
"""

import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from evaluation.run_evaluation import run

if __name__ == "__main__":
    config_path = ROOT / "configs" / "eval_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    run(config)
