"""scripts/train_triage.py — Train the boundary-aware triage model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train triage model")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    from src.triage.train_triage import train_triage

    train_path = os.path.join(cfg.get("data_dir", "data/processed"), "triage_train.jsonl")
    output_dir = os.path.join(cfg.get("output_dir", "outputs"), "triage")

    train_triage(
        train_path      = train_path,
        output_dir      = output_dir,
        model_name      = cfg.get("triage_model", "distilbert-base-uncased"),
        epochs          = cfg.get("triage_epochs", 3),
        batch_size      = cfg.get("triage_batch_size", 8),
        max_samples     = cfg.get("max_train_samples"),
        fp16            = cfg.get("fp16", True),
        seed            = cfg.get("seed", 42),
        mu              = cfg.get("mu_boundary", 0.15),
        lambda_boundary = cfg.get("lambda_boundary", 0.6),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 4),
    )
    logger.info("Triage training complete.")


if __name__ == "__main__":
    main()
