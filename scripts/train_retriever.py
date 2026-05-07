"""scripts/train_retriever.py — Train the retriever model."""
import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train retriever")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    from src.retrieval.train_retriever import train_retriever

    train_path = os.path.join(cfg.get("data_dir", "data/processed"), "retriever_train.jsonl")
    output_dir = os.path.join(cfg.get("output_dir", "outputs"), "retriever")

    train_retriever(
        train_path  = train_path,
        output_dir  = output_dir,
        model_name  = cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2"),
        epochs      = cfg.get("retriever_epochs", 1),
        batch_size  = cfg.get("batch_size", 8),
        max_samples = cfg.get("max_train_samples"),
        fp16        = cfg.get("fp16", True),
        seed        = cfg.get("seed", 42),
    )
    logger.info("Retriever training complete.")


if __name__ == "__main__":
    main()
