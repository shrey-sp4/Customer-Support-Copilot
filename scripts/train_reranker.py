"""scripts/train_reranker.py — Train the cross-encoder reranker."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train reranker")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    from src.reranking.train_reranker import train_reranker

    train_path = os.path.join(cfg.get("data_dir", "data/processed"), "reranker_train.jsonl")
    output_dir = os.path.join(cfg.get("output_dir", "outputs"), "reranker")

    train_reranker(
        train_path  = train_path,
        output_dir  = output_dir,
        model_name  = cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        epochs      = cfg.get("reranker_epochs", 1),
        batch_size  = cfg.get("reranker_batch_size", 4),
        max_samples = cfg.get("max_train_samples"),
        fp16        = cfg.get("fp16", True),
        seed        = cfg.get("seed", 42),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 4),
    )
    logger.info("Reranker training complete.")


if __name__ == "__main__":
    main()
