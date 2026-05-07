"""scripts/prepare_data.py — Full data preparation pipeline.

Runs:
1. Download IBM/multidoc2dial
2. Build KB chunks
3. Preprocess dialogues
4. Build triage training data (ANSWER + synthetic REJECT/TICKET)
5. Build preference pairs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare all data")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    raw_dir  = cfg.get("data_dir", "data/processed").replace("processed", "raw")
    out_dir  = cfg.get("data_dir", "data/processed")
    max_kb   = cfg.get("max_kb_chunks")
    max_trn  = cfg.get("max_train_samples")
    max_eval = cfg.get("max_eval_samples")

    # 1. Download dataset
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading IBM/multidoc2dial")
    logger.info("=" * 60)
    from src.data.load_multidoc2dial import main as download_main
    import types
    dl_args = types.SimpleNamespace(raw_dir=raw_dir)
    try:
        download_main(dl_args)
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        logger.warning("If offline, place raw JSONL files in data/raw/ manually.")
        raise

    # 2. Build KB chunks
    logger.info("=" * 60)
    logger.info("STEP 2: Building KB chunks")
    logger.info("=" * 60)
    from src.data.build_kb import build_kb_chunks
    build_kb_chunks(raw_dir, out_dir, max_chunks=max_kb)

    # 3. Preprocess dialogues
    logger.info("=" * 60)
    logger.info("STEP 3: Preprocessing dialogues")
    logger.info("=" * 60)
    from src.data.preprocess import main as preprocess_main
    pp_args = types.SimpleNamespace(
        raw_dir=raw_dir,
        out_dir=out_dir,
        max_train_samples=max_trn,
        max_eval_samples=max_eval,
    )
    preprocess_main(pp_args)

    # 4. Build domain keywords
    logger.info("=" * 60)
    logger.info("STEP 4: Building domain keywords")
    logger.info("=" * 60)
    from src.routing.domain_keywords import main as kw_main
    kw_args = types.SimpleNamespace(
        data_dir=out_dir,
        out_path=os.path.join(out_dir, "domain_keywords.json"),
        top_n=50,
        max_chunks=max_kb,
    )
    kw_main(kw_args)

    # 5. Make triage data (REJECT + TICKET)
    logger.info("=" * 60)
    logger.info("STEP 5: Creating triage training data")
    logger.info("=" * 60)
    from src.data.make_negatives import main as neg_main
    neg_args = types.SimpleNamespace(
        out_dir=out_dir,
        max_samples=max_trn,
    )
    neg_main(neg_args)

    # 6. Make preference pairs
    logger.info("=" * 60)
    logger.info("STEP 6: Creating preference pairs")
    logger.info("=" * 60)
    from src.data.make_preference_pairs import main as pref_main
    pref_args = types.SimpleNamespace(
        out_dir=out_dir,
        max_samples=max_trn,
    )
    pref_main(pref_args)

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
