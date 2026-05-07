"""Load IBM/multidoc2dial dataset from Hugging Face and save raw splits."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_dataset
from src.utils.io import write_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_multidoc2dial(cache_dir: str = "data/raw") -> dict:
    """Download and return multidoc2dial splits as HF datasets."""
    ensure_dir(cache_dir)
    logger.info("Downloading IBM/multidoc2dial from Hugging Face Hub …")
    # multidoc2dial has two configs: multidoc2dial_dial and multidoc2dial_doc
    dial_ds = load_dataset("IBM/multidoc2dial", "multidoc2dial", trust_remote_code=True)
    doc_ds  = load_dataset("IBM/multidoc2dial", "document_domain", trust_remote_code=True)
    logger.info(f"Dialogue splits: {list(dial_ds.keys())}")
    logger.info(f"Document splits: {list(doc_ds.keys())}")
    return {"dial": dial_ds, "doc": doc_ds}


def save_raw_documents(doc_ds, out_dir: str = "data/raw") -> None:
    """Flatten document dataset and save as JSONL."""
    ensure_dir(out_dir)
    records = []
    for split_name in doc_ds:
        for item in doc_ds[split_name]:
            records.append(dict(item))
    write_jsonl(records, os.path.join(out_dir, "documents.jsonl"))
    logger.info(f"Saved {len(records)} raw document records.")


def save_raw_dialogues(dial_ds, out_dir: str = "data/raw") -> None:
    """Flatten dialogue dataset and save as JSONL."""
    ensure_dir(out_dir)
    for split_name in dial_ds:
        records = []
        for item in dial_ds[split_name]:
            records.append(dict(item))
        write_jsonl(records, os.path.join(out_dir, f"dialogues_{split_name}.jsonl"))
        logger.info(f"Split '{split_name}': {len(records)} dialogues saved.")


def main(args):
    datasets = load_multidoc2dial(cache_dir=args.raw_dir)
    save_raw_documents(datasets["doc"], out_dir=args.raw_dir)
    save_raw_dialogues(datasets["dial"], out_dir=args.raw_dir)
    logger.info("Raw data download complete.")
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IBM/multidoc2dial dataset")
    parser.add_argument("--raw_dir", default="data/raw", help="Where to save raw data")
    args = parser.parse_args()
    main(args)
