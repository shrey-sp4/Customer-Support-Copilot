"""scripts/build_index.py — Build FAISS index for KB and compute domain centroids."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and Centroids")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_dir   = cfg.get("data_dir", "data/processed")
    index_dir  = cfg.get("index_dir", "data/indexes")
    model_path = os.path.join(cfg.get("output_dir", "outputs"), "retriever")
    # Fallback to base model if fine-tuned doesn't exist
    if not os.path.isdir(model_path):
        model_path = cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2")

    max_kb     = cfg.get("max_kb_chunks")
    batch_size = cfg.get("batch_size", 8) * 8  # Encoding is fast, use larger batch
    device_str = getattr(cfg, "device", "auto")

    from sentence_transformers import SentenceTransformer
    from src.utils.device import get_device
    device = get_device(device_str)
    encoder = SentenceTransformer(model_path, device=str(device))

    # 1. Build Centroids
    logger.info("=" * 60)
    logger.info("STEP 1: Building Domain Centroids")
    logger.info("=" * 60)
    from src.routing.centroids import compute_domain_centroids
    from src.utils.io import read_jsonl, write_json
    
    kb_path = os.path.join(data_dir, "kb_chunks.jsonl")
    kb_chunks = read_jsonl(kb_path)
    if max_kb:
        kb_chunks = kb_chunks[:max_kb]

    centroids = compute_domain_centroids(
        kb_chunks,
        model_name=model_path,
        device=device,
        batch_size=batch_size,
    )
    
    # Attach keywords
    kw_path = os.path.join(data_dir, "domain_keywords.json")
    if os.path.exists(kw_path):
        import json
        with open(kw_path) as f:
            kws = json.load(f)
        for d in centroids:
            centroids[d]["top_keywords"] = kws.get(d, [])[:20]
            
    write_json(centroids, os.path.join(data_dir, "domain_centroids.json"))

    # 2. Build FAISS
    logger.info("=" * 60)
    logger.info("STEP 2: Building FAISS Index")
    logger.info("=" * 60)
    from src.retrieval.build_faiss import build_faiss_index
    build_faiss_index(kb_chunks, encoder, index_dir, batch_size)

    logger.info("Index building complete.")


if __name__ == "__main__":
    main()
