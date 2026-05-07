"""Build and store domain centroids from KB chunk embeddings.

Output: data/processed/domain_centroids.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_json, ensure_dir
from src.utils.logging import get_logger
from src.utils.device import get_device

logger = get_logger(__name__)


def load_encoder(model_name: str, device: torch.device):
    """Load a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading encoder: {model_name}")
    model = SentenceTransformer(model_name, device=str(device))
    return model


def compute_domain_centroids(
    kb_chunks: List[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device = None,
    batch_size: int = 64,
) -> Dict[str, dict]:
    """Encode all chunks and compute per-domain centroid embeddings."""
    if device is None:
        device = get_device("auto")

    model = load_encoder(model_name, device)

    # Group chunks by domain
    domain_chunks: Dict[str, List[dict]] = defaultdict(list)
    for ch in kb_chunks:
        domain_chunks[ch.get("domain", "unknown")].append(ch)

    centroids = {}
    for domain, chunks in domain_chunks.items():
        texts = [ch["text"] for ch in chunks]
        logger.info(f"Encoding {len(texts)} chunks for domain '{domain}' ...")

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        centroid = embeddings.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

        centroids[domain] = {
            "domain":    domain,
            "centroid":  centroid_norm.tolist(),
            "num_chunks": len(chunks),
        }
        logger.info(f"Domain '{domain}': {len(chunks)} chunks, centroid shape={centroid_norm.shape}")

    return centroids


def main(args):
    ensure_dir(os.path.dirname(args.out_path))
    kb_path = os.path.join(args.data_dir, "kb_chunks.jsonl")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"KB chunks not found at {kb_path}.")
    kb_chunks = read_jsonl(kb_path)
    if args.max_chunks:
        kb_chunks = kb_chunks[:args.max_chunks]

    # Load domain keywords to embed in centroid metadata
    kw_path = os.path.join(args.data_dir, "domain_keywords.json")
    domain_keywords = {}
    if os.path.exists(kw_path):
        import json
        with open(kw_path) as f:
            domain_keywords = json.load(f)

    device = get_device(args.device)
    centroids = compute_domain_centroids(
        kb_chunks,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
    )

    # Attach top keywords
    for domain in centroids:
        centroids[domain]["top_keywords"] = domain_keywords.get(domain, [])[:20]

    write_json(centroids, args.out_path)
    logger.info(f"Domain centroids saved to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build domain centroids from KB chunks")
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--out_path",   default="data/processed/domain_centroids.json")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_chunks", type=int, default=None)
    args = parser.parse_args()
    main(args)
