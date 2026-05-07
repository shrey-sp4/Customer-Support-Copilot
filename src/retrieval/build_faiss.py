"""Build FAISS index over KB chunk embeddings.

Saves: data/indexes/kb_faiss.index  + data/indexes/kb_chunk_ids.json
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np
import faiss
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import get_logger
from src.utils.device import get_device

logger = get_logger(__name__)


def build_faiss_index(
    kb_chunks: List[dict],
    encoder,
    index_dir: str = "data/indexes",
    batch_size: int = 64,
    max_chunks: int = None,
) -> faiss.IndexFlatIP:
    """Encode KB chunks and build an inner-product (cosine after L2 norm) FAISS index."""
    ensure_dir(index_dir)

    if max_chunks:
        kb_chunks = kb_chunks[:max_chunks]

    logger.info(f"Encoding {len(kb_chunks)} KB chunks for FAISS index …")
    texts = [ch["text"] for ch in kb_chunks]
    chunk_ids = [ch["chunk_id"] for ch in kb_chunks]

    # Encode in batches
    all_embs = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 norm -> inner product = cosine
    )

    dim = all_embs.shape[1]
    logger.info(f"Embedding dimension: {dim}, Building IndexFlatIP ...")
    index = faiss.IndexFlatIP(dim)
    index.add(all_embs.astype(np.float32))

    index_path  = os.path.join(index_dir, "kb_faiss.index")
    ids_path    = os.path.join(index_dir, "kb_chunk_ids.json")

    faiss.write_index(index, index_path)
    with open(ids_path, "w") as f:
        json.dump(chunk_ids, f)

    logger.info(f"FAISS index saved: {index_path} ({index.ntotal} vectors)")
    logger.info(f"Chunk ID map saved: {ids_path}")
    return index, chunk_ids


def load_faiss_index(index_dir: str = "data/indexes"):
    """Load FAISS index and chunk ID list."""
    index_path = os.path.join(index_dir, "kb_faiss.index")
    ids_path   = os.path.join(index_dir, "kb_chunk_ids.json")
    index = faiss.read_index(index_path)
    with open(ids_path) as f:
        chunk_ids = json.load(f)
    logger.info(f"FAISS index loaded: {index.ntotal} vectors, {len(chunk_ids)} chunk IDs")
    return index, chunk_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from KB chunks")
    parser.add_argument("--kb_path",    default="data/processed/kb_chunks.jsonl")
    parser.add_argument("--index_dir",  default="data/indexes")
    parser.add_argument("--model_path", default="outputs/retriever",
                        help="Fine-tuned retriever path or HF model ID")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    device = get_device(args.device)
    if os.path.isdir(args.model_path):
        encoder = SentenceTransformer(args.model_path, device=str(device))
        logger.info(f"Loaded fine-tuned retriever from {args.model_path}")
    else:
        encoder = SentenceTransformer(args.model_path, device=str(device))

    kb_chunks = read_jsonl(args.kb_path)
    build_faiss_index(kb_chunks, encoder, args.index_dir, args.batch_size, args.max_chunks)
