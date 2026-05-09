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


def normalize_domain_name(domain: str) -> str:
    """Dataset-specific canonicalization for the four support domains.
    
    student_aid/student aid/federal_student_aid -> studentaid
    social_security -> ssa
    veterans_affairs -> va
    motor_vehicle/department_of_motor_vehicles -> dmv
    """
    d = (domain or "").lower().strip().replace(" ", "_").replace("-", "_")
    
    if d in {"student_aid", "student_aid", "federal_student_aid", "studentaid"}:
        return "studentaid"
    if d in {"social_security", "ssa"}:
        return "ssa"
    if d in {"veterans_affairs", "va"}:
        return "va"
    if d in {"motor_vehicle", "department_of_motor_vehicles", "dmv"}:
        return "dmv"
    
    return d


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
    
    # Save raw embeddings for 'no-index' baseline
    embs_path = os.path.join(index_dir, "kb_embs.npy")
    np.save(embs_path, all_embs.astype(np.float32))

    logger.info(f"FAISS index saved: {index_path}")
    logger.info(f"Raw embeddings saved: {embs_path}")
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
    parser.add_argument("--mode",       choices=["global", "raw", "domain"], default="global",
                        help="Building mode: global (all chunks), raw (all chunks with MiniLM), domain (split by domain)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--device",     default="auto")
    
    # New arguments to match high-level script
    parser.add_argument("--raw_index_dir",      default="data/indexes_raw")
    parser.add_argument("--domain_indexes_dir", default="data/indexes_by_domain")
    parser.add_argument("--raw_model",          default="sentence-transformers/all-MiniLM-L6-v2")
    
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    device = get_device(args.device)
    
    # Determine model and target dir based on mode
    model_name = args.model_path
    target_dir = args.index_dir
    
    if args.mode == "raw":
        model_name = args.raw_model
        target_dir = args.raw_index_dir
        logger.info(f"Building RAW baseline index using {model_name}")
    elif args.mode == "domain":
        target_dir = args.domain_indexes_dir
        logger.info(f"Building DOMAIN-specific indexes in {target_dir}")
    else:
        logger.info(f"Building GLOBAL index in {target_dir} using {model_name}")

    encoder = SentenceTransformer(model_name, device=str(device))
    kb_chunks = read_jsonl(args.kb_path)
    
    if args.mode == "domain":
        # Group chunks by domain
        domain_groups = {}
        for ch in kb_chunks:
            dom = ch.get("domain", "unknown")
            if dom not in domain_groups:
                domain_groups[dom] = []
            domain_groups[dom].append(ch)
            
        for dom, chunks in domain_groups.items():
            dom_dir = os.path.join(target_dir, dom)
            logger.info(f"--- Building index for domain: {dom} ({len(chunks)} chunks) ---")
            build_faiss_index(chunks, encoder, dom_dir, args.batch_size, args.max_chunks)
    else:
        build_faiss_index(kb_chunks, encoder, target_dir, args.batch_size, args.max_chunks)
