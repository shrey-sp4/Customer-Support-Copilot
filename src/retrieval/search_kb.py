"""KB search: encode query and search FAISS index with optional domain filtering."""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import faiss

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KBSearcher:
    """Retrieve KB passages using FAISS + optional domain filter."""

    def __init__(
        self,
        index: faiss.Index,
        chunk_ids: List[str],
        chunk_by_id: Dict[str, dict],
        encoder,
    ):
        self.index       = index
        self.chunk_ids   = chunk_ids
        self.chunk_by_id = chunk_by_id
        self.encoder     = encoder

    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str | List[str]] = None,
    ) -> List[dict]:
        """Search KB and return top-k passages (optionally filtered by domain(s))."""
        q_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Retrieve more than top_k to allow domain filtering
        search_k = min(top_k * 10, self.index.ntotal) if domain else top_k
        scores, indices = self.index.search(q_emb.astype(np.float32), search_k)

        results = []
        domains_to_filter = [domain] if isinstance(domain, str) else domain
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            cid   = self.chunk_ids[idx]
            chunk = self.chunk_by_id.get(cid)
            if chunk is None:
                continue
            
            if domains_to_filter and chunk.get("domain") not in domains_to_filter:
                continue
                
            results.append({
                "doc_id":     chunk.get("doc_id",     ""),
                "chunk_id":   chunk.get("chunk_id",   ""),
                "section_id": chunk.get("section_id", ""),
                "span_start": chunk.get("span_start", 0),
                "span_end":   chunk.get("span_end",   0),
                "domain":     chunk.get("domain",     ""),
                "text":       chunk.get("text",       ""),
                "score":      float(score),
            })
            if len(results) >= top_k:
                break

        return results

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Return normalized query embedding."""
        emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb[0]

    def get_nearest_chunk_sim(self, query: str) -> float:
        """Return cosine similarity to the nearest KB chunk."""
        results = self.search(query, top_k=1)
        return results[0]["score"] if results else 0.0

    def get_retrieval_score_gap(self, query: str, top_k: int = 5) -> float:
        """Return gap between top-1 and top-k score."""
        results = self.search(query, top_k=top_k)
        if len(results) < 2:
            return 0.0
        return results[0]["score"] - results[-1]["score"]


def load_searcher(
    index_dir: str,
    kb_path: str,
    encoder,
) -> KBSearcher:
    """Load FAISS index, chunk map, and return a KBSearcher."""
    from src.retrieval.build_faiss import load_faiss_index
    from src.utils.io import read_jsonl

    index, chunk_ids = load_faiss_index(index_dir)
    kb_chunks   = read_jsonl(kb_path)
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    return KBSearcher(index, chunk_ids, chunk_by_id, encoder)
