"""KB search: encode query and search FAISS index with optional domain filtering."""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import faiss

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KBSearcher:
    """Retrieve KB passages using FAISS (global or domain-specific)."""

    def __init__(
        self,
        index_dir: str,
        kb_path: str,
        encoder,
        domain_indexes_dir: Optional[str] = None,
        global_domain_filter_multiplier: int = 10,
    ):
        from src.retrieval.build_faiss import load_faiss_index, normalize_domain_name
        from src.utils.io import read_jsonl

        self.encoder = encoder
        self.kb_path = kb_path
        self.index_dir = index_dir
        self.domain_indexes_dir = domain_indexes_dir
        self.global_domain_filter_multiplier = global_domain_filter_multiplier
        self.normalize_domain_name = normalize_domain_name
        
        # Load global index
        self.index, self.chunk_ids = load_faiss_index(index_dir)
        
        # Load raw embeddings if available (for 'no-index' baseline)
        self.raw_embs = None
        embs_path = os.path.join(index_dir, "kb_embs.npy")
        if os.path.exists(embs_path):
            self.raw_embs = np.load(embs_path)
            logger.info(f"Raw embeddings loaded: {self.raw_embs.shape}")

        # Load KB chunks
        kb_chunks = read_jsonl(kb_path)
        self.chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
        
        # Cache for domain-specific indexes
        self._domain_indexes = {} # {domain: (index, chunk_ids)}

    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str | List[str]] = None,
        use_index: bool = True,
    ) -> List[dict]:
        """Search KB. 
        If use_index=False, performs a raw linear scan on the entire KB (no clusters).
        """
        q_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if not use_index and self.raw_embs is not None:
            return self._search_linear_scan(q_emb, top_k)
        elif domain and self.domain_indexes_dir:
            return self._search_domain_specific(q_emb, query, top_k, domain)
        else:
            return self._search_global(q_emb, top_k, domain)

    def _search_linear_scan(self, q_emb: np.ndarray, top_k: int) -> List[dict]:
        """Brute-force linear scan across the entire KB (Truly No-Index)."""
        # Compute dot product (embeddings are normalized, so dot product = cosine similarity)
        scores = np.dot(self.raw_embs, q_emb.T).flatten()
        
        # Get top-k indices
        indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in indices:
            chunk = self.chunk_by_id.get(self.chunk_ids[idx])
            if chunk:
                results.append(self._format_chunk(chunk, scores[idx]))
        return results

    def _search_global(self, q_emb: np.ndarray, top_k: int, domain_filter: Optional[str | List[str]]) -> List[dict]:
        # Retrieve more than top_k to allow domain filtering if domain_filter is metadata-only
        search_k = min(top_k * self.global_domain_filter_multiplier, self.index.ntotal) if domain_filter else top_k
        scores, indices = self.index.search(q_emb, search_k)
        
        domains_to_filter = [domain_filter] if isinstance(domain_filter, str) else domain_filter
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids): continue
            chunk = self.chunk_by_id.get(self.chunk_ids[idx])
            if chunk is None: continue
            if domains_to_filter and chunk.get("domain") not in domains_to_filter: continue
            
            results.append(self._format_chunk(chunk, score))
            if len(results) >= top_k: break
        return results

    def _search_domain_specific(self, q_emb: np.ndarray, query: str, top_k: int, domains: str | List[str]) -> List[dict]:
        domains = [domains] if isinstance(domains, str) else domains
        all_results = []
        
        for dom in domains:
            dom = self.normalize_domain_name(dom)
            idx, ids = self._get_domain_index(dom)
            if idx is None: continue
            
            scores, indices = idx.search(q_emb, min(top_k, idx.ntotal))
            for score, i in zip(scores[0], indices[0]):
                if i < 0 or i >= len(ids): continue
                chunk = self.chunk_by_id.get(ids[i])
                if chunk:
                    all_results.append(self._format_chunk(chunk, score))
        
        # Merge and sort if multiple domains
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def _get_domain_index(self, domain: str):
        domain = self.normalize_domain_name(domain)
        if domain in self._domain_indexes:
            return self._domain_indexes[domain]
        
        dom_path = os.path.join(self.domain_indexes_dir, domain)
        if not os.path.isdir(dom_path):
            logger.warning(f"Domain index for '{domain}' not found in {self.domain_indexes_dir}")
            return None, None
            
        from src.retrieval.build_faiss import load_faiss_index
        try:
            res = load_faiss_index(dom_path)
            self._domain_indexes[domain] = res
            return res
        except Exception as e:
            logger.error(f"Error loading domain index for {domain}: {e}")
            return None, None

    def _format_chunk(self, chunk: dict, score: float) -> dict:
        return {
            "doc_id":     chunk.get("doc_id",     ""),
            "chunk_id":   chunk.get("chunk_id",   ""),
            "section_id": chunk.get("section_id", ""),
            "span_start": chunk.get("span_start", 0),
            "span_end":   chunk.get("span_end",   0),
            "domain":     chunk.get("domain",     ""),
            "text":       chunk.get("text",       ""),
            "score":      float(score),
        }

    def get_query_embedding(self, query: str) -> np.ndarray:
        emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return emb[0]

    def get_nearest_chunk_sim(self, query: str) -> float:
        results = self.search(query, top_k=1)
        return results[0]["score"] if results else 0.0


def load_searcher(
    index_dir: str,
    kb_path: str,
    encoder,
    domain_indexes_dir: Optional[str] = None,
    global_domain_filter_multiplier: int = 10,
) -> KBSearcher:
    """Load FAISS index, chunk map, and return a KBSearcher."""
    return KBSearcher(index_dir, kb_path, encoder, domain_indexes_dir, global_domain_filter_multiplier)
