"""Domain router: combines lexical keyword gate + centroid-based routing.

Provides:
  - LexicalGate: fast keyword-based out-of-domain detection
  - DomainRouter: centroid-similarity based domain routing
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lexical gate
# ---------------------------------------------------------------------------

# Patterns that strongly indicate out-of-domain queries
OOD_PATTERNS = [
    r"\b(recipe|cook|bake|chef|food|meal|dish|restaurant)\b",
    r"\b(ipl|cricket|football|soccer|nba|nfl|sports|match|score|game)\b",
    r"\b(netflix|amazon|ebay|paypal|youtube|instagram|facebook|twitter)\b",
    r"\b(movie|film|series|show|streaming)\b",
    r"\b(joke|meme|riddle|funny|humor)\b",
    r"\b(code|programming|python|java|algorithm|software)\b",
    r"\b(bitcoin|crypto|ethereum|stock|forex|trading)\b",
    r"\b(weather|forecast|temperature|rain|sun)\b",
    r"\b(paris|france|london|capital|geography)\b",
    r"\b(black hole|quantum|physics|astronomy|space)\b",
]

OOD_COMPILED = [re.compile(p, re.IGNORECASE) for p in OOD_PATTERNS]


class LexicalGate:
    """Phase-1 fast out-of-domain detection using keyword patterns."""

    def __init__(self, domain_keywords: Dict[str, List[str]]):
        self.domain_keywords = domain_keywords
        # Build per-domain pattern sets for support domain matching
        self._support_tokens: set = set()
        for kws in domain_keywords.values():
            self._support_tokens.update(k.lower() for k in kws[:30])

    def check(self, query: str) -> Tuple[str, List[str]]:
        """Returns ('pass'|'reject', matched_ood_patterns)."""
        matched = []
        for pat in OOD_COMPILED:
            m = pat.search(query)
            if m:
                matched.append(m.group(0).lower())
        if matched:
            return "reject", matched
        return "pass", []

    def get_matched_support_keywords(self, query: str) -> List[str]:
        """Return support domain keywords found in query."""
        tokens = re.findall(r"[a-z]+", query.lower())
        return [t for t in tokens if t in self._support_tokens]


# ---------------------------------------------------------------------------
# Domain router
# ---------------------------------------------------------------------------

class DomainRouter:
    """Routes a query to top-k domains using centroid cosine similarity."""

    def __init__(self, centroids: Dict[str, dict], domain_keywords: Dict[str, List[str]]):
        self.domain_keywords = domain_keywords
        self.domains: List[str] = []
        self.centroid_matrix: Optional[np.ndarray] = None
        self._build_matrix(centroids)
        self.lexical_gate = LexicalGate(domain_keywords)

    def _build_matrix(self, centroids: Dict[str, dict]):
        """Stack centroids into a matrix for fast cosine similarity."""
        self.domains = []
        vecs = []
        for domain, info in centroids.items():
            vec = np.array(info["centroid"], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            self.domains.append(domain)
            vecs.append(vec)
        if vecs:
            self.centroid_matrix = np.stack(vecs, axis=0)  # (n_domains, dim)
            logger.info(f"DomainRouter: {len(self.domains)} domains loaded.")
        else:
            logger.warning("DomainRouter: no centroids found!")

    def route(
        self,
        query_embedding: np.ndarray,
        top_k: int = 2,
    ) -> List[dict]:
        """Return top-k domains with centroid similarities."""
        if self.centroid_matrix is None or len(self.domains) == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        sims = self.centroid_matrix @ q  # (n_domains,)
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            domain = self.domains[idx]
            sim = float(sims[idx])
            dist = float(1.0 - sim)
            matched_kws = self._get_matched_keywords(domain, query_embedding)
            results.append({
                "domain":               domain,
                "centroid_similarity":  sim,
                "centroid_distance":    dist,
                "matched_keywords":     matched_kws,
            })
        return results

    def _get_matched_keywords(self, domain: str, query_embedding: np.ndarray) -> List[str]:
        """Placeholder — returns empty; real matching via LexicalGate on raw query."""
        return []

    def full_route(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 2,
        tau_domain: float = 0.35,
    ) -> dict:
        """Full routing pipeline: lexical gate + centroid similarity."""
        gate_result, ood_matches = self.lexical_gate.check(query)
        support_kws = self.lexical_gate.get_matched_support_keywords(query)

        domain_results = self.route(query_embedding, top_k=top_k)

        if domain_results:
            top_sim = domain_results[0]["centroid_similarity"]
            margin  = (domain_results[0]["centroid_similarity"] - domain_results[1]["centroid_similarity"]
                       if len(domain_results) > 1 else top_sim)
        else:
            top_sim = 0.0
            margin  = 0.0

        # Domain routing decision
        if gate_result == "reject" and top_sim < tau_domain:
            decision = "reject"
        elif gate_result == "reject" and top_sim >= tau_domain:
            # Hard reject (keyword strongly OOD)
            decision = "reject"
        else:
            decision = "route"

        # Attach support keywords to top domain result
        if domain_results:
            domain_results[0]["matched_keywords"] = support_kws

        return {
            "gate_result":      gate_result,
            "ood_matches":      ood_matches,
            "support_keywords": support_kws,
            "domain_results":   domain_results,
            "top_domain":       domain_results[0]["domain"] if domain_results else None,
            "top_centroid_sim": top_sim,
            "centroid_margin":  margin,
            "route_confidence": top_sim,
            "decision":         decision,
        }


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------

def load_router(
    centroids_path: str,
    keywords_path: str,
) -> DomainRouter:
    """Load DomainRouter from saved centroid and keyword files."""
    with open(centroids_path, "r") as f:
        centroids = json.load(f)
    with open(keywords_path, "r") as f:
        domain_keywords = json.load(f)
    return DomainRouter(centroids, domain_keywords)
