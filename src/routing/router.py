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

# Patterns are loaded from config/OOD_PATTERNS in LexicalGate

class LexicalGate:
    """Phase-1 fast out-of-domain detection using keyword patterns."""

    def __init__(self, domain_keywords: Dict[str, List[str]], ood_patterns: List[str] = None):
        self.domain_keywords = domain_keywords
        self.ood_compiled = [re.compile(p, re.IGNORECASE) for p in (ood_patterns or [])]
        self.support_patterns = {
            domain: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in kws]
            for domain, kws in domain_keywords.items()
        }

    def check(self, query: str) -> Tuple[str, List[str]]:
        """Returns ('pass'|'reject', matched_ood_patterns)."""
        matched = []
        for pat in self.ood_compiled:
            m = pat.search(query)
            if m:
                matched.append(m.group(0).lower())
        if matched:
            return "reject", matched
        return "pass", []

    def get_matched_support_keywords(self, query: str) -> Dict[str, List[str]]:
        """Return support domain keywords found in query, grouped by domain."""
        matched = defaultdict(list)
        for domain, patterns in self.support_patterns.items():
            for pat in patterns:
                m = pat.search(query)
                if m:
                    matched[domain].append(m.group(0).lower())
        return dict(matched)


# ---------------------------------------------------------------------------
# Domain router
# ---------------------------------------------------------------------------

class DomainRouter:
    """Routes a query using multi-signal voting (centroid, keywords, aliases)."""

    def __init__(self, centroids: Dict[str, dict], domain_keywords: Dict[str, List[str]], ood_patterns: List[str] = None):
        self.domain_keywords = domain_keywords
        self.domains: List[str] = []
        self.centroid_matrix = None
        self._build_matrix(centroids)
        self.lexical_gate = LexicalGate(domain_keywords, ood_patterns=ood_patterns)
        
        # Weights for multi-signal voting
        self.w_centroid = 0.40
        self.w_keyword  = 0.50
        self.w_alias    = 0.10

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
        """Compatibility wrapper for centroid-only routing."""
        if self.centroid_matrix is None or len(self.domains) == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        sims = self.centroid_matrix @ q
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            domain = self.domains[idx]
            sim = float(sims[idx])
            results.append({
                "domain": domain,
                "centroid_similarity": sim,
                "centroid_distance": float(1.0 - sim),
                "matched_keywords": [],
            })
        return results

    def _get_ngram_votes(self, query: str) -> Dict[str, float]:
        """Compute keyword voting scores per domain using unigrams and bigrams."""
        query = query.lower()
        tokens = re.findall(r"\b\w+\b", query)
        bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        
        votes = defaultdict(float)
        matched_kws = defaultdict(list)
        
        for domain, keywords in self.domain_keywords.items():
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in tokens or kw_lower in bigrams:
                    # Strong match if keyword is multiple words and found in query
                    weight = 1.5 if " " in kw_lower else 1.0
                    votes[domain] += weight
                    matched_kws[domain].append(kw_lower)
        
        # Normalize scores
        total = sum(votes.values())
        if total > 0:
            scores = {d: v / total for d, v in votes.items()}
        else:
            scores = {d: 0.0 for d in self.domain_keywords}
            
        return scores, dict(matched_kws)

    def full_route(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 2,
        tau_domain: float = 0.30,
    ) -> dict:
        """Full routing pipeline: lexical gate + multi-signal voting."""
        gate_result, ood_matches = self.lexical_gate.check(query)
        
        # 1. Centroid similarities
        if self.centroid_matrix is not None:
            q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            centroid_sims = self.centroid_matrix @ q
            centroid_scores = {self.domains[i]: float(centroid_sims[i]) for i in range(len(self.domains))}
        else:
            centroid_scores = {d: 0.0 for d in self.domain_keywords}

        # 2. Keyword voting scores
        kw_scores, matched_kws = self._get_ngram_votes(query)
        
        # 3. Multi-signal aggregation
        final_scores = {}
        for d in self.domain_keywords:
            final_scores[d] = (
                self.w_centroid * centroid_scores.get(d, 0.0) +
                self.w_keyword  * kw_scores.get(d, 0.0)
            )

        # 4. Strong keyword overrides
        # If a domain has a high-quality keyword match, give it a massive bonus
        strong_bonus = 0.5
        for d, kws in matched_kws.items():
            if kws:
                final_scores[d] += strong_bonus

        # Sort results
        sorted_domains = sorted(final_scores.items(), key=lambda x: -x[1])
        top_results = []
        for d, score in sorted_domains[:top_k]:
            top_results.append({
                "domain": d,
                "score": score,
                "centroid_similarity": centroid_scores.get(d, 0.0),
                "keyword_score": kw_scores.get(d, 0.0),
                "matched_keywords": matched_kws.get(d, []),
            })

        top_sim = top_results[0]["centroid_similarity"] if top_results else 0.0
        top_score = top_results[0]["score"] if top_results else 0.0
        
        # All matched support keywords across all domains
        all_support_kws = []
        for kws in matched_kws.values():
            all_support_kws.extend(kws)
        all_support_kws = list(set(all_support_kws))

        # Decision logic
        if gate_result == "reject":
            decision = "reject"
        elif not all_support_kws and top_sim < tau_domain:
            decision = "reject"
        else:
            decision = "route"

        return {
            "gate_result":      gate_result,
            "ood_matches":      ood_matches,
            "support_keywords": all_support_kws,
            "domain_results":   top_results,
            "top_domain":       top_results[0]["domain"] if top_results else None,
            "top_centroid_sim": top_sim,
            "top_score":        top_score,
            "decision":         decision,
            "matched_kws_by_domain": matched_kws,
        }

from collections import defaultdict

def load_router(
    centroids_path: str,
    keywords_path: str,
    ood_patterns: List[str] = None,
) -> DomainRouter:
    """Load DomainRouter from saved centroid and keyword files."""
    with open(centroids_path, "r") as f:
        centroids = json.load(f)
    with open(keywords_path, "r") as f:
        domain_keywords = json.load(f)
    return DomainRouter(centroids, domain_keywords, ood_patterns=ood_patterns)
