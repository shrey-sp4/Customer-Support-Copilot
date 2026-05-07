"""Preference scorer inference — selects the best candidate answer."""
import os
import re
from typing import List, Optional

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PreferenceScorer:
    """Scores candidate answers and selects the best one using a trained ranker."""

    def __init__(self, model_path: str, device: torch.device = None, max_length: int = 384):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device is None:
            from src.utils.device import get_device
            device = get_device("auto")
        self.device     = device
        self.max_length = max_length

        logger.info(f"Loading preference scorer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score_candidates(self, query: str, candidates: List[str]) -> List[float]:
        """Return a preference score for each candidate."""
        scores = []
        for cand in candidates:
            enc = self.tokenizer(
                query,
                cand,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            logit = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0, 0]
            scores.append(float(logit.cpu()))
        return scores

    def select_best(self, query: str, candidates: List[str], passages: List[dict] = None) -> str:
        """Select the best candidate, with a citation-presence bonus."""
        if len(candidates) == 1:
            return candidates[0]

        scores = self.score_candidates(query, candidates)

        # Apply citation presence bonus
        for i, cand in enumerate(candidates):
            if re.search(r"\[doc_id=", cand):
                scores[i] += 1.0  # Bonus for citation

        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return candidates[best_idx]


def load_preference_scorer(model_path: str, device=None) -> Optional[PreferenceScorer]:
    """Load preference scorer; returns None if model path does not exist."""
    if not os.path.isdir(model_path):
        logger.warning(f"Preference scorer not found at {model_path}. Skipping.")
        return None
    return PreferenceScorer(model_path, device=device)
