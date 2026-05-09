"""Cross-encoder reranker inference."""
import os
from typing import List, Optional

import torch
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker that scores (query, passage) pairs."""

    def __init__(self, model_path: str, device: torch.device = None, max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device is None:
            from src.utils.device import get_device
            device = get_device("auto")
        self.device     = device
        self.max_length = max_length

        logger.info(f"Loading reranker from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
        except Exception as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower() or "paging file" in str(e).lower():
                logger.warning(f"Failed to load reranker on {device}. Falling back to CPU...")
                self.device = torch.device("cpu")
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model.to(self.device)
            else:
                raise e
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, passages: List[str], batch_size: int = 8) -> List[float]:
        """Return a relevance score for each passage."""
        scores = []
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            enc = self.tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            scores.extend(logits.squeeze(-1).cpu().tolist())
        return scores

    def rerank(self, query: str, passages: List[dict], top_k: int = 5) -> List[dict]:
        """Rerank a list of passage dicts (each must have 'text') and return top_k."""
        texts  = [p["text"] for p in passages]
        scores = self.score(query, texts)
        # Attach reranker score
        for p, s in zip(passages, scores):
            p["reranker_score"] = s
        ranked = sorted(passages, key=lambda p: p["reranker_score"], reverse=True)
        return ranked[:top_k]


def load_reranker(model_path: str, device=None) -> Optional[Reranker]:
    """Load reranker; returns None if model path does not exist yet."""
    if not os.path.isdir(model_path):
        logger.warning(f"Reranker not found at {model_path}. Using retrieval scores.")
        return None
    return Reranker(model_path, device=device)
