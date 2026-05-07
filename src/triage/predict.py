"""Triage model inference / prediction."""
import os
from typing import List, Optional

import torch
import numpy as np

from src.triage.features import build_triage_input, ID2LABEL, LABEL2ID
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TriagePredictor:
    """Predict ANSWER / TICKET / REJECT for a query given routing features."""

    def __init__(self, model_path: str, device: torch.device = None, max_length: int = 256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device is None:
            from src.utils.device import get_device
            device = get_device("auto")
        self.device     = device
        self.max_length = max_length

        logger.info(f"Loading triage model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        query: str,
        keyword_gate: str = "pass",
        centroid_domain: str = "unknown",
        centroid_sim_top1: float = 0.0,
        centroid_margin: float = 0.0,
        nearest_chunk_sim: float = 0.0,
        retrieval_score_gap: float = 0.0,
        history: str = "",
        tau_domain: float = 0.35,
        tau_chunk: float = 0.40,
    ) -> dict:
        """
        Predict triage decision. Returns decision, confidence, margin.
        If keyword_gate == 'reject' and sim signals are low, short-circuits to REJECT.
        """
        # Rule-based short-circuit for obvious rejections
        if keyword_gate == "reject" and nearest_chunk_sim < tau_chunk and centroid_sim_top1 < tau_domain:
            return {
                "decision":   "REJECT",
                "confidence": 1.0,
                "margin":     1.0,
                "logits":     [-10.0, -10.0, 10.0],
                "probs":      [0.0, 0.0, 1.0],
                "method":     "rule",
            }

        # Model-based prediction
        input_text = build_triage_input(
            query               = query,
            keyword_gate        = keyword_gate,
            centroid_domain     = centroid_domain,
            centroid_sim_top1   = centroid_sim_top1,
            centroid_margin     = centroid_margin,
            nearest_chunk_sim   = nearest_chunk_sim,
            retrieval_score_gap = retrieval_score_gap,
            history             = history,
        )

        enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        pred   = int(probs.argmax())
        probs_sorted = sorted(probs, reverse=True)
        margin = float(probs_sorted[0] - probs_sorted[1])

        return {
            "decision":   ID2LABEL[pred],
            "confidence": float(probs[pred]),
            "margin":     margin,
            "logits":     logits.cpu().tolist(),
            "probs":      probs.tolist(),
            "method":     "model",
        }


def load_predictor(model_path: str, device=None) -> Optional[TriagePredictor]:
    """Load triage predictor; returns None if model path does not exist."""
    if not os.path.isdir(model_path):
        logger.warning(f"Triage model not found at {model_path}. Falling back to rule-based triage.")
        return None
    return TriagePredictor(model_path, device=device)
