"""Triage feature extraction.

Builds the input text feature string for the triage model and extracts
numeric features (centroid sim, chunk sim, etc.) that are formatted into
the input text for easy integration with any classification model.
"""
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

TRIAGE_LABELS = ["ANSWER", "TICKET", "REJECT"]
LABEL2ID = {l: i for i, l in enumerate(TRIAGE_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(TRIAGE_LABELS)}


def build_triage_input(
    query: str,
    keyword_gate: str = "pass",
    centroid_domain: str = "unknown",
    centroid_sim_top1: float = 0.0,
    centroid_margin: float = 0.0,
    nearest_chunk_sim: float = 0.0,
    retrieval_score_gap: float = 0.0,
    history: str = "",
) -> str:
    """
    Build the flat text input for the triage model, incorporating
    all numeric and categorical features as formatted text fields.
    Example output:
        query: Can I renew my benefits online?
        history:
        keyword_gate: pass
        nearest_centroid_domain: ssa
        centroid_sim_top1: 0.78
        centroid_margin: 0.22
        nearest_chunk_sim: 0.74
        retrieval_score_gap: 0.14
    """
    lines = [
        f"query: {query.strip()}",
        f"history: {history.strip()[:200]}",
        f"keyword_gate: {keyword_gate}",
        f"nearest_centroid_domain: {centroid_domain}",
        f"centroid_sim_top1: {centroid_sim_top1:.4f}",
        f"centroid_margin: {centroid_margin:.4f}",
        f"nearest_chunk_sim: {nearest_chunk_sim:.4f}",
        f"retrieval_score_gap: {retrieval_score_gap:.4f}",
    ]
    return "\n".join(lines)


def extract_features_from_record(record: dict) -> dict:
    """Extract triage features from a pre-computed record (e.g. from triage_train.jsonl)."""
    return {
        "input_text":          build_triage_input(
            query               = record.get("query", ""),
            keyword_gate        = record.get("keyword_gate", "pass"),
            centroid_domain     = record.get("centroid_domain", "unknown"),
            centroid_sim_top1   = float(record.get("centroid_sim_top1", 0.0)),
            centroid_margin     = float(record.get("centroid_margin", 0.0)),
            nearest_chunk_sim   = float(record.get("nearest_chunk_sim", 0.0)),
            retrieval_score_gap = float(record.get("retrieval_score_gap", 0.0)),
            history             = record.get("history", ""),
        ),
        "label": LABEL2ID.get(record.get("gold_triage", "ANSWER"), 0),
        "gold_triage": record.get("gold_triage", "ANSWER"),
    }
