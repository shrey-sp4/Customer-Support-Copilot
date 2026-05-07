"""Evaluate triage / tool-policy model on triage_train.jsonl held-out portion."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_json
from src.evaluation.metrics import compute_triage_metrics
from src.triage.predict import load_predictor
from src.triage.features import ID2LABEL
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_triage(predictor, eval_set, cfg=None) -> dict:
    """Run triage prediction on all eval samples and compute metrics."""
    predictions, labels, logits_list = [], [], []
    tau_domain = getattr(cfg, "tau_domain", 0.35)
    tau_chunk  = getattr(cfg, "tau_chunk",  0.40)

    for sample in eval_set:
        result = predictor.predict(
            query               = sample["query"],
            keyword_gate        = sample.get("keyword_gate", "pass"),
            centroid_domain     = sample.get("centroid_domain", "unknown"),
            centroid_sim_top1   = float(sample.get("centroid_sim_top1", 0.0)),
            centroid_margin     = float(sample.get("centroid_margin", 0.0)),
            nearest_chunk_sim   = float(sample.get("nearest_chunk_sim", 0.0)),
            retrieval_score_gap = float(sample.get("retrieval_score_gap", 0.0)),
            history             = sample.get("history", ""),
            tau_domain          = tau_domain,
            tau_chunk           = tau_chunk,
        )
        predictions.append(result["decision"])
        labels.append(sample.get("gold_triage", "ANSWER"))
        logits_list.append(result.get("logits", [0.0, 0.0, 0.0]))

    metrics = compute_triage_metrics(
        predictions=predictions,
        labels=labels,
        logits_list=logits_list,
        mu_values=[0.10, 0.15, 0.20],
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate triage model")
    parser.add_argument("--eval_path",  default="data/processed/triage_train.jsonl")
    parser.add_argument("--model_path", default="outputs/triage")
    parser.add_argument("--out_path",   default="outputs/reports/triage_metrics.json")
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    predictor = load_predictor(args.model_path)
    if predictor is None:
        logger.error("No triage model found. Cannot evaluate.")
        sys.exit(1)

    from src.utils.io import read_jsonl
    eval_set = read_jsonl(args.eval_path)
    # Use last 10% as eval
    n_eval   = max(50, len(eval_set) // 10)
    eval_set = eval_set[:n_eval]

    metrics = evaluate_triage(predictor, eval_set)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    write_json(metrics, args.out_path)
