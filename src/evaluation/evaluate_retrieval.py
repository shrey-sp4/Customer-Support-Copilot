"""Evaluate retrieval performance on the eval set."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_json
from src.evaluation.metrics import compute_retrieval_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_retrieval(searcher, eval_set, top_k: int = 5, domain: str = None) -> dict:
    """Run retrieval on all eval queries and compute metrics."""
    results = []
    for sample in eval_set:
        query    = sample["query"]
        gold_cid = sample.get("gold_chunk_id", "")
        passages = searcher.search(query, top_k=top_k, domain=domain)
        ret_ids  = [p["chunk_id"] for p in passages]
        results.append({
            "query_id":             sample["query_id"],
            "gold_chunk_id":        gold_cid,
            "retrieved_chunk_ids":  ret_ids,
        })
    metrics = compute_retrieval_metrics(results, top_k=top_k)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--eval_path",  default="data/processed/eval_set.jsonl")
    parser.add_argument("--index_dir",  default="data/indexes")
    parser.add_argument("--kb_path",    default="data/processed/kb_chunks.jsonl")
    parser.add_argument("--model_path", default="outputs/retriever")
    parser.add_argument("--out_path",   default="outputs/reports/retrieval_metrics.json")
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from src.utils.device import get_device
    from src.retrieval.search_kb import load_searcher

    device  = get_device(args.device)
    encoder = SentenceTransformer(
        args.model_path if os.path.isdir(args.model_path) else "sentence-transformers/all-MiniLM-L6-v2",
        device=str(device),
    )
    searcher = load_searcher(args.index_dir, args.kb_path, encoder)
    eval_set = read_jsonl(args.eval_path)
    metrics  = evaluate_retrieval(searcher, eval_set, top_k=args.top_k)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    write_json(metrics, args.out_path)
