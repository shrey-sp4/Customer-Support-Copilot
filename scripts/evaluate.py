"""scripts/evaluate.py — Run E2E evaluation of baseline and proposed systems."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


import traceback

def exception_handler(type, value, tb):
    logger.error("Uncaught exception:")
    logger.error("".join(traceback.format_exception(type, value, tb)))
    with open("critical_error.log", "w") as f:
        traceback.print_exception(type, value, tb, file=f)

sys.excepthook = exception_handler

def main():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End pipeline")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device_str = getattr(cfg, "device", "auto")
    from src.utils.device import get_device
    device = get_device(device_str)

    data_dir   = cfg.get("data_dir", "data/processed")
    index_dir  = cfg.get("index_dir", "data/indexes")
    output_dir = cfg.get("output_dir", "outputs")
    rep_dir    = os.path.join(output_dir, "reports")
    
    from src.utils.io import ensure_dir, read_jsonl, write_json
    ensure_dir(rep_dir)

    # 1. Load components
    from sentence_transformers import SentenceTransformer
    from src.retrieval.search_kb import load_searcher
    from src.routing.router import load_router
    from src.triage.predict import load_predictor
    from src.reranking.rerank import load_reranker
    from src.generation.generate import load_generator
    from src.preference.score_candidates import load_preference_scorer
    from src.tools.executor import ToolExecutor, BaselineExecutor

    logger.info("Loading components for evaluation...")
    
    # Retriever / Searcher
    model_path = os.path.join(output_dir, "retriever")
    if not os.path.isdir(model_path):
        model_path = cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2")
    encoder = SentenceTransformer(model_path, device=str(device))
    searcher = load_searcher(index_dir, os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    
    # Router
    router = load_router(
        os.path.join(data_dir, "domain_centroids.json"),
        os.path.join(data_dir, "domain_keywords.json")
    )
    
    # Models
    triage     = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker   = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    pref       = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
    
    gen_path   = os.path.join(output_dir, "generator")
    if not os.path.isdir(gen_path):
        gen_path = None if cfg.get("generator_epochs", 0) == 0 else cfg.get("generator_model", "google/flan-t5-small")
    generator  = load_generator(gen_path, device=device)

    # Chunk by ID for GetPolicy
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    baseline_executor = BaselineExecutor(encoder, searcher, reranker, generator, cfg)
    proposed_executor = ToolExecutor(
        encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg
    )

    # 2. Load eval data
    eval_set = read_jsonl(os.path.join(data_dir, "eval_set.jsonl"))
    triage_trn = read_jsonl(os.path.join(data_dir, "triage_train.jsonl"))
    # Include some REJECT/TICKET examples in eval
    n_eval = cfg.get("max_eval_samples")
    if n_eval:
        eval_set = eval_set[:n_eval]
        extra_count = max(10, n_eval // 5)
    else:
        # If full eval, add a reasonable number of triage examples (e.g. 100)
        extra_count = 100
    
    extra = triage_trn[:extra_count]
    eval_set.extend(extra)
        
    # 3. Run E2E
    from src.evaluation.evaluate_end_to_end import run_e2e_eval
    import pandas as pd

    logger.info(f"Evaluating Baseline on {len(eval_set)} samples...")
    base_m, base_res = run_e2e_eval(baseline_executor, eval_set, "Baseline")
    
    logger.info(f"Evaluating Proposed on {len(eval_set)} samples...")
    prop_m, prop_res = run_e2e_eval(proposed_executor, eval_set, "Proposed")

    # Metrics table
    logger.info("Creating summary table...")
    import csv
    rows = [base_m, prop_m]
    keys = ["label", "EvidenceHit@5", "CitationPrecision", "TriageAccuracy", "AvgLatencyMs", "REE@5"]
    
    csv_path = os.path.join(rep_dir, "ablation_metrics.csv")
    logger.info(f"Writing to {csv_path}...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info("\n--- Results ---")
    print("label | EvidenceHit@5 | CitationPrecision | TriageAccuracy | AvgLatencyMs | REE@5")
    print("-" * 80)
    for r in rows:
        print(f"{r.get('label'):<10} | {r.get('EvidenceHit@5',0):.3f} | {r.get('CitationPrecision',0):.3f} | {r.get('TriageAccuracy',0):.3f} | {r.get('AvgLatencyMs',0):.1f} | {r.get('REE@5',0):.3f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
