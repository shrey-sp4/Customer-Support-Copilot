"""scripts/evaluate.py — Run E2E evaluation of baseline and proposed systems."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import ensure_dir, read_jsonl, write_json
from sentence_transformers import SentenceTransformer

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
    
    ensure_dir(rep_dir)

    # 1. Load components
    from src.retrieval.search_kb import load_searcher
    from src.routing.router import load_router
    from src.triage.predict import load_predictor
    from src.reranking.rerank import load_reranker
    from src.generation.generate import load_generator
    from src.preference.score_candidates import load_preference_scorer
    from src.tools.executor import ToolExecutor, BaselineExecutor, RuleWorkflowExecutor

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
        gen_path = None
    generator  = load_generator(gen_path, device=device, cfg=cfg)

    # Chunk by ID for GetPolicy
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    baseline_executor = BaselineExecutor(encoder, searcher, reranker, generator, cfg)
    rule_executor     = RuleWorkflowExecutor(encoder, searcher, router, reranker, generator, cfg)
    proposed_executor = ToolExecutor(
        encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg
    )

    # 2. Load eval data (Balanced set: 30 ANSWER, 30 TICKET, 30 REJECT)
    eval_set = read_jsonl(os.path.join(data_dir, "balanced_eval.jsonl"))
        
    # 3. Run E2E
    from src.evaluation.evaluate_end_to_end import run_e2e_eval
    import pandas as pd

    logger.info(f"Evaluating Baseline-1 (Simple RAG) on {len(eval_set)} samples...")
    base_m, base_res = run_e2e_eval(baseline_executor, eval_set, "Baseline-1")
    
    logger.info(f"Evaluating Baseline-2 (Rule Workflow) on {len(eval_set)} samples...")
    rule_m, rule_res = run_e2e_eval(rule_executor, eval_set, "Baseline-2")

    logger.info(f"Evaluating Proposed (Cluster Gated) on {len(eval_set)} samples...")
    prop_m, prop_res = run_e2e_eval(proposed_executor, eval_set, "Proposed")

    # Metrics table
    logger.info("Creating summary table...")
    import csv
    
    # Save individual JSONs
    write_json(base_m, os.path.join(rep_dir, "baseline_metrics.json"))
    write_json(rule_m, os.path.join(rep_dir, "rule_workflow_baseline_metrics.json"))
    write_json(prop_m, os.path.join(rep_dir, "proposed_metrics.json"))

    # Print label distribution
    logger.info(f"Label Distribution: {base_m.get('LabelDistGold')}")

    rows = [base_m, rule_m, prop_m]
    keys = [
        "label", "EvidenceHit@5", "CitationDocPrecision", "TriageAccuracy", "MacroF1", "WeightedF1",
        "AvgLatencyMs", "AvgClustersSearched", "AvgFractionKBScanned", "REE@5"
    ]
    
    csv_path = os.path.join(rep_dir, "ablation_metrics.csv")
    logger.info(f"Writing to {csv_path}...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info("\n--- Results ---")
    header = "label      | EvHit@5 | CitDocPrec | TriAcc | MacroF1 | LatMs | FracKB | REE@5"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.get('label'):<10} | {r.get('EvidenceHit@5',0):.3f} | {r.get('CitationDocPrecision',0):.3f} | "
              f"{r.get('TriageAccuracy',0):.3f} | {r.get('MacroF1',0):.3f} | {r.get('AvgLatencyMs',0):.1f} | "
              f"{r.get('AvgFractionKBScanned',0):.3f} | {r.get('REE@5',0):.3f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
