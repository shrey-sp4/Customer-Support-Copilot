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
    
    # --- 1. SETUP BASELINE-1 (TRUE RAW RAG) ---
    logger.info("Setting up Baseline-1 (True RAW RAG)...")
    raw_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    raw_index_dir = "data/indexes_raw"
    raw_searcher = load_searcher(raw_index_dir, os.path.join(data_dir, "kb_chunks.jsonl"), raw_encoder)
    
    baseline_executor = BaselineExecutor(raw_searcher, generator=None, cfg=cfg)

    # --- 2. SETUP BASELINE-2 (RULE WORKFLOW) ---
    logger.info("Setting up Baseline-2 (Rule Workflow)...")
    # Truly RAW: uses MiniLM and GLOBAL search
    # We use the same raw_encoder and raw_searcher as Baseline-1
    
    router = load_router(
        os.path.join(data_dir, "domain_centroids.json"), # These might be based on trained model, but we use MiniLM for query encoding
        os.path.join(data_dir, "domain_keywords.json")
    )
    
    rule_executor = RuleWorkflowExecutor(raw_searcher, router, generator=None, cfg=cfg)

    # --- 3. SETUP PROPOSED (CLUSTER GATED + DOMAIN INDEXES) ---
    logger.info("Setting up Proposed (Domain-Indexed)...")
    # Uses trained components
    trained_retriever_path = os.path.join(output_dir, "retriever")
    if os.path.isdir(trained_retriever_path):
        trained_encoder = SentenceTransformer(trained_retriever_path, device=str(device))
        logger.info(f"Loaded trained retriever for Proposed from {trained_retriever_path}")
    else:
        trained_encoder = raw_encoder
        
    global_index_dir = cfg.get("index_dir", "data/indexes")
    
    triage     = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker   = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    pref       = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
    
    gen_path   = os.path.join(output_dir, "generator")
    if not os.path.isdir(gen_path):
        gen_path = None
    generator  = load_generator(gen_path, device=device, cfg=cfg)

    domain_indexes_dir = "data/indexes_by_domain"
    proposed_searcher = load_searcher(
        global_index_dir, 
        os.path.join(data_dir, "kb_chunks.jsonl"), 
        trained_encoder,
        domain_indexes_dir=domain_indexes_dir
    )
    
    # Chunk by ID for GetPolicy
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    proposed_executor = ToolExecutor(
        trained_encoder, proposed_searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg
    )

    # --- Print System Configurations for Transparency ---
    logger.info("\n" + "="*50)
    logger.info("SYSTEM CONFIGURATIONS")
    logger.info("="*50)
    logger.info(f"Baseline-1 (RAW):")
    logger.info(f"  Encoder: sentence-transformers/all-MiniLM-L6-v2")
    logger.info(f"  Index:   {raw_index_dir} (Global Only)")
    logger.info(f"  Triage:  None (Always ANSWER)")
    logger.info(f"  Reranker: None")
    
    logger.info(f"Baseline-2 (Rule):")
    logger.info(f"  Encoder: sentence-transformers/all-MiniLM-L6-v2")
    logger.info(f"  Index:   {raw_index_dir} (Global Only)")
    logger.info(f"  Triage:  Rule-based (3 responses)")
    
    logger.info(f"Proposed:")
    logger.info(f"  Encoder: {'Trained' if trained_encoder != raw_encoder else 'MiniLM'}")
    logger.info(f"  Index:   {domain_indexes_dir} (Domain-Specific)")
    logger.info(f"  Triage:  BERT-based")
    logger.info(f"  Reranker: {'Yes' if reranker else 'None'}")
    logger.info("="*50 + "\n")

    # 4. Load eval data
    eval_set = read_jsonl(os.path.join(data_dir, "balanced_eval.jsonl"))
        
    # 5. Run E2E
    from src.evaluation.evaluate_end_to_end import run_e2e_eval
    import pandas as pd

    logger.info(f"Evaluating Baseline-1 (RAW RAG) on {len(eval_set)} samples...")
    base_m, base_res = run_e2e_eval(baseline_executor, eval_set, "Baseline-1")
    
    logger.info(f"Evaluating Baseline-2 (Rule Workflow) on {len(eval_set)} samples...")
    rule_m, rule_res = run_e2e_eval(rule_executor, eval_set, "Baseline-2")

    logger.info(f"Evaluating Proposed (Domain-Indexed) on {len(eval_set)} samples...")
    prop_m, prop_res = run_e2e_eval(proposed_executor, eval_set, "Proposed")

    # Metrics table
    logger.info("Creating summary table...")
    import csv
    
    write_json(base_m, os.path.join(rep_dir, "baseline_metrics.json"))
    write_json(rule_m, os.path.join(rep_dir, "rule_workflow_baseline_metrics.json"))
    write_json(prop_m, os.path.join(rep_dir, "proposed_metrics.json"))

    rows = [base_m, rule_m, prop_m]
    keys = [
        "label", "EvidenceHit@5", "CitationDocPrecision", "TriageAccuracy", "MacroF1", 
        "AvgLatencyMs", "AvgRouting_ms", "AvgSearch_ms", "AvgRerank_ms", "AvgGen_ms",
        "AvgClustersSearched", "AvgFractionKBScanned", "REE@5"
    ]
    
    csv_path = os.path.join(rep_dir, "ablation_metrics.csv")
    write_json(rows, os.path.join(rep_dir, "all_results.json")) # For transparency
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info("\n--- Results ---")
    header = "label      | EvHit@5 | TriAcc | MacroF1 | LatMs | Route | Search| Rerank| Gen   | FracKB | REE@5"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.get('label'):<10} | {r.get('EvidenceHit@5',0):.3f} | "
              f"{r.get('TriageAccuracy',0):.3f} | {r.get('MacroF1',0):.3f} | {r.get('AvgLatencyMs',0):.1f} | "
              f"{r.get('AvgRouting_ms',0):.1f} | {r.get('AvgSearch_ms',0):.1f} | {r.get('AvgRerank_ms',0):.1f} | {r.get('AvgGen_ms',0):.1f} | "
              f"{r.get('AvgFractionKBScanned',0):.3f} | {r.get('REE@5',0):.3f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
