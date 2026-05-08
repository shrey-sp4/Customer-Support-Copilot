import argparse
import os
import sys
import json
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import read_jsonl, ensure_dir, write_json
from src.tools.executor import ToolExecutor, BaselineExecutor, RuleWorkflowExecutor
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)
console = Console()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Answer Quality")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_dir   = cfg.get("data_dir", "data/processed")
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
    from src.evaluation.evaluate_end_to_end import run_e2e_eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = os.path.join(output_dir, "retriever")
    if not os.path.isdir(model_path):
        model_path = cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2")
    encoder = SentenceTransformer(model_path, device=device)
    searcher = load_searcher(os.path.join(cfg.get("index_dir", "data/indexes")), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    pref = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
    
    gen_path = os.path.join(output_dir, "generator")
    if not os.path.isdir(gen_path):
        gen_path = None
    generator = load_generator(gen_path, device=device, cfg=cfg)

    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    baseline_executor = BaselineExecutor(encoder, searcher, reranker, generator, cfg)
    rule_executor     = RuleWorkflowExecutor(encoder, searcher, router, reranker, generator, cfg)
    proposed_executor = ToolExecutor(encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg)

    # 2. Load eval data
    eval_set = read_jsonl(os.path.join(data_dir, "eval_answer_quality_100.jsonl"))

    # 3. Run E2E
    logger.info(f"Evaluating systems on {len(eval_set)} quality samples...")
    base_m, _ = run_e2e_eval(baseline_executor, eval_set, "Baseline-1")
    rule_m, _ = run_e2e_eval(rule_executor, eval_set, "Baseline-2")
    prop_m, prop_res = run_e2e_eval(proposed_executor, eval_set, "Proposed")

    # 4. Save metrics
    write_json(prop_m, os.path.join(rep_dir, "answer_quality_metrics.json"))
    from src.utils.io import write_jsonl
    write_jsonl(prop_res, os.path.join(rep_dir, "proposed_quality_results.jsonl"))

    # 5. Report
    table = Table(title="Answer Quality Evaluation Results")
    table.add_column("System", style="bold")
    table.add_column("EvHit@5")
    table.add_column("QualScore")
    table.add_column("FragRate")
    table.add_column("BadPunctRate")
    table.add_column("AvgLenWords")
    table.add_column("UnsuppRate")

    for m in [base_m, rule_m, prop_m]:
        table.add_row(
            m.get("label"),
            f"{m.get('EvidenceHit@5', 0):.3f}",
            f"{m.get('avg_answer_quality_rubric_score', 0):.3f}",
            f"{m.get('answer_fragment_rate', 0):.3f}",
            f"{m.get('bad_spacing_or_apostrophe_rate', 0):.3f}",
            f"{m.get('avg_answer_length_words', 0):.1f}",
            f"{m.get('unsupported_answer_rate', 0):.3f}"
        )

    console.print(table)

if __name__ == "__main__":
    import torch
    main()
