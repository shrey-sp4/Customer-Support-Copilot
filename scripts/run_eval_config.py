import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import read_jsonl, write_json, ensure_dir
from src.evaluation.evaluate_end_to_end import run_e2e_eval
from src.tools.executor import ToolExecutor

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_hard", type=float, default=0.08)
    parser.add_argument("--tau_soft", type=float, default=0.20)
    args = parser.parse_args()

    config_path = "configs/smoke.yaml"
    eval_file = "data/processed/eval_md2d_natural_1000.jsonl"
    
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    
    # Override
    cfg.tau_hard_reject = args.tau_hard
    cfg.tau_soft_domain = args.tau_soft
    
    # Load components
    from sentence_transformers import SentenceTransformer
    from src.retrieval.search_kb import load_searcher
    from src.routing.router import load_router
    from src.triage.predict import load_predictor
    from src.reranking.rerank import load_reranker
    from src.generation.generate import load_generator
    from src.preference.score_candidates import load_preference_scorer

    data_dir   = cfg.get("data_dir", "data/processed")
    index_dir  = cfg.get("index_dir", "data/indexes")
    output_dir = cfg.get("output_dir", "outputs")
    
    encoder = SentenceTransformer(cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2"))
    searcher = load_searcher(index_dir, os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"))
    reranker = load_reranker(os.path.join(output_dir, "reranker"))
    pref = load_preference_scorer(os.path.join(output_dir, "preference"))
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    generator = load_generator(None)

    executor = ToolExecutor(encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg)
    
    eval_set = read_jsonl(eval_file)
    metrics, results = run_e2e_eval(executor, eval_set, f"Proposed_H{args.tau_hard}_S{args.tau_soft}")
    
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
