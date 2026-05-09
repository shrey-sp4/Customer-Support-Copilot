import argparse
import os
import sys
import json
import itertools
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.io import read_jsonl, ensure_dir
from src.tools.executor import ToolExecutor
from sentence_transformers import SentenceTransformer
from src.retrieval.search_kb import load_searcher
from src.routing.router import load_router
from src.triage.predict import load_predictor
from src.reranking.rerank import load_reranker
from src.generation.generate import load_generator

def run_eval(executor, eval_set: List[dict]) -> dict:
    hits = 0
    correct_triage = 0
    preds = []
    golds = []
    
    for item in eval_set:
        res = executor.run(item["query"])
        preds.append(res["decision"])
        golds.append(item.get("gold_triage") or "ANSWER")
        
        # Grounding check
        if res["decision"] == "ANSWER" and res.get("citations"):
            hits += 1 # Simplified for speed
            
    from sklearn.metrics import f1_score
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    return {"macro_f1": f1, "answer_rate": preds.count("ANSWER") / len(preds)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--eval_file", default="data/processed/eval_workflow_balanced_300.jsonl")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(42)
    
    data_dir = cfg.get("data_dir", "data/processed")
    output_dir = cfg.get("output_dir", "outputs")
    report_dir = os.path.join(output_dir, "reports")
    ensure_dir(report_dir)
    
    eval_set = read_jsonl(args.eval_file)[:50] # Subset for speed
    
    # Load components
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2"), device=device)
    searcher = load_searcher(cfg.get("index_dir", "data/indexes"), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    generator = load_generator(None, device=device, cfg=cfg) # Template fallback for speed
    
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    
    # Grid Search Space
    tau_domains = [0.25, 0.30, 0.35, 0.40]
    tau_chunks = [0.40, 0.45, 0.50, 0.55]
    evidence_thresholds = [0.45, 0.50, 0.55]
    
    results = []
    
    param_grid = list(itertools.product(tau_domains, tau_chunks, evidence_thresholds))
    print(f"Starting Grid Search over {len(param_grid)} combinations...")
    
    for t_dom, t_ch, e_th in tqdm(param_grid):
        # Update config dynamically using the object's dictionary or setattr
        setattr(cfg, "tau_domain", t_dom)
        setattr(cfg, "tau_chunk", t_ch)
        setattr(cfg, "evidence_answer_threshold", e_th)
        
        executor = ToolExecutor(encoder, searcher, router, triage, reranker, generator, None, chunk_by_id, cfg)
        
        metrics = run_eval(executor, eval_set)
        results.append({
            "tau_domain": t_dom,
            "tau_chunk": t_ch,
            "evidence_threshold": e_th,
            "macro_f1": metrics["macro_f1"],
            "answer_rate": metrics["answer_rate"]
        })
        
    df = pd.DataFrame(results)
    output_path = os.path.join(report_dir, "threshold_optimization_results.csv")
    df.to_csv(output_path, index=False)
    
    best = df.loc[df["macro_f1"].idxmax()]
    print(f"\nOptimization Complete. Best Config:")
    print(best)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
