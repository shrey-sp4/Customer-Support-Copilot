import os
import sys
import json
import time
import pandas as pd
from typing import List
from tqdm import tqdm
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.io import read_jsonl, ensure_dir
from src.tools.executor import ToolExecutor, BaselineExecutor, RuleWorkflowExecutor
from src.retrieval.search_kb import load_searcher
from src.routing.router import load_router
from src.triage.predict import load_predictor
from src.reranking.rerank import load_reranker
from src.generation.generate import load_generator
from sentence_transformers import SentenceTransformer

def evaluate_subset(executor, eval_set: List[dict], n=50):
    subset = eval_set[:n]
    correct = 0
    total_latency = 0
    results = []
    
    for item in subset:
        query = item["query"]
        gold = item.get("gold_triage") or item.get("gold_label", "ANSWER")
        
        t0 = time.time()
        res = executor.run(query)
        t1 = time.time()
        
        total_latency += (t1 - t0) * 1000
        is_correct = 1 if res["decision"] == gold else 0
        correct += is_correct
        results.append({
            "gold": gold,
            "pred": res["decision"],
            "correct": is_correct,
            "latency": (t1 - t0) * 1000,
            "answer": res.get("final_answer", "")[:50]
        })
        
    return {
        "accuracy": correct / len(subset),
        "avg_latency": total_latency / len(subset),
        "results": results
    }

def main():
    cfg = load_config("configs/smoke.yaml")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_dir = "data/processed"
    output_dir = "outputs"
    
    encoder = SentenceTransformer(cfg.get("retriever_model"), device=device)
    searcher = load_searcher(cfg.get("index_dir"), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    generator = load_generator(None, device=device, cfg=cfg)
    
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    
    proposed = ToolExecutor(encoder, searcher, router, triage, reranker, generator, None, chunk_by_id, cfg)
    b1 = BaselineExecutor(encoder, searcher, reranker, generator, cfg)
    b2 = RuleWorkflowExecutor(encoder, searcher, router, reranker, generator, cfg)
    
    eval_set = read_jsonl(os.path.join(data_dir, "eval_workflow_balanced_300.jsonl"))
    
    print("\n--- QUICK COMPARISON (Balanced Set - First 50 samples) ---")
    
    print("Evaluating Baseline-1...")
    m1 = evaluate_subset(b1, eval_set, 50)
    print(f"B1 Acc: {m1['accuracy']:.3f}, Latency: {m1['avg_latency']:.1f}ms")
    
    print("Evaluating Baseline-2...")
    m2 = evaluate_subset(b2, eval_set, 50)
    print(f"B2 Acc: {m2['accuracy']:.3f}, Latency: {m2['avg_latency']:.1f}ms")
    
    print("Evaluating Proposed...")
    mp = evaluate_subset(proposed, eval_set, 50)
    print(f"Prop Acc: {mp['accuracy']:.3f}, Latency: {mp['avg_latency']:.1f}ms")
    
    # Corrected quality metrics call
    from src.evaluation.quality import compute_answer_quality_metrics
    # We need to run executor.run and keep the full result dict
    def get_full_results(executor, subset):
        full_res = []
        for item in subset:
            res = executor.run(item["query"])
            res["query"] = item["query"]
            res["gold_domain"] = item.get("gold_domain", "")
            full_res.append(res)
        return full_res
    
    print("\nComputing Quality Metrics for Proposed...")
    subset_50 = eval_set[:50]
    full_results_prop = get_full_results(proposed, subset_50)
    quality_prop = compute_answer_quality_metrics(full_results_prop)
    
    print("\n--- PROPOSED QUALITY METRICS (Subset 50) ---")
    for k, v in quality_prop.items():
        print(f"{k}: {v:.4f}")

    df = pd.DataFrame([
        {"System": "Baseline-1", "Accuracy": m1["accuracy"], "Latency": m1["avg_latency"]},
        {"System": "Baseline-2", "Accuracy": m2["accuracy"], "Latency": m2["avg_latency"]},
        {"System": "Proposed", "Accuracy": mp["accuracy"], "Latency": mp["avg_latency"]}
    ])
    print("\n" + df.to_markdown(index=False))

if __name__ == "__main__":
    main()
