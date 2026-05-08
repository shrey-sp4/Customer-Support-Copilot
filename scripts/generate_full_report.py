import os
import sys
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.io import read_jsonl, ensure_dir, write_json
from src.tools.executor import ToolExecutor, BaselineExecutor, RuleWorkflowExecutor
from src.retrieval.search_kb import load_searcher
from src.routing.router import load_router
from src.triage.predict import load_predictor
from src.reranking.rerank import load_reranker
from src.generation.generate import load_generator
from src.evaluation.quality import compute_answer_quality_metrics
from sentence_transformers import SentenceTransformer

def get_metrics(results: List[Dict], eval_set: List[Dict], system_name: str):
    # 1. Triage Metrics
    y_true = [item.get("gold_triage") or item.get("gold_label", "ANSWER") for item in eval_set]
    y_pred = [res["decision"] for res in results]
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ans_f1 = report.get("ANSWER", {}).get("f1-score", 0.0)
    tkt_f1 = report.get("TICKET", {}).get("f1-score", 0.0)
    rej_f1 = report.get("REJECT", {}).get("f1-score", 0.0)
    
    # 2. Retrieval Metrics
    hits = []
    doc_hits = []
    precisions = []
    for res, gold in zip(results, eval_set):
        gold_chunks = gold.get("gold_chunks", [])
        gold_docs = gold.get("gold_docs", [])
        
        pred_chunks = []
        pred_docs = []
        for trace in res.get("tool_trace", []):
            if trace["tool"] == "SearchKB":
                passages = trace["result"].get("passages", [])
                pred_chunks = [p["chunk_id"] for p in passages[:5]]
                pred_docs = [p["doc_id"] for p in passages[:5]]
                break
        
        hit = 1 if any(c in gold_chunks for c in pred_chunks) else 0
        doc_hit = 1 if any(d in gold_docs for d in pred_docs) else 0
        hits.append(hit)
        doc_hits.append(doc_hit)
        
        # Citation precision
        citations = res.get("citations", [])
        if citations:
            p_prec = sum(1 for c in citations if any(d in c for d in gold_docs)) / len(citations)
            precisions.append(p_prec)
        else:
            precisions.append(0.0)

    # 3. Quality Metrics
    quality = compute_answer_quality_metrics(results)
    
    # 4. Efficiency
    avg_latency = np.mean([res["latency_ms"] for res in results])
    avg_fraction = np.mean([res.get("fraction_kb", 1.0) for res in results])
    
    # REE@5 calculation (simplified)
    # REE = Accuracy / FractionKB
    ree = acc / avg_fraction if avg_fraction > 0 else 0
    
    return {
        "EvidenceHit@5": np.mean(hits),
        "EvidenceDocHit@5": np.mean(doc_hits),
        "CitationDocPrecision": np.mean(precisions),
        "Triage Accuracy": acc,
        "Macro-F1": macro_f1,
        "ANSWER F1": ans_f1,
        "TICKET F1": tkt_f1,
        "REJECT F1": rej_f1,
        "UnsupportedAnswerRate": quality.get("UnsupportedAnswerRate", 0.0),
        "WrongDomainCitationRate": quality.get("WrongDomainCitationRate", 0.0),
        "DirectAnswerRate": quality.get("DirectAnswerRate", 0.0),
        "FragmentRate": quality.get("FragmentRate", 0.0),
        "RepetitionRate": quality.get("RepetitionRate", 0.0),
        "AnswerQualityScore": quality.get("AnswerQualityScore", 0.0),
        "Avg Fraction KB Scanned": avg_fraction,
        "REE@5": ree,
        "Avg Latency": avg_latency
    }

def main():
    cfg = load_config("configs/smoke.yaml")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_dir = "data/processed"
    output_dir = "outputs"
    ensure_dir("outputs/reports")
    
    # Load components
    encoder = SentenceTransformer(cfg.get("retriever_model"), device=device)
    searcher = load_searcher(cfg.get("index_dir"), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    generator = load_generator(None, device=device, cfg=cfg)
    
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    
    # Executors
    proposed = ToolExecutor(encoder, searcher, router, triage, reranker, generator, None, chunk_by_id, cfg)
    b1 = BaselineExecutor(encoder, searcher, reranker, generator, cfg)
    b2 = RuleWorkflowExecutor(encoder, searcher, router, reranker, generator, cfg)
    
    # Eval Set (Subset for speed)
    eval_set_full = read_jsonl(os.path.join(data_dir, "eval_md2d_natural_1000.jsonl"))
    eval_set = eval_set_full[:200] # Use 200 samples
    
    systems = [
        ("Baseline-1", b1),
        ("Baseline-2", b2),
        ("Proposed", proposed)
    ]
    
    final_results = {}
    
    for name, exec_obj in systems:
        print(f"\nEvaluating {name}...")
        results = []
        for item in tqdm(eval_set):
            res = exec_obj.run(item["query"])
            # Inject gold for quality metrics if needed
            res["query"] = item["query"]
            res["gold_domain"] = item.get("gold_domain", "")
            results.append(res)
            
        final_results[name] = get_metrics(results, eval_set, name)
        
    # Pivot to final table
    df = pd.DataFrame(final_results).T
    df.index.name = "Metric"
    df = df.T
    
    # Order metrics as requested
    metric_order = [
        "EvidenceHit@5", "EvidenceDocHit@5", "CitationDocPrecision",
        "Triage Accuracy", "Macro-F1", "ANSWER F1", "TICKET F1", "REJECT F1",
        "UnsupportedAnswerRate", "WrongDomainCitationRate", "DirectAnswerRate",
        "FragmentRate", "RepetitionRate", "AnswerQualityScore",
        "Avg Fraction KB Scanned", "REE@5", "Avg Latency"
    ]
    
    df = df.reindex(metric_order)
    
    print("\n=== FINAL COMPARISON REPORT (200 SAMPLES) ===")
    print(df.to_markdown())
    
    df.to_csv("outputs/reports/full_comparison_report.csv")
    
    with open("outputs/reports/full_comparison_report.md", "w") as f:
        f.write("# Full Comparison Report\n\n")
        f.write(df.to_markdown())
        f.write("\n\n*Evaluated on 200 samples from MD2D Natural set.*")

if __name__ == "__main__":
    main()
