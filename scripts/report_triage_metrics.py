import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.io import read_jsonl, ensure_dir
from src.triage.predict import load_predictor
from src.routing.router import load_router
from src.retrieval.search_kb import load_searcher

def main():
    cfg = load_config("configs/smoke.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs"
    data_dir = "data/processed"
    ensure_dir("outputs/reports")
    
    # Load components for feature extraction
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    encoder = SentenceTransformer(cfg.get("retriever_model"), device=device)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    searcher = load_searcher(cfg.get("index_dir"), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    
    eval_set = read_jsonl(os.path.join(data_dir, "eval_workflow_balanced_300.jsonl"))
    
    y_true = []
    y_pred = []
    
    print("Evaluating Triage model on Balanced set with real features...")
    for item in tqdm(eval_set):
        query = item["query"]
        gold = item.get("gold_triage") or item.get("gold_label", "ANSWER")
        
        # Extract features (same as ToolExecutor.run)
        query_emb = encoder.encode([query])[0]
        route_res = router.route(query_emb, top_k=2)
        top_sim = route_res[0]["centroid_similarity"] if route_res else 0.0
        margin = (route_res[0]["centroid_similarity"] - route_res[1]["centroid_similarity"]) if len(route_res) > 1 else 0.0
        
        kws = router.lexical_gate.get_matched_support_keywords(query)
        kw_gate = "pass" if kws else "reject"
        
        # Retrieval features
        kb_res = searcher.search(query, top_k=5, domain=None)
        best_ev = kb_res[0]["score"] if kb_res else 0.0
        gap = (kb_res[0]["score"] - kb_res[1]["score"]) if len(kb_res) > 1 else 0.0
        
        res = triage.predict(
            query=query,
            keyword_gate=kw_gate,
            centroid_domain=route_res[0]["domain"] if route_res else "unknown",
            centroid_sim_top1=top_sim,
            centroid_margin=margin,
            nearest_chunk_sim=best_ev,
            retrieval_score_gap=gap
        )
        pred = res["decision"]
        
        y_true.append(gold)
        y_pred.append(pred)
        
    # Metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    labels = ["ANSWER", "TICKET", "REJECT"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Triage Model Confusion Matrix')
    plt.savefig("outputs/reports/triage_confusion_matrix.png")
    
    # Save metrics JSON
    with open("outputs/reports/triage_metrics_detailed.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\nSaved report to outputs/reports/triage_metrics_detailed.json")
    print("Saved confusion matrix to outputs/reports/triage_confusion_matrix.png")

if __name__ == "__main__":
    main()
