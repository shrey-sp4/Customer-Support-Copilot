import argparse
import os
import sys
import json
import time
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import read_jsonl, ensure_dir, write_json
from src.tools.executor import ToolExecutor, BaselineExecutor, RuleWorkflowExecutor

logger = get_logger(__name__)

def evaluate_system(executor, eval_set: List[dict], system_name: str) -> dict:
    results = []
    logger.info(f"Evaluating {system_name} on {len(eval_set)} samples...")
    
    total_latency = 0
    total_fraction_kb = 0
    
    for item in tqdm(eval_set, desc=system_name):
        query = item["query"]
        gold_decision = item.get("gold_triage") or item.get("gold_label", "ANSWER")
        
        t_start = time.time()
        res = executor.run(query)
        t_end = time.time()
        
        latency = (t_end - t_start) * 1000
        total_latency += latency
        total_fraction_kb += res.get("fraction_kb", 1.0)
        
        # Check grounding (EvidenceHit@5)
        # This is a bit complex for a generic script, but let's do a simple version
        # If ANSWER, check if gold_chunk_id is in tool_trace passages
        hit = 0
        gold_chunk = item.get("gold_chunk_id")
        if gold_chunk:
            for trace in res.get("tool_trace", []):
                if trace["tool"] == "SearchKB":
                    passages = trace["result"].get("passages", [])
                    if any(p.get("chunk_id") == gold_chunk for p in passages[:5]):
                        hit = 1
                        break
        
        results.append({
            "query": query,
            "gold_decision": gold_decision,
            "predicted_decision": res["decision"],
            "hit": hit,
            "latency": latency,
            "res": res,
            "item": item
        })
        
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    preds = [r["predicted_decision"] for r in results]
    golds = [r["gold_decision"] for r in results]
    
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    hit_rate = sum(r["hit"] for r in results) / max(len([r for r in results if r.get("item",{}).get("gold_chunk_id")]), 1)
    
    metrics = {
        "system": system_name,
        "accuracy": acc,
        "macro_f1": f1,
        "evidence_hit_at_5": hit_rate,
        "avg_latency": total_latency / len(eval_set),
        "avg_fraction_kb": total_fraction_kb / len(eval_set),
        "ree_at_5": hit_rate / (total_fraction_kb / len(eval_set)) if total_fraction_kb > 0 else 0.0
    }
    
    return metrics, results

def generate_error_analysis(results: List[dict], output_path: str, executor_mode: str):
    error_data = []
    for r in results:
        res = r["res"]
        item = r["item"]
        
        # Extract features for analysis
        gate_res = next((t for t in res["tool_trace"] if t["tool"] == "RouteDomain"), {}).get("result", {})
        search_res = next((t for t in res["tool_trace"] if t["tool"] == "TriagePredictor"), {}).get("result", {})
        kb_res = next((t for t in res["tool_trace"] if t["tool"] == "SearchKB"), {}).get("result", {})
        
        error_type = "CORRECT" if r["predicted_decision"] == r["gold_decision"] else f"{r['gold_decision']}->{r['predicted_decision']}"
        
        from src.tools.executor import is_personal_or_action_request
        
        error_data.append({
            "query": r["query"],
            "gold_decision": r["gold_decision"],
            "predicted_decision": r["predicted_decision"],
            "error_type": error_type,
            "gold_domain": item.get("gold_domain", "unknown"),
            "selected_domains": gate_res.get("selected_domains", []),
            "top_sim": gate_res.get("top_centroid_sim", 0.0),
            "support_keywords": gate_res.get("matched_keywords", []),
            "personal_action_flag": is_personal_or_action_request(r["query"]),
            "best_evidence_score": kb_res.get("best_evidence_score", 0.0) if "best_evidence_score" in kb_res else (res.get("best_evidence_score", 0.0)),
            "final_evidence_doc_id": next((p.get("doc_id") for p in res.get("final_evidence", [])), "None") if "final_evidence" in res else "None",
            "generator_mode": executor_mode
        })
    
    df = pd.DataFrame(error_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Error analysis saved to {output_path}")
    
    # Print top errors
    if not df.empty:
        errors = df[df["error_type"] != "CORRECT"]
        if not errors.empty:
            print("\n--- Top Errors ---")
            print(errors["error_type"].value_counts().head(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    
    data_dir = cfg.get("data_dir", "data/processed")
    output_dir = cfg.get("output_dir", "outputs")
    report_dir = os.path.join(output_dir, "reports")
    ensure_dir(report_dir)
    
    # Load eval sets
    eval_sets = {
        "natural": read_jsonl(os.path.join(data_dir, "eval_md2d_natural_1000.jsonl")),
        "balanced": read_jsonl(os.path.join(data_dir, "eval_workflow_balanced_300.jsonl")),
        "robustness": read_jsonl(os.path.join(data_dir, "eval_robustness_150.jsonl"))
    }
    
    from src.retrieval.search_kb import load_searcher
    from src.routing.router import load_router
    from src.triage.predict import load_predictor
    from src.reranking.rerank import load_reranker
    from src.generation.generate import load_generator
    from src.preference.score_candidates import load_preference_scorer
    from src.tools.executor import ToolExecutor
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2"), device=device)
    searcher = load_searcher(cfg.get("index_dir", "data/indexes"), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    # Corrected paths to match training outputs
    gen_path   = os.path.join(output_dir, "generator_lora")
    pref_path  = os.path.join(output_dir, "preference_dpo")
    if not os.path.isdir(gen_path):
        gen_path = None
    generator = load_generator(gen_path, device=device, cfg=cfg)
    
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    
    proposed = ToolExecutor(encoder, searcher, router, triage, reranker, generator, None, chunk_by_id, cfg)
    
    # Baseline 1
    b1 = BaselineExecutor(encoder, searcher, reranker, generator, cfg)
    
    # Baseline 2
    b2 = RuleWorkflowExecutor(encoder, searcher, router, reranker, generator, cfg)
    
    all_metrics = []
    
    for set_name, eval_set in eval_sets.items():
        if not eval_set: continue
        print(f"\n=== EVALUATING ON {set_name.upper()} SET ===")
        
        m_b1, _ = evaluate_system(b1, eval_set, f"Baseline-1 ({set_name})")
        m_b2, _ = evaluate_system(b2, eval_set, f"Baseline-2 ({set_name})")
        m_prop, results_prop = evaluate_system(proposed, eval_set, f"Proposed ({set_name})")
        
        all_metrics.extend([m_b1, m_b2, m_prop])
        
        if set_name == "natural":
            generate_error_analysis(results_prop, os.path.join(report_dir, "proposed_error_analysis.csv"), proposed.generator_mode)

    # Final report
    df_metrics = pd.DataFrame(all_metrics)
    print("\n=== FINAL METRICS SUMMARY ===")
    cols = ["system", "accuracy", "macro_f1", "recall_at_1", "recall_at_3", "evidence_hit_at_5", "ree_at_5", "avg_latency"]
    # Filter columns that exist
    cols = [c for c in cols if c in df_metrics.columns]
    print(df_metrics[cols])
    
    df_metrics.to_csv(os.path.join(report_dir, "final_results.csv"), index=False)
    write_json(all_metrics, os.path.join(report_dir, "final_metrics.json"))
    
    # Run Quality Report for Proposed system on Natural set
    if "results_prop" in locals():
        from src.evaluation.quality import compute_answer_quality_metrics
        # Pass encoder for Neural Semantic Fidelity
        quality_metrics = compute_answer_quality_metrics([r["res"] for r in results_prop], encoder=encoder)
        write_json(quality_metrics, os.path.join(report_dir, "final_quality_metrics.json"))
        
        print("\n=== PROPOSED QUALITY METRICS (NATURAL SET) ===")
        for k, v in quality_metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    import torch
    main()
