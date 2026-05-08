import os
import json
import csv
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import read_jsonl, ensure_dir
from src.tools.executor import ToolExecutor

logger = get_logger(__name__)

def main():
    config_path = "configs/smoke.yaml"
    eval_file = "data/processed/eval_md2d_natural_1000.jsonl"
    
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    
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
    analysis_rows = []
    false_rejects = []

    logger.info(f"Analyzing {len(eval_set)} samples...")
    for sample in eval_set:
        query = sample["query"]
        gold_decision = sample.get("gold_decision", sample.get("gold_triage", "ANSWER"))
        
        result = executor.run(query)
        
        # Extract metadata from tool_trace
        route_info = next((t for t in result["tool_trace"] if t["tool"] == "RouteDomain"), {})
        gating_info = next((t for t in result["tool_trace"] if t["tool"] == "ClusterGating"), {})
        
        top_sim = route_info.get("top_centroid_sim", 0.0)
        margin = route_info.get("centroid_margin", 0.0)
        matched_kws = route_info.get("support_keywords", [])
        gate_decision = gating_info.get("result", {}).get("decision", "unknown")
        
        row = {
            "query": query,
            "gold_decision": gold_decision,
            "predicted_decision": result["decision"],
            "top_centroid_similarity": top_sim,
            "margin": margin,
            "matched_keywords": "|".join(matched_kws),
            "support_keyword_count": len(matched_kws),
            "selected_domains": "|".join(result.get("selected_domains", [])),
            "gate_decision": gate_decision,
            "retrieval_called": any(t["tool"] == "SearchKB" for t in result["tool_trace"]),
            "final_decision": result["decision"]
        }
        analysis_rows.append(row)
        
        if gold_decision in ["ANSWER", "TICKET"] and result["decision"] == "REJECT":
            false_rejects.append({
                "query": query,
                "gold": gold_decision,
                "gold_domain": sample.get("gold_domain", "unk"),
                "top_sim": top_sim,
                "kws": "|".join(matched_kws)
            })

    # Save CSV
    ensure_dir("outputs/reports")
    pd.DataFrame(analysis_rows).to_csv("outputs/reports/domain_gate_analysis.csv", index=False)
    
    # Save Markdown
    with open("outputs/reports/false_reject_examples.md", "w") as f:
        f.write("# False Reject Examples Analysis\n\n")
        f.write(f"Total False Rejects: {len(false_rejects)}\n\n")
        f.write("| Query | Gold | Domain | TopSim | Keywords |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for fr in false_rejects[:50]: # top 50
            f.write(f"| {fr['query']} | {fr['gold']} | {fr['gold_domain']} | {fr['top_sim']:.3f} | {fr['kws']} |\n")

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
