"""scripts/evaluate.py — Run E2E evaluation of baseline and proposed systems."""
import argparse
import os
import sys
import csv
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End pipeline")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from src.utils.config import load_config
        from src.utils.seed import set_seed
        from src.utils.logging import get_logger
        from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
        from src.utils.device import get_device

        cfg = load_config(args.config)
        set_seed(cfg.get("seed", 42))

        device_str = getattr(cfg, "device", "auto")
        device = get_device(device_str)

        logger = get_logger(__name__)

        data_dir   = cfg.get("data_dir", "data/processed")
        index_dir  = cfg.get("index_dir", "data/indexes")
        output_dir = cfg.get("output_dir", "outputs")

        raw_model_name = cfg.get("raw_retriever_model", "sentence-transformers/all-MiniLM-L6-v2")
        raw_index_dir = cfg.get("raw_index_dir", "data/indexes_raw")
        domain_indexes_dir = cfg.get("domain_indexes_dir", "data/indexes_by_domain")
        eval_file = cfg.get("eval_file", "balanced_eval.jsonl")
        rep_dir = cfg.get("reports_dir", os.path.join(output_dir, "reports"))
        
        ensure_dir(rep_dir)
        
        # --- 0. ARCHIVE STALE REPORTS ---
        archive_dir = os.path.join(rep_dir, "archive")
        ensure_dir(archive_dir)
        for f in os.listdir(rep_dir):
            if f.endswith(".json") or f.endswith(".csv") or f.endswith(".jsonl"):
                try:
                    os.rename(os.path.join(rep_dir, f), os.path.join(archive_dir, f))
                except: pass
        logger.info(f"Archived stale reports to {archive_dir}")

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
        raw_encoder = SentenceTransformer(raw_model_name, device=str(device))
        raw_searcher = load_searcher(raw_index_dir, os.path.join(data_dir, "kb_chunks.jsonl"), raw_encoder)
        baseline_executor = BaselineExecutor(raw_searcher, generator=None, cfg=cfg)

        # --- 2. SETUP BASELINE-2 (RULE WORKFLOW) ---
        safety_cfg = cfg.get("safety_gate", {})
        ood_patterns = safety_cfg.get("ood_patterns", [])
        
        router = load_router(
            os.path.join(data_dir, "domain_centroids.json"),
            os.path.join(data_dir, "domain_keywords.json"),
            ood_patterns=ood_patterns
        )
        rule_executor = RuleWorkflowExecutor(raw_searcher, router, generator=None, cfg=cfg)

        # --- 3. SETUP PROPOSED (DOMAIN ROUTED) ---
        trained_retriever_path = os.path.join(output_dir, "retriever")
        if os.path.isdir(trained_retriever_path):
            trained_encoder = SentenceTransformer(trained_retriever_path, device=str(device))
        else:
            trained_encoder = raw_encoder
            
        global_index_dir = cfg.get("index_dir", "data/indexes")
        triage     = load_predictor(os.path.join(output_dir, "triage"), device=device)
        reranker   = load_reranker(os.path.join(output_dir, "reranker"), device=device)
        pref       = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
        
        gen_path   = os.path.join(output_dir, "generator")
        generator  = load_generator(gen_path if os.path.isdir(gen_path or "") else None, device=device, cfg=cfg)

        proposed_searcher = load_searcher(
            global_index_dir, 
            os.path.join(data_dir, "kb_chunks.jsonl"), 
            trained_encoder,
            domain_indexes_dir=domain_indexes_dir,
            global_domain_filter_multiplier=cfg.get("global_domain_filter_multiplier", 10)
        )
        
        kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
        chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

        proposed_executor = ToolExecutor(
            trained_encoder, proposed_searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg
        )

        # --- Print System Configurations ---
        logger.info("\n" + "="*50)
        logger.info("SYSTEM CONFIGURATIONS")
        logger.info("="*50)
        logger.info(f"Baseline-1: {raw_model_name} | {raw_index_dir} | No Router | No Triage | No Reranker | No Generator")
        logger.info(f"Baseline-2: {raw_model_name} | {raw_index_dir} | Router (Rule) | Rule Triage | No Reranker | No Generator")
        logger.info(f"Proposed:   {'Trained' if trained_encoder != raw_encoder else 'MiniLM'} | {domain_indexes_dir} | Router | {'Triage' if triage else 'No Triage'} | {'Reranker' if reranker else 'No Reranker'} | {'Generator' if generator else 'No Generator'}")
        logger.info("="*50 + "\n")

        # 4. Load eval data
        eval_set = read_jsonl(os.path.join(data_dir, eval_file))
        max_samples = cfg.get("max_eval_samples")
        if max_samples:
            eval_set = eval_set[:max_samples]
            
        # 5. Run E2E
        from src.evaluation.evaluate_end_to_end import run_e2e_eval

        logger.info(f"Evaluating systems on {len(eval_set)} samples...")
        base_m, base_res = run_e2e_eval(baseline_executor, eval_set, "Baseline-1", cfg=cfg)
        rule_m, rule_res = run_e2e_eval(rule_executor, eval_set, "Baseline-2", cfg=cfg)
        prop_m, prop_res = run_e2e_eval(proposed_executor, eval_set, "Proposed", cfg=cfg)

        # 6. Save Reports
        write_json(base_m, os.path.join(rep_dir, "baseline_metrics.json"))
        write_json(rule_m, os.path.join(rep_dir, "rule_workflow_baseline_metrics.json"))
        write_json(prop_m, os.path.join(rep_dir, "proposed_metrics.json"))
        
        write_jsonl(base_res, os.path.join(rep_dir, "baseline_results.jsonl"))
        write_jsonl(rule_res, os.path.join(rep_dir, "rule_workflow_baseline_results.jsonl"))
        write_jsonl(prop_res, os.path.join(rep_dir, "proposed_results.jsonl"))

        rows = [base_m, rule_m, prop_m]
        write_json(rows, os.path.join(rep_dir, "all_metrics.json"))
        
        keys = [
            "label", "EvidenceHit@5", "CitationDocPrecision", "CitationChunkPrecision",
            "GroundedAnswerRate", "UnsupportedAnswerRate", "WrongDomainCitationRate",
            "TriageAccuracy", "MacroF1", "WeightedF1", "FalseAcceptRateExplicit",
            "GoldAnswerAnsweredRate", "GoldAnswerTicketRate", "GoldAnswerRejectRate",
            "FinalAnswerRate", "FinalTicketRate", "FinalRejectRate",
            "PredictedAnswerCitationRate", "DomainAccuracy", "DomainRecall@2",
            "AvgLatencyMs", "AvgSearch_ms", "AvgFractionKBScanned", "NeuralGenRate", "REE@5"
        ]
        
        csv_path = os.path.join(rep_dir, "ablation_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info("\n--- METRIC SUMMARY ---")
        header = f"{'label':<12} | EvHit5 | WDDom  | TriAcc | MacF1  | GA_Ans | GA_Tkt | Neural% | LatMs | FracKB | REE@5"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r.get('label'):<12} | {r.get('EvidenceHit@5',0):.3f} | "
                f"{r.get('WrongDomainCitationRate',0):.3f} | "
                f"{r.get('TriageAccuracy',0):.3f} | {r.get('MacroF1',0):.3f} | "
                f"{r.get('GoldAnswerAnsweredRate',0):.3f} | "
                f"{r.get('GoldAnswerTicketRate',0):.3f} | "
                f"{r.get('NeuralGenRate',0)*100:>6.1f}% | "
                f"{r.get('AvgLatencyMs',0):.1f} | "
                f"{r.get('AvgFractionKBScanned',0):.3f} | {r.get('REE@5',0):.3f}")

        logger.info(f"Evaluation complete. Authoritative reports saved in {rep_dir}")

    except Exception as e:
        logger.error(f"FATAL ERROR in evaluation script: {e}")
        import traceback
        with open("critical_error.log", "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()