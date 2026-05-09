"""End-to-end evaluation for baseline and proposed systems."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_json, write_jsonl
from src.evaluation.metrics import (
    compute_retrieval_metrics,
    grounded_answer_rate,
    compute_triage_metrics,
    compute_routing_metrics,
    compute_latency_metrics,
    compute_component_latency_metrics,
    compute_ree_at_k,
    compute_cluster_metrics,
)
from src.evaluation.quality import compute_answer_quality_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_e2e_eval(executor, eval_set, label: str = "system", max_samples: int = None) -> dict:
    """
    Run end-to-end evaluation for a given executor (baseline or proposed).
    Returns aggregated metrics dict.
    """
    if max_samples:
        eval_set = eval_set[:max_samples]

    all_results      = []
    latencies_ms     = []
    latency_breakdowns = []
    triage_preds     = []
    triage_labels    = []
    triage_logits    = []
    routing_results  = []
    retrieval_results = []
    cluster_results  = []

    for i, sample in enumerate(eval_set):
        try:
            query    = sample.get("query", "")
            history  = sample.get("history", "")
            gold_triage = sample.get("gold_triage", "ANSWER")
            gold_cid    = sample.get("gold_chunk_id", "")
            gold_domain = sample.get("gold_domain", "")

            result = executor.run(query, history)

            latencies_ms.append(result.get("latency_ms", 0.0))
            latency_breakdowns.append(result.get("latency_breakdown", {}))
            triage_preds.append(result.get("decision", "ANSWER"))
            triage_labels.append(gold_triage)
            triage_logits.append(
                next(
                    (t["result"].get("logits", [0.0, 0.0, 0.0])
                     for t in result.get("tool_trace", [])
                     if t.get("tool") == "Triage"),
                    [0.0, 0.0, 0.0],
                )
            )

            # Extract retrieved chunk IDs from tool trace.
            # NOTE: For Baselines, the executor explicitly adds a 'SearchKB' tool trace 
            # with retrieved passages to ensure fair 'EvidenceHit@5' evaluation.
            retrieved_ids = []
            pred_domain   = None
            for trace in result.get("tool_trace", []):
                if trace.get("tool") == "SearchKB":
                    passages = trace.get("result", {}).get("passages", [])
                    retrieved_ids.extend([p.get("chunk_id", "") for p in passages])
                if trace.get("tool") == "RouteDomain":
                    domains = trace.get("result", {}).get("domains", [])
                    if domains:
                        pred_domain = domains[0].get("domain", "")

            retrieval_results.append({
                "query_id":            sample.get("query_id", ""),
                "gold_chunk_id":       gold_cid,
                "retrieved_chunk_ids": list(dict.fromkeys(retrieved_ids)),  # dedupe, preserve order
            })
            routing_results.append({
                "predicted_domain": pred_domain,
                "gold_domain":      gold_domain,
                "predicted_domains_top2": [pred_domain] if pred_domain else [],
            })

            all_results.append({
                **sample,
                "final_answer": result.get("final_answer", ""),
                "citations":    result.get("citations", []),
                "decision":     result.get("decision", "ANSWER"),
            })
            cluster_results.append({
                "n_clusters":   result.get("n_clusters", 1),
                "fraction_kb":  result.get("fraction_kb", 1.0),
            })
            if (i+1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(eval_set)} samples...")
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise e

    # Aggregate metrics
    try:
        ret_metrics     = compute_retrieval_metrics(retrieval_results, top_k=5)
        grounding_m     = grounded_answer_rate(all_results)
        triage_m        = compute_triage_metrics(triage_preds, triage_labels, triage_logits, mu_values=[0.10, 0.15, 0.20])
        routing_m       = compute_routing_metrics(routing_results)
        latency_m       = compute_latency_metrics(latencies_ms)
        comp_latency_m  = compute_component_latency_metrics(latency_breakdowns)
        cluster_m       = compute_cluster_metrics(cluster_results)
        quality_m       = compute_answer_quality_metrics(all_results)

        # REE@5
        avg_fraction = cluster_m.get("AvgFractionKBScanned", 1.0)
        ree = compute_ree_at_k(ret_metrics.get("EvidenceHit@5", 0.0), fraction_kb_scanned=avg_fraction)

        all_metrics = {
            "label":    label,
            "n_eval":   len(eval_set),
            **ret_metrics,
            **grounding_m,
            **triage_m,
            **routing_m,
            **latency_m,
            **comp_latency_m,
            **cluster_m,
            **quality_m,
            "REE@5":    ree,
        }
    except Exception as e:
        logger.error(f"Error aggregating metrics for {label}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        all_metrics = {"label": label, "error": str(e)}
    
    return all_metrics, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    print("End-to-end evaluation should be called from scripts/evaluate.py")
