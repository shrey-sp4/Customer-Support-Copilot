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

DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_TBP_MU_VALUES = [0.10, 0.15, 0.20]


def run_e2e_eval(executor, eval_set, label: str = "system", max_samples: int = None, cfg: dict = None) -> dict:
    """
    Run end-to-end evaluation for a given executor (baseline or proposed).
    Returns aggregated metrics dict.
    """
    if max_samples:
        eval_set = eval_set[:max_samples]

    try:
        all_results      = []
        latencies_ms     = []
        latency_breakdowns = []
        triage_preds     = []
        triage_labels    = []
        triage_logits    = []
        routing_results  = []
        retrieval_results = []
        cluster_results  = []
        gen_meta_results = []

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
                
                # Extract logits from tool trace if present
                found_logits = [0.0, 0.0, 0.0]
                for t in result.get("tool_trace", []):
                    if t.get("tool") == "Triage":
                        found_logits = t["result"].get("logits", [0.0, 0.0, 0.0])
                        break
                triage_logits.append(found_logits)

                # Extract retrieved chunk IDs from tool trace.
                retrieved_ids = []
                pred_domain   = None
                pred_domains_top2 = []
                for trace in result.get("tool_trace", []):
                    if trace.get("tool") == "SearchKB":
                        passages = trace.get("result", {}).get("passages", [])
                        retrieved_ids.extend([p.get("chunk_id", "") for p in passages])
                    if trace.get("tool") == "RouteDomain":
                        domains = trace.get("result", {}).get("domains", [])
                        if domains:
                            pred_domain = domains[0].get("domain", "")
                            pred_domains_top2 = [d.get("domain", "") for d in domains[:2]]

                retrieval_results.append({
                    "query_id":            sample.get("query_id", ""),
                    "gold_chunk_id":       gold_cid,
                    "retrieved_chunk_ids": list(dict.fromkeys(retrieved_ids)),  # dedupe, preserve order
                })
                routing_results.append({
                    "predicted_domain": pred_domain,
                    "gold_domain":      gold_domain,
                    "predicted_domains_top2": pred_domains_top2,
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
                gen_meta_results.append(result.get("generation_metadata", {}))
                if (i+1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(eval_set)} samples...")
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise e

        # Aggregate metrics
        ret_metrics     = compute_retrieval_metrics(retrieval_results, top_k=DEFAULT_RETRIEVAL_TOP_K)
        triage_m        = compute_triage_metrics(
            triage_preds, 
            triage_labels, 
            triage_logits, 
            mu_values=DEFAULT_TBP_MU_VALUES
        )
        routing_m       = compute_routing_metrics(routing_results)
        latency_m       = compute_latency_metrics(latencies_ms)
        comp_latency_m  = compute_component_latency_metrics(latency_breakdowns)
        cluster_m       = compute_cluster_metrics(cluster_results)
        # Pass encoder for Neural Semantic Fidelity check
        encoder = getattr(executor, "encoder", None)
        quality_m       = compute_answer_quality_metrics(all_results, encoder=encoder, cfg=cfg)
        grounding_m     = grounded_answer_rate(all_results) # Final authority on grounding & decisions
        
        # Neural Gen Metrics
        neural_attempts = [m for m in gen_meta_results if m.get("source") != "unknown"]
        neural_successes = [m for m in neural_attempts if m.get("neural", False)]
        neural_gen_rate = len(neural_successes) / len(neural_attempts) if neural_attempts else 0.0

        # REE@5
        avg_fraction = cluster_m.get("AvgFractionKBScanned", 1.0)
        ree = compute_ree_at_k(ret_metrics.get("EvidenceHit@5", 0.0), fraction_kb_scanned=avg_fraction)

        all_metrics = {
            "label":    label,
            "n_eval":   len(eval_set),
            **ret_metrics,
            **routing_m,
            **latency_m,
            **comp_latency_m,
            **cluster_m,
            **quality_m,
            **triage_m,
            **grounding_m,
            "NeuralGenRate": neural_gen_rate,
            "REE@5":    ree,
        }
    except Exception as e:
        logger.error(f"Error aggregating metrics for {label}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        all_metrics = {"label": label, "error": str(e)}
        all_results = []
    
    return all_metrics, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    print("End-to-end evaluation should be called from scripts/evaluate.py")
