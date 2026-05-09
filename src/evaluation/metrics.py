"""All evaluation metrics for the support copilot.

Implements:
  - Retrieval: Recall@k, MRR@10, nDCG@10, EvidenceHit@k
  - Grounding: CitationPrecision, CitationRecall, GroundedAnswerRate, UnsupportedClaimRate
  - Triage: Accuracy, Macro-F1, per-class F1, TBP@mu, FalseRejectRate, FalseAcceptRate
  - Routing: DomainAccuracy, DomainRecall@2
  - Latency: AvgLatencyMs, P95LatencyMs, QueriesPerSecond
  - Custom: TBP@mu, REE@k
"""
import re
import math
from typing import List, Dict, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], gold_id: str, k: int) -> float:
    return float(gold_id in retrieved_ids[:k])


def mrr_at_k(retrieved_ids: List[str], gold_id: str, k: int = 10) -> float:
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], gold_id: str, k: int = 10) -> float:
    """nDCG@k for binary relevance (gold=1, others=0)."""
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid == gold_id:
            dcg = 1.0 / math.log2(rank + 1)
            break
    idcg = 1.0  # ideal: gold at rank 1
    return dcg / idcg if idcg > 0 else 0.0


def evidence_hit_at_k(retrieved_ids: List[str], gold_id: str, k: int) -> float:
    return float(gold_id in retrieved_ids[:k])


def compute_retrieval_metrics(eval_results: List[dict], top_k: int = 5) -> dict:
    """Compute retrieval metrics over a list of result dicts.
    Each result must have 'retrieved_chunk_ids' (list) and 'gold_chunk_id' (str).
    """
    if not eval_results:
        return {}
    r1 = r5 = mrr = ndcg = eh5 = 0.0
    n = len(eval_results)
    for r in eval_results:
        ids      = r.get("retrieved_chunk_ids", [])
        gold     = r.get("gold_chunk_id", "")
        r1   += recall_at_k(ids, gold, 1)
        r5   += recall_at_k(ids, gold, top_k)
        mrr  += mrr_at_k(ids, gold, 10)
        ndcg += ndcg_at_k(ids, gold, 10)
        eh5  += evidence_hit_at_k(ids, gold, top_k)
    return {
        "Recall@1":      r1 / n,
        "Recall@5":      r5 / n,
        "MRR@10":        mrr / n,
        "nDCG@10":       ndcg / n,
        "EvidenceHit@5": eh5 / n,
    }


# ---------------------------------------------------------------------------
# Grounding metrics (heuristic-based)
# ---------------------------------------------------------------------------

CITATION_PATTERN = re.compile(r"\[([^:\]]+)(?::([^\]]+))?\]")


def has_citation(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer))


def extract_citations_from_text(text: str) -> List[dict]:
    """Extract dicts of {doc_id, chunk_id, span} from text."""
    matches = CITATION_PATTERN.findall(text)
    results = []
    for m in matches:
        doc_id = m[0].strip()
        chunk_part = m[1].strip() if m[1] else None
        
        chunk_id = chunk_part
        span = None
        if chunk_part and " " in chunk_part:
            # Check if the last part looks like a span (digits-digits)
            parts = chunk_part.rsplit(" ", 1)
            if re.match(r"\d+-\d+", parts[1]):
                chunk_id = parts[0].strip()
                span = parts[1].strip()

        results.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "span": span
        })
    return results


def citation_doc_precision(answer: str, gold_doc_id: str) -> float:
    """1 if any cited doc matches gold_doc_id, 0 otherwise."""
    cits = extract_citations_from_text(answer)
    if not cits:
        return 0.0
    gold = gold_doc_id.strip()
    return float(any(c["doc_id"] == gold for c in cits))


def citation_chunk_precision(answer: str, gold_chunk_id: str) -> float:
    """1 if any cited chunk matches gold_chunk_id, 0 otherwise. 
    Returns 1.0 if gold_chunk_id is empty/missing.
    """
    if not gold_chunk_id:
        return 1.0
    cits = extract_citations_from_text(answer)
    if not cits:
        return 0.0
    gold = gold_chunk_id.strip()
    return float(any(c["chunk_id"] == gold for c in cits))


def citation_recall(answer: str, gold_doc_id: str) -> float:
    """1 if gold doc is cited anywhere in answer."""
    return float(gold_doc_id.strip() in answer)


def grounded_answer_rate(answers: List[dict]) -> dict:
    """Compute citation and grounding metrics over ANSWER decisions."""
    answer_recs = [r for r in answers if r.get("gold_triage") == "ANSWER"]
    if not answer_recs:
        return {
            "CitationDocPrecision":   0.0,
            "CitationChunkPrecision": 0.0,
            "GroundedAnswerRate":     0.0,
        }
    n = len(answer_recs)
    cdp = ccp = gar = 0.0
    for r in answer_recs:
        ans   = r.get("final_answer", "")
        gold_doc   = r.get("gold_doc_id", "")
        gold_chunk = r.get("gold_chunk_id", "")
        
        cdp += citation_doc_precision(ans, gold_doc)
        ccp += citation_chunk_precision(ans, gold_chunk)
        gar += float(has_citation(ans))
        
    return {
        "CitationDocPrecision":    cdp / n,
        "CitationChunkPrecision":  ccp / n,
        "GroundedAnswerRate":      gar / n,
        "UnsupportedAnswerRate":   1.0 - (gar / n),
    }


# ---------------------------------------------------------------------------
# Triage metrics
# ---------------------------------------------------------------------------

def compute_triage_metrics(
    predictions: List[str],
    labels: List[str],
    logits_list: List[List[float]] = None,
    mu_values: List[float] = None,
) -> dict:
    """Compute accuracy, F1, TBP@mu for triage predictions."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from collections import Counter

    if not predictions:
        return {}

    acc   = accuracy_score(labels, predictions)
    macro = f1_score(labels, predictions, average="macro", zero_division=0)
    weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    per   = f1_score(labels, predictions, average=None, zero_division=0,
                     labels=["ANSWER", "TICKET", "REJECT"])

    # Label distribution
    pred_dist = dict(Counter(predictions))
    gold_dist = dict(Counter(labels))

    # Confusion matrix
    conf = confusion_matrix(labels, predictions, labels=["ANSWER", "TICKET", "REJECT"]).tolist()

    # False rates
    n_ans = labels.count("ANSWER")
    n_rej = labels.count("REJECT")
    n_tkt = labels.count("TICKET")
    
    false_reject = sum(1 for p, l in zip(predictions, labels) if p == "REJECT" and l == "ANSWER")
    false_ticket = sum(1 for p, l in zip(predictions, labels) if p == "TICKET" and l == "ANSWER")
    false_accept = sum(1 for p, l in zip(predictions, labels) if p == "ANSWER" and l != "ANSWER")

    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=["ANSWER", "TICKET", "REJECT"], zero_division=0
    )

    result = {
        "TriageAccuracy":   acc,
        "MacroF1":          macro,
        "WeightedF1":       weighted,
        "ANSWER_F1":        float(f1[0]),
        "ANSWER_P":         float(prec[0]),
        "ANSWER_R":         float(rec[0]),
        "TICKET_F1":        float(f1[1]),
        "TICKET_P":         float(prec[1]),
        "TICKET_R":         float(rec[1]),
        "REJECT_F1":        float(f1[2]),
        "REJECT_P":         float(prec[2]),
        "REJECT_R":         float(rec[2]),
        "FalseRejectRate":  false_reject / max(n_ans, 1),
        "FalseTicketRate":  false_ticket / max(n_ans, 1),
        "FalseAcceptRate":  false_accept / max(n_rej + n_tkt, 1),
        "LabelDistPred":    pred_dist,
        "LabelDistGold":    gold_dist,
        "ConfusionMatrix":  conf,
    }

    return result


# ---------------------------------------------------------------------------
# Domain routing metrics
# ---------------------------------------------------------------------------

def compute_routing_metrics(results: List[dict]) -> dict:
    """Compute DomainAccuracy, DomainRecall@2."""
    if not results:
        return {}
    total = len(results)
    correct_top1 = sum(1 for r in results if r.get("predicted_domain") == r.get("gold_domain"))
    correct_top2 = sum(
        1 for r in results
        if r.get("gold_domain") in r.get("predicted_domains_top2", [r.get("predicted_domain", "")])
    )
    return {
        "DomainAccuracy":  correct_top1 / total,
        "DomainRecall@2":  correct_top2 / total,
    }


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def compute_latency_metrics(latencies_ms: List[float]) -> dict:
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    n   = len(arr)
    elapsed_s = arr.sum() / 1000.0
    return {
        "AvgLatencyMs":   float(arr.mean()),
        "P50LatencyMs":   float(np.percentile(arr, 50)),
        "P95LatencyMs":   float(np.percentile(arr, 95)),
        "P99LatencyMs":   float(np.percentile(arr, 99)),
        "QueriesPerSec":  float(n / elapsed_s) if elapsed_s > 0 else 0.0,
        "TotalSamples":   n,
    }


def compute_component_latency_metrics(breakdowns: List[Dict[str, float]]) -> dict:
    """Compute average latency per component from a list of breakdown dicts."""
    if not breakdowns:
        return {}
    
    components = ["routing_ms", "search_ms", "rerank_ms", "gen_ms"]
    totals = {c: 0.0 for c in components}
    n = len(breakdowns)
    
    for b in breakdowns:
        for c in components:
            totals[c] += b.get(c, 0.0)
            
    return {
        f"Avg{c.capitalize()}": totals[c] / n for c in components
    }


# ---------------------------------------------------------------------------
# Cluster-gated metrics
# ---------------------------------------------------------------------------

def compute_cluster_metrics(results: List[dict]) -> dict:
    """Compute AvgClustersSearched and AvgFractionKBScanned."""
    if not results:
        return {}
    n = len(results)
    avg_n = sum(r.get("n_clusters", 0) for r in results) / n
    avg_f = sum(r.get("fraction_kb", 1.0) for r in results) / n
    return {
        "AvgClustersSearched": avg_n,
        "AvgFractionKBScanned": avg_f,
    }


# ---------------------------------------------------------------------------
# Custom metric: REE@k
# ---------------------------------------------------------------------------

def compute_ree_at_k(evidence_hit_k: float, fraction_kb_scanned: float, epsilon: float = 1e-6) -> float:
    """REE@k = EvidenceHit@k / max(FractionOfKBScanned, epsilon)"""
    denom = max(fraction_kb_scanned, epsilon)
    return evidence_hit_k / denom
