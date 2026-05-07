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

CITATION_PATTERN = re.compile(r"\[doc_id=[^\]]+\]")


def has_citation(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer))


def extract_cited_doc_id(answer: str) -> Optional[str]:
    m = re.search(r"doc_id=([^,\]]+)", answer)
    return m.group(1).strip() if m else None


def citation_precision(answer: str, gold_doc_id: str) -> float:
    """1 if cited doc matches gold, 0 otherwise. 0 if no citation."""
    cited = extract_cited_doc_id(answer)
    if cited is None:
        return 0.0
    return float(cited.strip() == gold_doc_id.strip())


def citation_recall(answer: str, gold_doc_id: str) -> float:
    """1 if gold doc is cited anywhere in answer."""
    return float(gold_doc_id in answer)


def grounded_answer_rate(answers: List[dict]) -> dict:
    """Compute citation and grounding metrics over ANSWER decisions."""
    answer_recs = [r for r in answers if r.get("gold_triage") == "ANSWER"]
    if not answer_recs:
        return {
            "CitationPrecision":  0.0,
            "CitationRecall":     0.0,
            "GroundedAnswerRate": 0.0,
            "UnsupportedClaimRate": 0.0,
        }
    n = len(answer_recs)
    cp = cr = gar = ucr = 0.0
    for r in answer_recs:
        ans   = r.get("final_answer", "")
        gold  = r.get("gold_doc_id", "")
        cp  += citation_precision(ans, gold)
        cr  += citation_recall(ans, gold)
        gar += float(has_citation(ans))
        ucr += float(not has_citation(ans))
    return {
        "CitationPrecision":    cp / n,
        "CitationRecall":       cr / n,
        "GroundedAnswerRate":   gar / n,
        "UnsupportedClaimRate": ucr / n,
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
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from scipy.special import softmax as scipy_softmax

    if not predictions:
        return {}

    acc   = accuracy_score(labels, predictions)
    macro = f1_score(labels, predictions, average="macro", zero_division=0)
    per   = f1_score(labels, predictions, average=None, zero_division=0,
                     labels=["ANSWER", "TICKET", "REJECT"])

    n_ans = labels.count("ANSWER")
    n_all = len(labels)

    # False reject rate: REJECT when should be ANSWER
    false_reject = sum(1 for p, l in zip(predictions, labels) if p == "REJECT" and l == "ANSWER")
    frr = false_reject / max(n_ans, 1)

    # False accept rate: ANSWER when should be REJECT
    n_rej = labels.count("REJECT")
    false_accept = sum(1 for p, l in zip(predictions, labels) if p == "ANSWER" and l == "REJECT")
    far = false_accept / max(n_rej, 1)

    result = {
        "TriageAccuracy": acc,
        "MacroF1":        macro,
        "ANSWER_F1":      float(per[0]) if len(per) > 0 else 0.0,
        "TICKET_F1":      float(per[1]) if len(per) > 1 else 0.0,
        "REJECT_F1":      float(per[2]) if len(per) > 2 else 0.0,
        "FalseRejectRate": frr,
        "FalseAcceptRate": far,
    }

    # TBP@mu
    if logits_list is not None:
        for mu in (mu_values or [0.10, 0.15, 0.20]):
            tbp = 0
            for logits, pred, label in zip(logits_list, predictions, labels):
                if pred != label:
                    continue
                p = scipy_softmax(logits)
                p_sorted = sorted(p, reverse=True)
                margin = p_sorted[0] - p_sorted[1]
                if margin >= mu:
                    tbp += 1
            result[f"TBP@{mu:.2f}"] = tbp / max(len(predictions), 1)

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


# ---------------------------------------------------------------------------
# Custom metric: REE@k
# ---------------------------------------------------------------------------

def compute_ree_at_k(evidence_hit_k: float, fraction_kb_scanned: float) -> float:
    """REE@k = EvidenceHit@k / FractionOfKBScanned"""
    if fraction_kb_scanned <= 0:
        return 0.0
    return evidence_hit_k / fraction_kb_scanned
