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
from typing import List, Dict
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
    """nDCG@k for binary relevance: gold=1, others=0."""
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid == gold_id:
            dcg = 1.0 / math.log2(rank + 1)
            break
    idcg = 1.0
    return dcg / idcg if idcg > 0 else 0.0


def evidence_hit_at_k(retrieved_ids: List[str], gold_id: str, k: int) -> float:
    return float(gold_id in retrieved_ids[:k])


def doc_hit_at_k(retrieved_ids: List[str], gold_id: str, k: int) -> float:
    """Check if the document (prefix before _chunk_) matches."""
    if not gold_id: return 0.0
    gold_doc = gold_id.split("_chunk_")[0] if "_chunk_" in gold_id else gold_id
    for rid in retrieved_ids[:k]:
        rdoc = rid.split("_chunk_")[0] if "_chunk_" in rid else rid
        if rdoc == gold_doc:
            return 1.0
    return 0.0


def compute_retrieval_metrics(eval_results: List[dict], top_k: int = 5) -> dict:
    """Compute retrieval metrics over a list of result dicts.

    Each result must have:
      - retrieved_chunk_ids: list[str]
      - gold_chunk_id: str
    """
    if not eval_results:
        return {}

    r1 = r3 = r5 = mrr = ndcg = eh1 = eh3 = eh5 = dh5 = 0.0
    n = len(eval_results)

    for r in eval_results:
        ids = r.get("retrieved_chunk_ids", [])
        gold = r.get("gold_chunk_id", "")

        r1 += recall_at_k(ids, gold, 1)
        r3 += recall_at_k(ids, gold, 3)
        r5 += recall_at_k(ids, gold, top_k)
        mrr += mrr_at_k(ids, gold, 10)
        ndcg += ndcg_at_k(ids, gold, 10)
        eh1 += evidence_hit_at_k(ids, gold, 1)
        eh3 += evidence_hit_at_k(ids, gold, 3)
        eh5 += evidence_hit_at_k(ids, gold, top_k)
        dh5 += doc_hit_at_k(ids, gold, top_k)

    return {
        "Recall@1": r1 / n,
        "Recall@3": r3 / n,
        "Recall@5": r5 / n,
        "MRR@10": mrr / n,
        "nDCG@10": ndcg / n,
        "EvidenceHit@1": eh1 / n,
        "EvidenceHit@3": eh3 / n,
        "EvidenceHit@5": eh5 / n,
        "EvidenceDocHit@5": dh5 / n,
    }


# ---------------------------------------------------------------------------
# Grounding metrics
# ---------------------------------------------------------------------------

# Supports citations like:
#   [doc_id:chunk_id]
#   [doc_id:chunk_id 0-250]
#   [doc_id]
CITATION_PATTERN = re.compile(r"\[([^:\]]+)(?::([^\]]+))?\]")


def has_citation(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer or ""))


def extract_citations_from_text(text: str) -> List[dict]:
    """Extract dicts of {doc_id, chunk_id, span} from answer text."""
    matches = CITATION_PATTERN.findall(text or "")
    results = []

    for m in matches:
        doc_id = m[0].strip()
        chunk_part = m[1].strip() if m[1] else None

        chunk_id = chunk_part
        span = None

        if chunk_part and " " in chunk_part:
            parts = chunk_part.rsplit(" ", 1)
            if len(parts) == 2 and re.match(r"\d+-\d+", parts[1]):
                chunk_id = parts[0].strip()
                span = parts[1].strip()

        results.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "span": span,
        })

    return results


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def citation_doc_precision(answer: str, gold_doc_id: str) -> float:
    """1 if any cited doc matches gold_doc_id exactly or by containment."""
    cits = extract_citations_from_text(answer)
    if not cits:
        return 0.0

    gold = _norm_text(gold_doc_id)
    if not gold:
        return 0.0

    for c in cits:
        cited = _norm_text(c.get("doc_id", ""))
        if cited == gold or gold in cited or cited in gold:
            return 1.0

    return 0.0


def citation_chunk_precision(answer: str, gold_chunk_id: str) -> float:
    """1 if any cited chunk matches gold_chunk_id.

    Returns 1.0 if gold_chunk_id is empty/missing because chunk-level labels
    may not always be available.
    """
    if not gold_chunk_id:
        return 1.0

    cits = extract_citations_from_text(answer)
    if not cits:
        return 0.0

    gold = _norm_text(gold_chunk_id)

    for c in cits:
        cited = _norm_text(c.get("chunk_id", ""))
        if cited == gold or gold in cited or cited in gold:
            return 1.0

    return 0.0


def citation_recall(answer: str, gold_doc_id: str) -> float:
    """1 if gold doc appears anywhere in answer text."""
    return float(_norm_text(gold_doc_id) in _norm_text(answer))


def _infer_domain_from_doc_id(doc_id: str) -> str:
    """Infer broad support domain from a citation doc_id."""
    d = _norm_text(doc_id)

    if any(x in d for x in ["dmv", "motor vehicle", "license", "driver", "non-driver"]):
        return "dmv"

    if any(x in d for x in ["ssa", "social security"]):
        return "ssa"

    if any(x in d for x in ["va", "veteran"]):
        return "va"

    if any(x in d for x in ["studentaid", "student aid", "fafsa", "federal student aid"]):
        return "studentaid"

    return "unknown"


def _normalize_gold_domain(domain: str) -> str:
    d = _norm_text(domain)

    if d in {"student_aid", "student-aid", "student aid", "federal student aid"}:
        return "studentaid"

    if d in {"dmv", "ssa", "va", "studentaid"}:
        return d

    return d


def wrong_domain_citation(answer: str, gold_domain: str) -> float:
    """1 if answer cites at least one source but none match the gold domain.

    If there are no citations, this returns 0.0 because unsupported/no-citation
    is already captured by UnsupportedAnswerRate.
    """
    cits = extract_citations_from_text(answer)
    if not cits:
        return 0.0

    gold = _normalize_gold_domain(gold_domain)
    if not gold:
        return 0.0

    cited_domains = {
        _infer_domain_from_doc_id(c.get("doc_id", ""))
        for c in cits
    }

    cited_domains.discard("unknown")

    if not cited_domains:
        return 0.0

    return float(gold not in cited_domains)


def grounded_answer_rate(all_samples: List[dict]) -> dict:
    """Compute citation, grounding, and final decision metrics across all samples.
    
    This function is the source of truth for:
    - Gold decision rates (how often we correctly answered/ticketed/rejected gold ANSWERs)
    - Final decision distributions
    - Citation precision/grounding over PRED=ANSWER
    - False Accept Rate (Explicit)
    """
    total = len(all_samples)
    if total == 0:
        return {}

    # 1. Decision Distributions (All Samples)
    n_pred_answer = sum(1 for r in all_samples if r.get("decision") == "ANSWER")
    n_pred_ticket = sum(1 for r in all_samples if r.get("decision") == "TICKET")
    n_pred_reject = sum(1 for r in all_samples if r.get("decision") == "REJECT")
    
    # 2. Decisions over Gold=ANSWER samples
    gold_answer_samples = [r for r in all_samples if r.get("gold_triage") == "ANSWER"]
    n_gold_answer = len(gold_answer_samples)
    
    ga_answered = sum(1 for r in gold_answer_samples if r.get("decision") == "ANSWER")
    ga_ticket   = sum(1 for r in gold_answer_samples if r.get("decision") == "TICKET")
    ga_reject   = sum(1 for r in gold_answer_samples if r.get("decision") == "REJECT")
    
    # 3. False Accepts (Pred=ANSWER but Gold != ANSWER)
    fa_count = sum(1 for r in all_samples if r.get("decision") == "ANSWER" and r.get("gold_triage") != "ANSWER")
    fa_rate_expl = fa_count / max(1, total - n_gold_answer)

    # 4. Citation & Domain metrics over PRED=ANSWER
    pred_answer_samples = [r for r in all_samples if r.get("decision") == "ANSWER"]
    n_pa = len(pred_answer_samples)
    
    doc_prec = 0.0
    chunk_prec = 0.0
    wrong_domain = 0.0
    unsupported = 0.0
    has_cit_count = 0.0
    
    for res in pred_answer_samples:
        ans = res.get("final_answer", "")
        gold_doc = res.get("gold_doc_id", "")
        gold_chunk = res.get("gold_chunk_id", "")
        gold_domain = res.get("gold_domain", "")
        
        cits = extract_citations_from_text(ans)
        if cits:
            has_cit_count += 1
            if any(c["doc_id"] == gold_doc for c in cits):
                doc_prec += 1
            if not gold_chunk or any(c["chunk_id"] == gold_chunk for c in cits):
                chunk_prec += 1
            if wrong_domain_citation(ans, gold_domain) > 0:
                wrong_domain += 1
        else:
            unsupported += 1

        if "I could not find enough evidence" in ans or "I could not generate" in ans:
            if cits: unsupported += 1

    return {
        "CitationDocPrecision": doc_prec / max(1, n_pa),
        "CitationChunkPrecision": chunk_prec / max(1, n_pa),
        "GroundedAnswerRate": (n_pa - unsupported) / max(1, total),
        "UnsupportedAnswerRate": unsupported / max(1, total),
        "WrongDomainCitationRate": wrong_domain / max(1, total),
        "GoldAnswerAnsweredRate": ga_answered / max(1, n_gold_answer),
        "GoldAnswerTicketRate": ga_ticket / max(1, n_gold_answer),
        "GoldAnswerRejectRate": ga_reject / max(1, n_gold_answer),
        "FinalAnswerRate": n_pred_answer / max(1, total),
        "FinalTicketRate": n_pred_ticket / max(1, total),
        "FinalRejectRate": n_pred_reject / max(1, total),
        "FalseAcceptCount": fa_count,
        "FalseAcceptRateExplicit": fa_rate_expl,
        "PredictedAnswerCitationRate": has_cit_count / max(1, n_pa),
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
    """Compute accuracy and F1 for triage predictions."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support
    from collections import Counter

    if not predictions:
        return {}

    acc = accuracy_score(labels, predictions)
    macro = f1_score(labels, predictions, average="macro", zero_division=0)
    weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    prec, rec, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=["ANSWER", "TICKET", "REJECT"],
        zero_division=0,
    )

    pred_dist = dict(Counter(predictions))
    gold_dist = dict(Counter(labels))

    conf = confusion_matrix(
        labels,
        predictions,
        labels=["ANSWER", "TICKET", "REJECT"],
    ).tolist()

    n_ans = labels.count("ANSWER")
    n_rej = labels.count("REJECT")
    n_tkt = labels.count("TICKET")

    false_reject = sum(
        1 for p, l in zip(predictions, labels)
        if p == "REJECT" and l == "ANSWER"
    )
    false_ticket = sum(
        1 for p, l in zip(predictions, labels)
        if p == "TICKET" and l == "ANSWER"
    )
    false_accept = sum(
        1 for p, l in zip(predictions, labels)
        if p == "ANSWER" and l != "ANSWER"
    )

    return {
        "TriageAccuracy": acc,
        "MacroF1": macro,
        "WeightedF1": weighted,

        "ANSWER_F1": float(f1[0]),
        "ANSWER_P": float(prec[0]),
        "ANSWER_R": float(rec[0]),

        "TICKET_F1": float(f1[1]),
        "TICKET_P": float(prec[1]),
        "TICKET_R": float(rec[1]),

        "REJECT_F1": float(f1[2]),
        "REJECT_P": float(prec[2]),
        "REJECT_R": float(rec[2]),

        "FalseRejectRate": false_reject / max(n_ans, 1),
        "FalseTicketRate": false_ticket / max(n_ans, 1),
        "FalseAcceptRate": false_accept / max(n_rej + n_tkt, 1),

        "LabelDistPred": pred_dist,
        "LabelDistGold": gold_dist,
        "ConfusionMatrix": conf,
    }


# ---------------------------------------------------------------------------
# Domain routing metrics
# ---------------------------------------------------------------------------

def compute_routing_metrics(results: List[dict]) -> dict:
    """Compute DomainAccuracy and DomainRecall@2."""
    if not results:
        return {}

    total = len(results)

    correct_top1 = sum(
        1 for r in results
        if r.get("predicted_domain") == r.get("gold_domain")
    )

    correct_top2 = sum(
        1 for r in results
        if r.get("gold_domain") in r.get(
            "predicted_domains_top2",
            [r.get("predicted_domain", "")],
        )
    )

    return {
        "DomainAccuracy": correct_top1 / total,
        "DomainRecall@2": correct_top2 / total,
    }


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def compute_latency_metrics(latencies_ms: List[float]) -> dict:
    if not latencies_ms:
        return {}

    arr = np.array(latencies_ms)
    n = len(arr)
    elapsed_s = arr.sum() / 1000.0

    return {
        "AvgLatencyMs": float(arr.mean()),
        "P50LatencyMs": float(np.percentile(arr, 50)),
        "P95LatencyMs": float(np.percentile(arr, 95)),
        "P99LatencyMs": float(np.percentile(arr, 99)),
        "QueriesPerSec": float(n / elapsed_s) if elapsed_s > 0 else 0.0,
        "TotalSamples": n,
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
        f"Avg{c.capitalize()}": totals[c] / n
        for c in components
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

def compute_ree_at_k(
    evidence_hit_k: float,
    fraction_kb_scanned: float,
    epsilon: float = 1e-6,
) -> float:
    """REE@k = EvidenceHit@k / max(FractionOfKBScanned, epsilon)."""
    denom = max(fraction_kb_scanned, epsilon)
    return evidence_hit_k / denom