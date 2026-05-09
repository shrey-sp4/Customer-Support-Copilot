"""Tool executor: orchestrates the full proposed pipeline.

Pipeline:
  1. RouteDomain using centroid + lexical gate
  2. SearchKB using routed domain indexes
  3. Evidence validation and grounding guard
  4. Triage decision: ANSWER / TICKET / REJECT
  5. Generate cited answer or create ticket/reject
"""

import re
import time
from typing import Dict, List

from src.tools.tools import route_domain, search_kb, get_policy, create_ticket, reject_query
from src.generation.generate import generate_answer, template_answer
from src.generation.templates import format_reject_response, format_ticket_response
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vague / generic query detector
# ---------------------------------------------------------------------------
_VAGUE_PATTERN = re.compile(
    r"^\s*("
    r"why am i here|what is this|help me|what should i do|tell me something"
    r"|explain this|who are you|are you real|what do you do|what can you do"
    r"|what is your purpose|i don.t know|idk|huh|ok|okay|thanks|thank you"
    r"|hi+|hey+|hello+|yes|no|sure|maybe|alright|got it|i see"
    r")\s*[.?!]*\s*$",
    re.IGNORECASE,
)


def is_vague_query(query: str) -> bool:
    """Return True if query has no clear support-domain intent."""
    q = query.strip()

    if _VAGUE_PATTERN.match(q):
        return True

    tokens = [t for t in q.lower().split() if len(t) > 2]
    if len(tokens) <= 2 and len(q) < 30:
        return True

    return False


# ---------------------------------------------------------------------------
# Personal / action request detector
# ---------------------------------------------------------------------------
_PERSONAL_ACTION_PATTERNS = [
    r"\b(check|view|see|status of|track|where is)\b.*\b(my|mine|i have)\b.*\b(status|application|case|claim|payment|benefit|account|order|file|form)\b",
    r"\b(update|change|reset|edit|modify)\b.*\b(my|mine)\b.*\b(info|information|address|phone|email|bank|direct deposit|account)\b",
    r"\b(submit|send|file|upload)\b.*\b(my|the)\b.*\b(form|document|file|application)\b.*\b(for me|on my behalf)\b",
    r"\b(guarantee|promise|assure)\b.*\b(approval|acceptance|eligibility|success)\b",
    r"\b(my|exact|specific)\b.*\b(amount|payment|benefit|check|balance)\b",
    r"\b(why|reason)\b.*\b(my|mine|i haven't)\b.*\b(delayed|late|not received|not arrived|missing)\b",
    r"\b(decide|decision|judge|approve)\b.*\b(my|this)\b.*\b(case|claim|application|request)\b",
    r"\b(private|personal|your|my)\b.*\b(phone|number|cell|mobile|address|email)\b",
]

_PERSONAL_ACTION_REGEX = re.compile("|".join(_PERSONAL_ACTION_PATTERNS), re.IGNORECASE)


def is_personal_or_action_request(query: str) -> bool:
    """Detect whether the query asks for private/account-specific action."""
    return bool(_PERSONAL_ACTION_REGEX.search(query or ""))


# ---------------------------------------------------------------------------
# Evidence/domain guards
# ---------------------------------------------------------------------------
def infer_passage_domain(p: dict) -> str:
    """Infer passage domain from metadata/doc_id when domain field is missing."""
    p_domain = (p.get("domain") or "").lower().strip()
    if p_domain:
        return p_domain

    p_doc_id = (p.get("doc_id") or "").lower()

    if "student aid" in p_doc_id or "studentaid" in p_doc_id or "fafsa" in p_doc_id:
        return "studentaid"
    if "social security" in p_doc_id or "ssa" in p_doc_id:
        return "ssa"
    if "veteran" in p_doc_id or "va" in p_doc_id:
        return "va"
    if "dmv" in p_doc_id or "motor vehicle" in p_doc_id or "license" in p_doc_id:
        return "dmv"

    return "unknown"


def has_wrong_intent(query: str, p: dict) -> bool:
    """Block passages that match the broad domain but answer a different intent."""
    q = (query or "").lower()
    text = ((p.get("doc_id") or "") + " " + (p.get("text") or "")).lower()

    asks_driver_license = (
        "driver license" in q
        or "driver's license" in q
        or ("driver" in q and "license" in q)
    )

    if asks_driver_license and "non-driver" in text:
        return True

    if "license" in q and "id card" in text and "driver" not in text:
        return True

    return False


def filter_grounded_evidence(
    query: str,
    passages: List[dict],
    selected_domains: List[str],
    min_overlap: int = 2,
) -> List[dict]:
    """
    Keep only passages that are in the selected domain and directly overlap the query.
    This prevents wrong-domain/wrong-intent citations.
    """
    q_tokens = set(re.findall(r"\b\w+\b", (query or "").lower()))
    stop_words = {
        "how", "to", "the", "a", "an", "do", "i", "need", "have", "is", "for",
        "if", "what", "can", "you", "my", "of", "who", "when", "where", "why",
        "me", "please", "tell", "about",
    }
    content_tokens = {t for t in q_tokens if t not in stop_words and len(t) > 2}

    selected = {d.lower() for d in selected_domains if d}
    kept = []

    for p in passages:
        p_domain = infer_passage_domain(p)
        p_text = ((p.get("doc_id") or "") + " " + (p.get("text") or "")).lower()

        overlap = sum(
            1
            for t in content_tokens
            if t in p_text or t.rstrip("s") in p_text
        )

        wrong_domain = bool(selected) and p_domain != "unknown" and p_domain not in selected
        wrong_intent = has_wrong_intent(query, p)

        p["citation_domain"] = p_domain
        p["citation_overlap"] = overlap
        p["wrong_domain"] = wrong_domain
        p["wrong_intent"] = wrong_intent

        if wrong_domain or wrong_intent:
            continue

        if overlap >= min_overlap or p.get("score", 0.0) >= 0.80:
            kept.append(p)

    return kept


def validate_answerability(
    query: str,
    final_evidence: List[dict],
    selected_domains: List[str],
) -> dict:
    """Check if evidence directly addresses the query and is coherent."""
    if not final_evidence:
        return {
            "answerable": False,
            "reason": "No evidence found",
            "coherence_score": 0.0,
            "best_evidence": [],
        }

    query_tokens = set(re.findall(r"\b\w+\b", (query or "").lower()))

    stop_words = {
        "how", "to", "the", "a", "an", "do", "i", "need", "have", "is",
        "for", "if", "what", "can", "you", "my", "of", "who", "when",
        "where", "why", "me", "please",
    }
    content_tokens = {t for t in query_tokens if t not in stop_words and len(t) > 2}

    action_terms = {
        "renew", "appli", "apply", "updat", "check", "submit", "eligibil",
        "document", "contact", "status", "registr", "enroll", "file",
    }
    query_actions = {a for a in action_terms if any(a in qt for qt in query_tokens)}

    best_p = final_evidence[0]
    p_text = (best_p.get("text") or "").lower()
    p_tokens = set(re.findall(r"\b\w+\b", p_text))

    overlap_terms = [
        t for t in content_tokens
        if t in p_text or t.rstrip("s") in p_text
    ]
    overlap = len(overlap_terms)

    action_match = any(a in p_text for a in query_actions) if query_actions else True

    p_domain = infer_passage_domain(best_p)
    selected = {d.lower() for d in selected_domains if d}
    domain_match = p_domain in selected if selected else p_domain != "unknown"

    coherence_score = 1.0
    if len(final_evidence) > 1:
        p2 = final_evidence[1]
        p2_text = (p2.get("text") or "").lower()
        p2_tokens = set(re.findall(r"\b\w+\b", p2_text))
        p2_domain = infer_passage_domain(p2)

        if p_domain != p2_domain:
            coherence_score = 0.4
        elif len(p2_tokens.intersection(p_tokens)) < 3:
            coherence_score = 0.6

    logger.info(
        f"[Validate] Overlap terms: {overlap_terms}, "
        f"Actions: {query_actions}, "
        f"ActionMatch: {action_match}, "
        f"Score: {best_p.get('score', 0.0):.4f}, "
        f"DomainMatch: {domain_match}"
    )

    wrong_intent = has_wrong_intent(query, best_p)

    score = best_p.get("score", 0.0)
    strong_overlap = overlap >= 2
    strong_score = score >= 0.55
    borderline_but_domain_matched = domain_match and score >= 0.45 and strong_overlap

    is_answerable = (
        domain_match
        and not wrong_intent
        and action_match
        and (strong_score or borderline_but_domain_matched)
    )

    if not domain_match:
        reason = (
            f"Evidence is from wrong domain: "
            f"selected={selected_domains}, passage_domain={p_domain}"
        )
        is_answerable = False
    elif wrong_intent:
        reason = "Evidence matches broad domain but wrong intent"
        is_answerable = False
    elif not strong_overlap and score < 0.80:
        reason = f"Evidence has weak query overlap: overlap={overlap}, terms={overlap_terms}"
        is_answerable = False
    elif not action_match:
        reason = f"Evidence does not match query action: actions={query_actions}"
        is_answerable = False
    elif coherence_score < 0.5:
        reason = "Multiple evidence chunks are incoherent/mixed domains"
        is_answerable = False
    else:
        reason = "Validated"

    return {
        "answerable": is_answerable,
        "reason": reason,
        "best_evidence": (
            final_evidence[:2]
            if is_answerable
            else []
        ),
        "coherence_score": coherence_score,
    }


class ToolExecutor:
    """Stateless executor that runs the full proposed pipeline."""

    def __init__(
        self,
        encoder,
        searcher,
        router,
        triage_predictor,
        reranker=None,
        generator=None,
        preference_scorer=None,
        chunk_by_id: Dict[str, dict] = None,
        cfg=None,
    ):
        self.encoder = encoder
        self.searcher = searcher
        self.router = router
        self.triage = triage_predictor
        self.reranker = reranker
        self.generator = generator
        self.preference_scorer = preference_scorer
        self.chunk_by_id = chunk_by_id or {}
        self.cfg = cfg

        self.generator_mode = "llm" if generator is not None else "template"

        self.top_k_retrieval = getattr(cfg, "top_k_retrieval", 10)
        self.top_k_rerank = getattr(cfg, "top_k_rerank", 4)
        self.top_k_domains = getattr(cfg, "top_k_domains", 2)

        self.tau_domain = getattr(cfg, "tau_domain", 0.35)
        self.tau_chunk = getattr(cfg, "tau_chunk", 0.40)

        self.evidence_answer_threshold = getattr(cfg, "evidence_answer_threshold", 0.40)
        self.evidence_ticket_threshold = getattr(cfg, "evidence_ticket_threshold", 0.10)

        self.tau_hard_reject = getattr(cfg, "tau_hard_reject", 0.08)
        self.tau_soft_domain = getattr(cfg, "tau_soft_domain", 0.20)

        self.cluster_out_of_domain_threshold = getattr(cfg, "cluster_out_of_domain_threshold", 0.20)
        self.cluster_confident_margin = getattr(cfg, "cluster_confident_margin", 0.10)
        self.cluster_ambiguous_top_k = getattr(cfg, "cluster_ambiguous_top_k", self.top_k_domains)
        self.max_clusters_for_confident_query = getattr(cfg, "max_clusters_for_confident_query", 1)
        self.max_clusters_for_ambiguous_query = getattr(
            cfg,
            "max_clusters_for_ambiguous_query",
            self.top_k_domains,
        )

    def run(self, query: str, history: str = "") -> dict:
        """Execute cluster-gated pipeline and return structured result."""
        t_start = time.time()
        tool_trace = []

        latency_breakdown = {
            "routing_ms": 0.0,
            "search_ms": 0.0,
            "rerank_ms": 0.0,
            "gen_ms": 0.0,
        }

        final_answer = ""
        citations = []
        confidence = 0.0
        triage_method = "unknown"

        # ------------------------------------------------------------------
        # Step 1: Encode query and RouteDomain
        # ------------------------------------------------------------------
        t_route_start = time.time()
        query_embedding = self.searcher.get_query_embedding(query)

        route_result = route_domain(
            query,
            query_embedding,
            self.router,
            top_k_domains=max(self.top_k_domains, self.max_clusters_for_ambiguous_query),
            tau_domain=self.tau_domain,
        )

        latency_breakdown["routing_ms"] = (time.time() - t_route_start) * 1000
        tool_trace.append(route_result)

        top_sim = route_result.get("top_centroid_sim", 0.0)
        margin = route_result.get("centroid_margin", 0.0)
        gate_result = route_result.get("gate_result", "pass")
        domain_results = route_result.get("result", {}).get("domains", [])

        # ------------------------------------------------------------------
        # Step 2: Domain gate
        # ------------------------------------------------------------------
        selected_domains = []
        gating_status = "unknown"
        domain_relevant = True

        matched_kws_by_domain = route_result.get("result", {}).get("matched_kws_by_domain", {})
        all_support_kws = route_result.get("support_keywords", [])
        kw_count = len(all_support_kws)

        strong_intent_domains = [d for d, kws in matched_kws_by_domain.items() if kws]

        if top_sim < self.tau_hard_reject and kw_count == 0:
            decision = "REJECT"
            gating_status = "out_of_domain"
            domain_relevant = False

        elif top_sim < self.tau_soft_domain and kw_count == 0:
            if is_vague_query(query):
                decision = "REJECT"
                gating_status = "vague_out_of_domain"
                domain_relevant = False
            else:
                decision = "route"
                selected_domains = [d["domain"] for d in domain_results[:2]]
                gating_status = "uncertain"
                domain_relevant = True

        elif strong_intent_domains:
            decision = "route"
            selected_domains = strong_intent_domains[:]

            if domain_results and domain_results[0]["domain"] not in selected_domains:
                selected_domains.append(domain_results[0]["domain"])

            selected_domains = selected_domains[:2]
            gating_status = "keyword_intent"
            domain_relevant = True

        elif margin >= self.cluster_confident_margin:
            decision = "route"
            selected_domains = [domain_results[0]["domain"]] if domain_results else []
            gating_status = "confident"
            domain_relevant = True

        else:
            decision = "route"
            selected_domains = [
                d["domain"]
                for d in domain_results[:self.max_clusters_for_ambiguous_query]
            ]
            gating_status = "ambiguous"
            domain_relevant = True

        # ------------------------------------------------------------------
        # Step 3: Retrieval and evidence validation
        # ------------------------------------------------------------------
        passages = []
        final_evidence = []
        best_evidence_score = 0.0

        if not domain_relevant:
            logger.info("[Gate] Query is out-of-domain. REJECT decision finalized.")

        else:
            if selected_domains:
                t_search_start = time.time()

                kb_result = search_kb(
                    query,
                    self.searcher,
                    top_k=self.top_k_retrieval,
                    domain=selected_domains,
                )

                latency_breakdown["search_ms"] = (time.time() - t_search_start) * 1000
                tool_trace.append(kb_result)

                passages = kb_result.get("result", {}).get("passages", [])

            if passages:
                query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
                action_terms = {
                    "renew", "apply", "update", "check", "submit", "eligibility",
                    "documents", "contact", "status", "enroll", "file",
                }
                query_actions = query_tokens.intersection(action_terms)

                # Domain-aware reranking score adjustment.
                for p in passages:
                    base_score = p.get("score", 0.0)
                    p_text = (p.get("text") or "").lower()
                    p_doc_id = (p.get("doc_id") or "").lower()
                    p_domain = infer_passage_domain(p)

                    p_tokens = set(re.findall(r"\b\w+\b", p_text))
                    overlap = len(query_tokens.intersection(p_tokens))

                    domain_bonus = 0.0
                    if p_domain in strong_intent_domains:
                        domain_bonus = 0.25
                    elif p_domain in selected_domains:
                        domain_bonus = 0.05

                    action_bonus = 0.20 if any(a in p_text for a in query_actions) else 0.0

                    domain_penalty = 0.0
                    if strong_intent_domains and p_domain not in strong_intent_domains:
                        domain_penalty = 0.40

                    query_l = query.lower()
                    doc_text_l = (p_doc_id + " " + p_text).lower()

                    intent_penalty = 0.0

                    if (
                        "driver license" in query_l
                        or "driver's license" in query_l
                        or ("driver" in query_l and "license" in query_l)
                    ) and "non-driver" in doc_text_l:
                        intent_penalty += 0.75

                    if "license" in query_l and "id card" in doc_text_l and "driver" not in doc_text_l:
                        intent_penalty += 0.50

                    p["score"] = base_score + domain_bonus + action_bonus - domain_penalty - intent_penalty
                    p["overlap"] = overlap
                    p["intent_penalty"] = intent_penalty

                if self.reranker:
                    t_rr_start = time.time()
                    reranked = self.reranker.rerank(
                        query,
                        passages,
                        top_k=self.top_k_retrieval,
                    )
                    latency_breakdown["rerank_ms"] = (time.time() - t_rr_start) * 1000

                    for p in reranked:
                        p_domain = infer_passage_domain(p)
                        if p_domain in strong_intent_domains:
                            p["score"] = p.get("score", 0.0) + 0.15

                    passages = reranked

                # Strict relevance + grounding guard.
                guarded_passages = filter_grounded_evidence(
                    query=query,
                    passages=passages,
                    selected_domains=selected_domains,
                    min_overlap=2,
                )

                for p in guarded_passages:
                    score = p.get("score", 0.0)
                    overlap = p.get("citation_overlap", p.get("overlap", 0))

                    is_relevant = (
                        score >= self.evidence_answer_threshold
                        and overlap >= 2
                        and not p.get("wrong_domain", False)
                        and not p.get("wrong_intent", False)
                    )

                    high_confidence_relevant = (
                        score >= 0.80
                        and overlap >= 1
                        and not p.get("wrong_domain", False)
                        and not p.get("wrong_intent", False)
                    )

                    if is_relevant or high_confidence_relevant:
                        final_evidence.append(p)

                    if len(final_evidence) >= self.top_k_rerank:
                        break

                final_evidence.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                val_res = validate_answerability(query, final_evidence, selected_domains)
                final_evidence = val_res["best_evidence"]
                final_evidence.sort(key=lambda p: p.get("score", 0.0), reverse=True)

                best_evidence_score = (
                    final_evidence[0].get("score", 0.0)
                    if final_evidence and val_res["answerable"]
                    else 0.0
                )

        # ------------------------------------------------------------------
        # Step 4: Authoritative triage
        # ------------------------------------------------------------------
        triage_pred = None
        triage_conf = 0.0
        triage_logits = [0.0, 0.0, 0.0]

        if self.triage is not None:
            triage_result = self.triage.predict(
                query=query,
                keyword_gate=gate_result,
                centroid_domain=selected_domains[0] if selected_domains else "unknown",
                centroid_sim_top1=top_sim,
                centroid_margin=margin,
                nearest_chunk_sim=best_evidence_score,
                retrieval_score_gap=0.0,
                history=history,
                tau_domain=self.tau_domain,
                tau_chunk=self.tau_chunk,
            )

            triage_pred = triage_result.get("label") or triage_result.get("prediction")
            triage_conf = triage_result.get("confidence", 0.0)
            triage_logits = triage_result.get("logits", [0.0, 0.0, 0.0])

        has_valid_evidence = (
            bool(final_evidence)
            and best_evidence_score >= self.evidence_answer_threshold
        )
        personal_action = is_personal_or_action_request(query)

        if not domain_relevant:
            decision = "REJECT"
            confidence = max(triage_conf, 1.0 - top_sim)
            triage_method = "domain_gate_reject"

        elif personal_action:
            decision = "TICKET"
            confidence = max(triage_conf, 0.80)
            triage_method = "personal_action_ticket"

        elif not has_valid_evidence:
            decision = "TICKET" if selected_domains else "REJECT"
            confidence = max(triage_conf, 0.70)
            triage_method = "insufficient_evidence"

        elif triage_pred in {"REJECT", "TICKET"} and triage_conf >= 0.55:
            decision = triage_pred
            confidence = triage_conf
            triage_method = "triage_model_authoritative"

        elif has_valid_evidence:
            decision = "ANSWER"
            confidence = max(triage_conf, min(1.0, best_evidence_score))
            triage_method = "validated_evidence_answer"

        else:
            decision = "TICKET"
            confidence = max(triage_conf, 0.50)
            triage_method = "safe_fallback_ticket"

        tool_trace.append({
            "tool": "ClusterGating",
            "args": {
                "top_sim": top_sim,
                "margin": margin,
                "selected_domains": selected_domains,
                "gating_status": gating_status,
            },
            "result": {
                "best_evidence_score": best_evidence_score,
                "decision": decision,
                "confidence": confidence,
                "triage_pred": triage_pred,
                "triage_conf": triage_conf,
                "triage_logits": triage_logits,
                "triage_method": triage_method,
            },
        })

        # ------------------------------------------------------------------
        # Step 5: Execute final action
        # ------------------------------------------------------------------
        if decision == "REJECT":
            rej_result = reject_query(
                reason="out_of_domain",
                nearest_kb_distance=1.0 - best_evidence_score,
                nearest_centroid_distance=1.0 - top_sim,
                confidence=confidence,
            )
            tool_trace.append(rej_result)
            final_answer = format_reject_response()
            citations = []

        elif decision == "TICKET":
            tkt_result = create_ticket(
                summary=query,
                category=selected_domains[0] if selected_domains else "general",
                severity="medium",
            )
            tool_trace.append(tkt_result)
            final_answer = format_ticket_response(tkt_result["result"]["ticket_id"], query)
            citations = []

        else:
            # Final citation safety check before generation.
            final_evidence = filter_grounded_evidence(
                query=query,
                passages=final_evidence,
                selected_domains=selected_domains,
                min_overlap=2,
            )

            if not final_evidence:
                decision = "TICKET"
                tkt_result = create_ticket(
                    summary=query,
                    category=selected_domains[0] if selected_domains else "general",
                    severity="medium",
                )
                tool_trace.append(tkt_result)
                final_answer = format_ticket_response(tkt_result["result"]["ticket_id"], query)
                citations = []

            else:
                pol_res = get_policy(
                    final_evidence[0]["doc_id"],
                    final_evidence[0]["section_id"],
                    self.chunk_by_id,
                )
                tool_trace.append(pol_res)

                t_gen_start = time.time()
                final_answer, citations, is_insufficient = generate_answer(
                    query=query,
                    passages=final_evidence,
                    generator=self.generator,
                    preference_scorer=self.preference_scorer,
                )
                latency_breakdown["gen_ms"] = (time.time() - t_gen_start) * 1000

                if is_insufficient:
                    logger.warning(
                        "[Executor] Generator failed despite validated evidence; "
                        "falling back to template answer."
                    )
                    final_answer, citations, is_insufficient = template_answer(query, final_evidence)
                    decision = "ANSWER"

        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000

        return {
            "query": query,
            "decision": decision,
            "confidence": confidence,
            "tool_trace": tool_trace,
            "final_answer": final_answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "latency_breakdown": latency_breakdown,
            "n_clusters": len(selected_domains),
            "fraction_kb": (
                len(selected_domains) / max(len(self.router.domains), 1)
                if self.router
                else 1.0
            ),
        }


class BaselineExecutor:
    """Baseline-1: raw full-KB retrieval + template answer.

    No routing, no triage, no reranker, no preference model.
    """

    def __init__(self, searcher, generator=None, cfg=None):
        self.searcher = searcher
        self.generator = generator
        self.top_k = getattr(cfg, "top_k_retrieval", 10)
        self.top_k_rr = getattr(cfg, "top_k_rerank", 5)

    def run(self, query: str, history: str = "") -> dict:
        t_start = time.time()

        t_search_start = time.time()
        results = self.searcher.search(
            query,
            top_k=self.top_k,
            domain=None,
            use_index=False,
        )
        search_ms = (time.time() - t_search_start) * 1000

        t_gen_start = time.time()
        final_answer, citations, _ = generate_answer(
            query,
            results[:self.top_k_rr],
            generator=self.generator,
        )
        gen_ms = (time.time() - t_gen_start) * 1000

        latency_ms = (time.time() - t_start) * 1000

        tool_trace = [{
            "tool": "SearchKB",
            "args": {
                "query": query,
                "top_k": self.top_k,
                "domain": None,
            },
            "result": {
                "passages": results,
            },
        }]

        return {
            "query": query,
            "decision": "ANSWER",
            "confidence": 1.0,
            "tool_trace": tool_trace,
            "final_answer": final_answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "latency_breakdown": {
                "routing_ms": 0.0,
                "search_ms": search_ms,
                "rerank_ms": 0.0,
                "gen_ms": gen_ms,
            },
            "n_clusters": 1,
            "fraction_kb": 1.0,
        }


class RuleWorkflowExecutor:
    """Baseline-2: full-KB retrieval + rule-based triage."""

    def __init__(self, searcher, router=None, generator=None, cfg=None):
        self.searcher = searcher
        self.router = router
        self.generator = generator
        self.top_k = getattr(cfg, "top_k_retrieval", 10)
        self.top_k_rr = getattr(cfg, "top_k_rerank", 5)
        self.evidence_answer_threshold = getattr(cfg, "evidence_answer_threshold", 0.40)
        self.ood_threshold = getattr(cfg, "cluster_out_of_domain_threshold", 0.20)

    def run(self, query: str, history: str = "") -> dict:
        t_start = time.time()

        latency_breakdown = {
            "routing_ms": 0.0,
            "search_ms": 0.0,
            "rerank_ms": 0.0,
            "gen_ms": 0.0,
        }

        t_search_start = time.time()
        results = self.searcher.search(
            query,
            top_k=self.top_k,
            domain=None,
            use_index=False,
        )
        latency_breakdown["search_ms"] = (time.time() - t_search_start) * 1000

        best_score = results[0].get("score", 0.0) if results else 0.0

        if best_score < self.ood_threshold:
            decision = "REJECT"
        elif best_score < self.evidence_answer_threshold:
            decision = "TICKET"
        else:
            decision = "ANSWER"

        tool_trace = [{
            "tool": "SearchKB",
            "args": {
                "query": query,
                "top_k": self.top_k,
            },
            "result": {
                "passages": results,
            },
        }]

        if decision == "REJECT":
            final_answer = format_reject_response()
            citations = []

        elif decision == "TICKET":
            final_answer = format_ticket_response("T-RULE-123", query)
            citations = []

        else:
            t_gen_start = time.time()
            final_answer, citations, _ = generate_answer(
                query,
                results[:self.top_k_rr],
                generator=self.generator,
            )
            latency_breakdown["gen_ms"] = (time.time() - t_gen_start) * 1000

        latency_ms = (time.time() - t_start) * 1000

        return {
            "query": query,
            "decision": decision,
            "confidence": 0.5,
            "tool_trace": tool_trace,
            "final_answer": final_answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "latency_breakdown": latency_breakdown,
            "n_clusters": 1,
            "fraction_kb": 1.0,
        }