"""Tool executor: orchestrates the full proposed pipeline.

Pipeline:
  1. RouteDomain (centroid + lexical gate)
  2. Triage prediction (ANSWER / TICKET / REJECT)
  3. If REJECT → RejectQuery
     If TICKET → SearchKB + CreateTicket
     If ANSWER → SearchKB (routed) → optional GetPolicy → Rerank → Generate
"""
import json
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from src.tools.tools import route_domain, search_kb, get_policy, create_ticket, reject_query
from src.generation.generate import generate_answer, template_answer
from src.generation.templates import format_reject_response, format_ticket_response
from src.utils.logging import get_logger
import re

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Vague / generic query detector (Part B)
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
    # Short queries (≤3 meaningful tokens) that match no domain context
    tokens = [t for t in q.lower().split() if len(t) > 2]
    if len(tokens) <= 2 and len(q) < 30:
        return True
    return False


# ---------------------------------------------------------------------------
# Personal / Action Request Detector (Part C)
# ---------------------------------------------------------------------------
_PERSONAL_ACTION_PATTERNS = [
    r"\b(check|view|see|status of|track|where is)\b.*\b(my|mine|i have)\b.*\b(status|application|case|claim|payment|benefit|account|order|file|form)\b",
    r"\b(update|change|reset|edit|modify)\b.*\b(my|mine)\b.*\b(info|information|address|phone|email|bank|direct deposit|account)\b",
    r"\b(submit|send|file|upload)\b.*\b(my|the)\b.*\b(form|document|file|application)\b.*\b(for me|on my behalf)\b",
    r"\b(guarantee|promise|assure)\b.*\b(approval|acceptance|eligibility|success)\b",
    r"\b(my|exact|specific)\b.*\b(amount|payment|benefit|check|balance)\b",
    r"\b(why|reason)\b.*\b(my|mine|i haven't)\b.*\b(delayed|late|not received|not arrived|missing)\b",
    r"\b(decide|decision|judge|approve)\b.*\b(my|this)\b.*\b(case|claim|application|request)\b",
    r"\b(private|personal|your|my)\b.*\b(phone|number|cell|mobile|address|email)\b"
]
_PERSONAL_ACTION_REGEX = re.compile("|".join(_PERSONAL_ACTION_PATTERNS), re.IGNORECASE)

def is_personal_or_action_request(query: str) -> bool:
    """Detect if the query asks for personal account actions or private info."""
    return bool(_PERSONAL_ACTION_REGEX.search(query))


def validate_answerability(query: str, final_evidence: List[dict], selected_domains: List[str]) -> dict:
    """Check if evidence directly addresses the query and is coherent."""
    if not final_evidence:
        return {"answerable": False, "reason": "No evidence found", "coherence_score": 0.0, "best_evidence": []}

    query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
    # Content terms: exclude very common stop words
    stop_words = {"how", "to", "the", "a", "an", "do", "i", "need", "have", "is", "for", "if", "what", "can", "you", "my", "of", "who"}
    content_tokens = {t for t in query_tokens if t not in stop_words and len(t) > 2}
    
    # Action terms indicate intent (stemmed/partial matching)
    action_terms = {"renew", "appli", "updat", "check", "submit", "eligibil", "document", "contact", "status", "registr", "enroll"}
    query_actions = {a for a in action_terms if any(a in qt for qt in query_tokens)}

    best_p = final_evidence[0]
    p_text = best_p.get("text", "").lower()
    p_tokens = set(re.findall(r"\b\w+\b", p_text))
    
    # Check 1: Direct content overlap (substring matching)
    overlap_terms = [t for t in content_tokens if t in p_text or t.rstrip('s') in p_text]
    overlap = len(overlap_terms)
    
    # Check 2: Action match
    action_match = any(a in p_text for a in query_actions) if query_actions else True
    
    # Check 3: Domain match
    p_doc_id = best_p.get("doc_id", "").lower()
    p_domain = best_p.get("domain", "").lower()
    if not p_domain:
        if "student aid" in p_doc_id or "studentaid" in p_doc_id: p_domain = "studentaid"
        elif "social security" in p_doc_id or "ssa" in p_doc_id: p_domain = "ssa"
        elif "veteran" in p_doc_id or "va" in p_doc_id: p_domain = "va"
        elif "dmv" in p_doc_id or "motor vehicle" in p_doc_id: p_domain = "dmv"
    domain_match = any(sd.lower() == p_domain for sd in selected_domains)

    # Coherence for multiple citations
    coherence_score = 1.0
    if len(final_evidence) > 1:
        p2_text = final_evidence[1].get("text","").lower()
        p2_tokens = set(re.findall(r"\b\w+\b", p2_text))
        p2_domain = final_evidence[1].get("domain", "").lower() or ("studentaid" if "studentaid" in final_evidence[1].get("doc_id","").lower() else "")
        if p_domain != p2_domain:
            coherence_score = 0.4 # Penalty for mixing domains
        elif len(p2_tokens.intersection(p_tokens)) < 3:
            coherence_score = 0.6 # Low overlap between chunks

    logger.info(f"[Validate] Overlap terms: {overlap_terms}, Actions: {query_actions}, ActionMatch: {action_match}, Score: {best_p.get('score'):.4f}, DomainMatch: {domain_match}")

    # Answerability rules
    is_answerable = (overlap >= 1 or not content_tokens) and action_match and (best_p.get("score", 0.0) >= 0.40 or (domain_match and best_p.get("score", 0.0) >= 0.30))
    
    if not is_answerable:
        reason = f"Evidence does not directly address query intent (Overlap={overlap}, ActionMatch={action_match})"
    elif coherence_score < 0.5:
        reason = "Multiple evidence chunks are incoherent/mixed domains"
        is_answerable = False
    else:
        reason = "Validated"

    return {
        "answerable": is_answerable,
        "reason": reason,
        "best_evidence": [best_p] if is_answerable and coherence_score < 0.8 else final_evidence[:2] if is_answerable else [],
        "coherence_score": coherence_score
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
        self.encoder           = encoder
        self.searcher          = searcher
        self.router            = router
        self.triage            = triage_predictor
        self.reranker          = reranker
        self.generator         = generator
        self.preference_scorer = preference_scorer
        self.chunk_by_id       = chunk_by_id or {}
        self.cfg               = cfg

        # Generator mode (Part A)
        self.generator_mode = "llm" if generator is not None else "template"
        self.top_k_retrieval  = getattr(cfg, "top_k_retrieval",  10)
        self.top_k_rerank     = getattr(cfg, "top_k_rerank",     4)
        self.top_k_domains    = getattr(cfg, "top_k_domains",    2)
        self.tau_domain       = getattr(cfg, "tau_domain",       0.35)
        self.tau_chunk        = getattr(cfg, "tau_chunk",        0.40)
        self.evidence_answer_threshold = getattr(cfg, "evidence_answer_threshold", 0.40)
        self.evidence_ticket_threshold = getattr(cfg, "evidence_ticket_threshold", 0.10)
        
        # Calibration thresholds
        self.tau_hard_reject = getattr(cfg, "tau_hard_reject", 0.08)
        self.tau_soft_domain = getattr(cfg, "tau_soft_domain", 0.20)
        
        self.cluster_out_of_domain_threshold = getattr(cfg, "cluster_out_of_domain_threshold", 0.20)
        self.cluster_confident_margin        = getattr(cfg, "cluster_confident_margin",        0.10)
        self.cluster_ambiguous_top_k         = getattr(cfg, "cluster_ambiguous_top_k",         self.top_k_domains)
        self.max_clusters_for_confident_query = getattr(cfg, "max_clusters_for_confident_query", 1)
        self.max_clusters_for_ambiguous_query = getattr(cfg, "max_clusters_for_ambiguous_query", self.top_k_domains)

    def run(self, query: str, history: str = "") -> dict:
        """Execute cluster-gated pipeline and return structured result."""
        t_start   = time.time()
        tool_trace = []
        
        latency_breakdown = {
            "routing_ms": 0.0,
            "search_ms": 0.0,
            "rerank_ms": 0.0,
            "gen_ms": 0.0
        }

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

        top_sim         = route_result.get("top_centroid_sim", 0.0)
        margin          = route_result.get("centroid_margin", 0.0)
        gate_result     = route_result.get("gate_result", "pass")
        route_decision  = route_result.get("decision", "route")
        domain_results  = route_result.get("result", {}).get("domains", [])

        # ------------------------------------------------------------------
        # Step 2: Domain Gate (REJECT check before retrieval)
        # ------------------------------------------------------------------
        selected_domains = []
        gating_status    = "unknown"
        domain_relevant  = True
        reject_allowed   = True
        retrieval_called = False

        matched_kws_by_domain = route_result.get("result", {}).get("matched_kws_by_domain", {})
        all_support_kws = route_result.get("support_keywords", [])
        kw_count = len(all_support_kws)

        # Logic: If query has strong domain keywords, focus on that domain
        strong_intent_domains = [d for d, kws in matched_kws_by_domain.items() if kws]
        
        decision = "route"
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
                selected_domains = [d["domain"] for d in domain_results[:2]]
                gating_status = "uncertain"
                domain_relevant = True
        elif strong_intent_domains:
            selected_domains = strong_intent_domains
            if domain_results and domain_results[0]["domain"] not in selected_domains:
                selected_domains.append(domain_results[0]["domain"])
            selected_domains = selected_domains[:2]
            gating_status = "keyword_intent"
            domain_relevant = True
        elif margin >= self.cluster_confident_margin:
            selected_domains = [domain_results[0]["domain"]] if domain_results else []
            gating_status = "confident"
            domain_relevant = True
        else:
            selected_domains = [d["domain"] for d in domain_results[:self.max_clusters_for_ambiguous_query]]
            gating_status = "ambiguous"
            domain_relevant = True

        # ------------------------------------------------------------------
        # Step 3: Retrieval (Gated)
        # ------------------------------------------------------------------
        passages = []
        best_evidence_score = 0.0
        final_evidence = []

        if not domain_relevant:
            logger.info(f"[Gate] Query is out-of-domain. REJECT decision finalized.")
        else:
            reject_allowed = False
            if selected_domains:
                t_search_start = time.time()
                # Use domain-specific indexes
                kb_result = search_kb(query, self.searcher, top_k=self.top_k_retrieval, domain=selected_domains)
                latency_breakdown["search_ms"] = (time.time() - t_search_start) * 1000
                tool_trace.append(kb_result)
                passages = kb_result["result"]["passages"]
                retrieval_called = True

            # --------------------------------------------------------------
            # Step 4: Domain-Aware Reranking & Guard
            # --------------------------------------------------------------
            if passages:
                query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
                action_terms = {"renew", "apply", "update", "check", "submit", "eligibility", "documents", "contact", "status", "enroll"}
                query_actions = query_tokens.intersection(action_terms)
                
                # Apply domain-intent scores
                for p in passages:
                    base_score = p.get("score", 0.0)
                    p_text = p.get("text", "").lower()
                    p_doc_id = p.get("doc_id", "").lower()
                    p_domain = p.get("domain", "").lower()
                    
                    # 1. Keyword overlap
                    p_tokens = set(re.findall(r"\b\w+\b", p_text))
                    overlap = len(query_tokens.intersection(p_tokens))
                    
                    # 2. Domain Intent Match
                    domain_bonus = 0.0
                    if p_domain in strong_intent_domains:
                        domain_bonus = 0.25
                    elif p_domain in selected_domains:
                        domain_bonus = 0.05
                    
                    # 3. Action Match Bonus
                    action_bonus = 0.20 if any(a in p_text for a in query_actions) else 0.0
                    
                    # 4. Wrong Domain / Wrong Intent Penalty
                    domain_penalty = 0.0
                    if strong_intent_domains and p_domain not in strong_intent_domains:
                        domain_penalty = 0.40

                    # Fine-grained intent penalty.
                    # Example: "driver's license" should not be answered from "non-driver ID card" evidence.
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

                # 4a. Rerank if model available
                if self.reranker:
                    t_rr_start = time.time()
                    reranked = self.reranker.rerank(query, passages, top_k=self.top_k_retrieval)
                    latency_breakdown["rerank_ms"] = (time.time() - t_rr_start) * 1000
                    for p in reranked:
                        p_domain = p.get("domain", "").lower()
                        if p_domain in strong_intent_domains:
                            p["score"] += 0.15
                    passages = reranked

                # 4b. Strict Relevance Guard
                for p in passages:
                    score = p.get("score", 0.0)
                    p_domain = p.get("domain", "").lower()
                    overlap = p.get("overlap", 0)
                    intent_match = True
                    if strong_intent_domains and p_domain not in strong_intent_domains:
                        intent_match = False
                        
                    is_relevant = (intent_match and (score >= self.evidence_answer_threshold or (overlap >= 4 and score >= self.evidence_ticket_threshold)))
                    if is_relevant: final_evidence.append(p)
                    if len(final_evidence) >= self.top_k_rerank: break

            # Sort by updated score (domain intent)
            final_evidence.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # Answerability Validation
            val_res = validate_answerability(query, final_evidence, selected_domains)
            final_evidence = val_res["best_evidence"]
            final_evidence.sort(key=lambda p: p.get("score", 0.0), reverse=True)
            best_evidence_score = final_evidence[0]["score"] if final_evidence and val_res["answerable"] else 0.0

        # 3.2 Triage Overrides (Part 6 Fix)
        if best_evidence_score > 0.80 and decision != "ANSWER":
            decision = "ANSWER"
            confidence = 1.0
            triage_method = "override_high_evidence"
            
        # ------------------------------------------------------------------
        # Step 5: Authoritative Triage
        # ------------------------------------------------------------------
        if not domain_relevant:
            decision = "REJECT"
        elif best_evidence_score >= self.evidence_answer_threshold:
            decision = "ANSWER"
        else:
            decision = "TICKET"

        # Use triage model for confidence estimation
        if self.triage is not None:
            triage_result = self.triage.predict(
                query               = query,
                keyword_gate        = gate_result,
                centroid_domain     = selected_domains[0] if selected_domains else "unknown",
                centroid_sim_top1   = top_sim,
                centroid_margin     = margin,
                nearest_chunk_sim   = best_evidence_score,
                retrieval_score_gap = 0.0,
                history             = history,
                tau_domain          = self.tau_domain,
                tau_chunk           = self.tau_chunk,
            )
            confidence = triage_result["confidence"]
            triage_method = "model_confidence"
        else:
            confidence = 1.0
            triage_method = "rule_default"

        tool_trace.append({
            "tool": "ClusterGating",
            "args": {"top_sim": top_sim, "margin": margin, "selected_domains": selected_domains},
            "result": {"best_evidence_score": best_evidence_score, "decision": decision, "confidence": confidence}
        })

        # ------------------------------------------------------------------
        # Step 5: Execute Final Action
        # ------------------------------------------------------------------
        if decision == "REJECT":
            rej_result = reject_query(reason="out_of_domain", nearest_kb_distance=1.0 - best_evidence_score, nearest_centroid_distance=1.0 - top_sim, confidence=confidence)
            tool_trace.append(rej_result)
            final_answer = format_reject_response()
            citations    = []

        elif decision == "TICKET":
            tkt_result = create_ticket(summary=query, category=selected_domains[0] if selected_domains else "general", severity="medium")
            tool_trace.append(tkt_result)
            final_answer = format_ticket_response(tkt_result["result"]["ticket_id"], query)
            citations  = []

        else:  # ANSWER
            if final_evidence:
                pol_res = get_policy(final_evidence[0]["doc_id"], final_evidence[0]["section_id"], self.chunk_by_id)
                tool_trace.append(pol_res)
            
            t_gen_start = time.time()
            final_answer, citations, is_insufficient = generate_answer(query=query, passages=final_evidence, generator=self.generator, preference_scorer=self.preference_scorer)
            latency_breakdown["gen_ms"] = (time.time() - t_gen_start) * 1000
            
            if is_insufficient:
                # If final_evidence exists, retrieval and validation succeeded.
                # A generator failure should fall back to template answer, not create a ticket.
                if final_evidence:
                    logger.warning(
                        "[Executor] Generator marked answer insufficient despite validated evidence; "
                        "using template fallback and keeping decision=ANSWER."
                    )
                    final_answer, citations, is_insufficient = template_answer(query, final_evidence)
                    decision = "ANSWER"
                else:
                    decision = "TICKET"
                    tkt_result = create_ticket(
                        summary=query,
                        category=selected_domains[0] if selected_domains else "general",
                        severity="medium"
                    )
                    tool_trace.append(tkt_result)
                    final_answer = format_ticket_response(tkt_result["result"]["ticket_id"], query)
                    citations = []

        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000

        return {
            "query":        query,
            "decision":     decision,
            "confidence":   confidence,
            "tool_trace":   tool_trace,
            "final_answer": final_answer,
            "citations":    citations,
            "latency_ms":   latency_ms,
            "latency_breakdown": latency_breakdown,
            "n_clusters":   len(selected_domains),
            "fraction_kb":  len(selected_domains) / max(len(self.router.domains), 1) if self.router else 1.0
        }


class BaselineExecutor:
    """Baseline-1: full-KB retrieval + template answer (no routing, no triage, no preference).
    Truly RAW: loads only pre-trained MiniLM and raw global index.
    """

    def __init__(self, searcher, generator=None, cfg=None):
        self.searcher = searcher
        self.generator = generator
        self.top_k    = getattr(cfg, "top_k_retrieval", 10)
        self.top_k_rr = getattr(cfg, "top_k_rerank",   5)

    def run(self, query: str, history: str = "") -> dict:
        t_start = time.time()
        
        # 1. RAW Global Linear Scan (no indexing, no clusters)
        t_search_start = time.time()
        results = self.searcher.search(query, top_k=self.top_k, domain=None, use_index=False)
        search_ms = (time.time() - t_search_start) * 1000
        
        # 2. RAW Generation (no reranking, no preference)
        t_gen_start = time.time()
        final_answer, citations, _ = generate_answer(query, results[:self.top_k_rr], generator=self.generator)
        gen_ms = (time.time() - t_gen_start) * 1000
        
        latency_ms = (time.time() - t_start) * 1000

        tool_trace = [{
            "tool": "SearchKB",
            "args": {"query": query, "top_k": self.top_k, "domain": None},
            "result": {"passages": results}
        }]

        return {
            "query":        query,
            "decision":     "ANSWER",
            "confidence":   1.0,
            "tool_trace":   tool_trace,
            "final_answer": final_answer,
            "citations":    citations,
            "latency_ms":   latency_ms,
            "latency_breakdown": {
                "routing_ms": 0.0,
                "search_ms": search_ms,
                "rerank_ms": 0.0,
                "gen_ms": gen_ms
            },
            "n_clusters":   1,
            "fraction_kb":  1.0
        }


class RuleWorkflowExecutor:
    """Baseline-2: full-KB retrieval + rule-based triage."""

    def __init__(self, searcher, router=None, generator=None, cfg=None):
        self.searcher = searcher
        self.router   = router
        self.generator = generator
        self.top_k    = getattr(cfg, "top_k_retrieval", 10)
        self.top_k_rr = getattr(cfg, "top_k_rerank",   5)
        self.evidence_answer_threshold = getattr(cfg, "evidence_answer_threshold", 0.40)
        self.ood_threshold             = getattr(cfg, "cluster_out_of_domain_threshold", 0.20)

    def run(self, query: str, history: str = "") -> dict:
        t_start = time.time()
        latency_breakdown = {"routing_ms": 0.0, "search_ms": 0.0, "rerank_ms": 0.0, "gen_ms": 0.0}

        # 1. ALWAYS perform RAW Linear Scan first (no early gating)
        t_search_start = time.time()
        results = self.searcher.search(query, top_k=self.top_k, domain=None, use_index=False)
        latency_breakdown["search_ms"] = (time.time() - t_search_start) * 1000
        
        best_score = results[0]["score"] if results else 0.0

        # 2. Decide response type AFTER search
        # Simple thresholds for 3-way triage
        if best_score < self.ood_threshold:
            decision = "REJECT"
        elif best_score < self.evidence_answer_threshold:
            decision = "TICKET"
        else:
            decision = "ANSWER"

        tool_trace = [{"tool": "SearchKB", "args": {"query": query, "top_k": self.top_k}, "result": {"passages": results}}]
        
        if decision == "REJECT":
            final_answer = format_reject_response()
            citations = []
        elif decision == "TICKET":
            final_answer = format_ticket_response("T-RULE-123", query)
            citations = []
        else:
            t_gen_start = time.time()
            final_answer, citations, _ = generate_answer(query, results[:self.top_k_rr], generator=self.generator)
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
            "fraction_kb": 1.0
        }
