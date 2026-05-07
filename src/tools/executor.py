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
from src.generation.generate import generate_answer
from src.generation.templates import format_reject_response, format_ticket_response
from src.utils.logging import get_logger

logger = get_logger(__name__)


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

        # Config defaults
        self.top_k_retrieval  = getattr(cfg, "top_k_retrieval",  10)
        self.top_k_rerank     = getattr(cfg, "top_k_rerank",     5)
        self.top_k_domains    = getattr(cfg, "top_k_domains",    2)
        self.tau_domain       = getattr(cfg, "tau_domain",       0.35)
        self.tau_chunk        = getattr(cfg, "tau_chunk",        0.40)
        self.mu_boundary      = getattr(cfg, "mu_boundary",      0.15)

    def run(self, query: str, history: str = "") -> dict:
        """Execute full pipeline and return structured result with tool trace."""
        t_start   = time.time()
        tool_trace = []

        # ------------------------------------------------------------------
        # Step 1: Encode query
        # ------------------------------------------------------------------
        query_embedding = self.searcher.get_query_embedding(query)

        # ------------------------------------------------------------------
        # Step 2: RouteDomain
        # ------------------------------------------------------------------
        route_result = route_domain(
            query,
            query_embedding,
            self.router,
            top_k_domains=self.top_k_domains,
            tau_domain=self.tau_domain,
        )
        tool_trace.append(route_result)
        top_domain      = route_result.get("top_domain")
        top_centroid_sim = route_result.get("top_centroid_sim", 0.0)
        centroid_margin  = route_result.get("centroid_margin", 0.0)
        gate_result      = route_result.get("gate_result", "pass")
        route_decision   = route_result.get("decision", "route")

        # ------------------------------------------------------------------
        # Step 3: Quick retrieval to get KB proximity signals
        # ------------------------------------------------------------------
        quick_results = self.searcher.search(query, top_k=3, domain=top_domain)
        nearest_chunk_sim   = quick_results[0]["score"] if quick_results else 0.0
        retrieval_score_gap = (
            quick_results[0]["score"] - quick_results[-1]["score"]
            if len(quick_results) >= 2 else 0.0
        )

        # ------------------------------------------------------------------
        # Step 4: Triage / tool-policy prediction
        # ------------------------------------------------------------------
        if self.triage is not None:
            triage_result = self.triage.predict(
                query               = query,
                keyword_gate        = gate_result,
                centroid_domain     = top_domain or "unknown",
                centroid_sim_top1   = top_centroid_sim,
                centroid_margin     = centroid_margin,
                nearest_chunk_sim   = nearest_chunk_sim,
                retrieval_score_gap = retrieval_score_gap,
                history             = history,
                tau_domain          = self.tau_domain,
                tau_chunk           = self.tau_chunk,
            )
            decision   = triage_result["decision"]
            confidence = triage_result["confidence"]
        else:
            # Fallback rule-based triage
            if route_decision == "reject":
                decision, confidence = "REJECT", 0.95
            elif nearest_chunk_sim < self.tau_chunk:
                decision, confidence = "TICKET", 0.70
            else:
                decision, confidence = "ANSWER", 0.80
            triage_result = {"decision": decision, "confidence": confidence, "method": "rule"}

        tool_trace.append({
            "tool": "Triage",
            "args": {
                "query": query,
                "keyword_gate": gate_result,
                "centroid_domain": top_domain,
                "nearest_chunk_sim": nearest_chunk_sim,
            },
            "result": triage_result,
        })

        # ------------------------------------------------------------------
        # Step 5: Execute based on decision
        # ------------------------------------------------------------------
        t_retrieval_start = time.time()

        if decision == "REJECT":
            rej_result = reject_query(
                reason                   = "out_of_domain" if gate_result == "reject" else "too_far_from_kb",
                nearest_kb_distance      = 1.0 - nearest_chunk_sim,
                nearest_centroid_distance = 1.0 - top_centroid_sim,
                confidence               = confidence,
            )
            tool_trace.append(rej_result)
            final_answer = format_reject_response()
            citations    = []

        elif decision == "TICKET":
            # SearchKB for context
            kb_result = search_kb(query, self.searcher, top_k=self.top_k_retrieval, domain=top_domain)
            tool_trace.append(kb_result)
            passages = kb_result["result"]["passages"]

            tkt_result = create_ticket(
                summary  = query,
                category = top_domain or "general",
                severity = "medium",
            )
            tool_trace.append(tkt_result)
            ticket_id  = tkt_result["result"]["ticket_id"]
            final_answer = format_ticket_response(ticket_id, query)
            citations  = []

        else:  # ANSWER
            # Search top domain first
            kb_result1 = search_kb(query, self.searcher, top_k=self.top_k_retrieval, domain=top_domain)
            tool_trace.append(kb_result1)
            passages = kb_result1["result"]["passages"]

            # Fallback to second domain if evidence is weak
            route_domains = route_result.get("result", {}).get("domains", [])
            if len(route_domains) > 1 and (not passages or passages[0]["score"] < self.tau_chunk):
                second_domain = route_domains[1]["domain"]
                kb_result2 = search_kb(query, self.searcher, top_k=self.top_k_retrieval // 2, domain=second_domain)
                tool_trace.append(kb_result2)
                passages = passages + kb_result2["result"]["passages"]

            # Optional GetPolicy for top doc
            if passages:
                top_doc_id = passages[0]["doc_id"]
                top_sec_id = passages[0]["section_id"]
                pol_result = get_policy(top_doc_id, top_sec_id, self.chunk_by_id)
                tool_trace.append(pol_result)

            # Rerank
            if self.reranker and passages:
                passage_dicts = [{"text": p["text"], **p} for p in passages]
                reranked = self.reranker.rerank(query, passage_dicts, top_k=self.top_k_rerank)
                passages = reranked

            retrieval_latency_ms = (time.time() - t_retrieval_start) * 1000

            # Generate answer
            final_answer, citations = generate_answer(
                query=query,
                passages=passages[:self.top_k_rerank],
                generator=self.generator,
                preference_scorer=self.preference_scorer,
            )

        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000

        return {
            "query":        query,
            "history":      history,
            "decision":     decision,
            "confidence":   confidence,
            "tool_trace":   tool_trace,
            "final_answer": final_answer,
            "citations":    citations,
            "latency_ms":   latency_ms,
        }


class BaselineExecutor:
    """Baseline: full-KB retrieval + template answer (no routing, no triage, no preference)."""

    def __init__(self, encoder, searcher, reranker=None, generator=None, cfg=None):
        self.encoder  = encoder
        self.searcher = searcher
        self.reranker = reranker
        self.generator = generator
        self.top_k    = getattr(cfg, "top_k_retrieval", 10)
        self.top_k_rr = getattr(cfg, "top_k_rerank",   5)

    def run(self, query: str, history: str = "") -> dict:
        t_start = time.time()
        results = self.searcher.search(query, top_k=self.top_k, domain=None)
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=self.top_k_rr)
        final_answer, citations = generate_answer(query, results[:self.top_k_rr], generator=self.generator)
        latency_ms = (time.time() - t_start) * 1000
        return {
            "query":        query,
            "decision":     "ANSWER",
            "confidence":   1.0,
            "tool_trace":   [],
            "final_answer": final_answer,
            "citations":    citations,
            "latency_ms":   latency_ms,
        }
