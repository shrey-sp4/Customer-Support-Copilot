"""Tool implementations — pure Python, no LLM required for tool execution."""
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional

from src.tools.schema import (
    RouteDomainResult, DomainResult,
    SearchKBResult, PassageResult,
    GetPolicyResult,
    CreateTicketResult,
    RejectQueryResult,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global ticket counter (in-memory for demo)
_ticket_counter = 0


def route_domain(
    query: str,
    query_embedding,
    router,
    top_k_domains: int = 2,
    tau_domain: float = 0.35,
) -> dict:
    """Execute RouteDomain tool."""
    result = router.full_route(query, query_embedding, top_k=top_k_domains, tau_domain=tau_domain)
    domains = [
        DomainResult(
            domain=dr["domain"],
            centroid_similarity=dr["centroid_similarity"],
            centroid_distance=float(1.0 - dr["centroid_similarity"]),
            matched_keywords=dr.get("matched_keywords", []),
        )
        for dr in result.get("domain_results", [])
    ]
    return {
        "tool": "RouteDomain",
        "args": {"query": query, "top_k_domains": top_k_domains},
        "result": asdict(RouteDomainResult(
            domains=domains,
            route_confidence=result.get("top_score", 0.0),
            matched_kws_by_domain=result.get("matched_kws_by_domain", {}),
        )),
        "gate_result":      result.get("gate_result", "pass"),
        "top_domain":       result.get("top_domain"),
        "top_centroid_sim": result.get("top_centroid_sim", 0.0),
        "top_score":        result.get("top_score", 0.0),
        "decision":         result.get("decision", "route"),
        "support_keywords": result.get("support_keywords", []),
    }


def search_kb(
    query: str,
    searcher,
    top_k: int = 5,
    domain: Optional[str] = None,
) -> dict:
    """Execute SearchKB tool."""
    raw_results = searcher.search(query, top_k=top_k, domain=domain)
    passages = [
        PassageResult(
            doc_id=r["doc_id"],
            chunk_id=r["chunk_id"],
            section_id=r["section_id"],
            span_start=r["span_start"],
            span_end=r["span_end"],
            text=r["text"],
            score=r["score"],
            domain=r.get("domain", ""),
        )
        for r in raw_results
    ]
    return {
        "tool": "SearchKB",
        "args": {"query": query, "top_k": top_k, "domain": domain},
        "result": asdict(SearchKBResult(passages=passages)),
    }


def get_policy(
    doc_id: str,
    section_id: str = "",
    chunk_by_id: Dict[str, dict] = None,
) -> dict:
    """Execute GetPolicy tool — fetches full policy text for a doc/section."""
    policy_text = ""
    if chunk_by_id:
        # Concatenate all chunks for this section
        matching = [
            ch["text"] for ch in chunk_by_id.values()
            if ch.get("doc_id") == doc_id
            and (not section_id or ch.get("section_id") == section_id)
        ]
        policy_text = " ".join(matching).strip()
        
    if not policy_text:
        policy_text = f"[Policy text for {doc_id}/{section_id} not found in KB]"
        
    return {
        "tool": "GetPolicy",
        "args": {"doc_id": doc_id, "section_id": section_id},
        "result": asdict(GetPolicyResult(
            policy_text=policy_text,
            doc_id=doc_id,
            section_id=section_id,
        )),
    }


def create_ticket(
    summary: str,
    category: str = "general",
    severity: str = "medium",
) -> dict:
    """Execute CreateTicket tool."""
    global _ticket_counter
    _ticket_counter += 1
    ticket_id = f"TCK-{_ticket_counter:06d}"
    logger.info(f"Created ticket {ticket_id}: {summary[:60]}")
    return {
        "tool": "CreateTicket",
        "args": {"summary": summary, "category": category, "severity": severity},
        "result": asdict(CreateTicketResult(ticket_id=ticket_id, status="created")),
    }


def reject_query(
    reason: str = "out_of_domain",
    nearest_kb_distance: float = 1.0,
    nearest_centroid_distance: float = 1.0,
    confidence: float = 1.0,
) -> dict:
    """Execute RejectQuery tool."""
    message = (
        "I can only help with questions covered by this support knowledge base. "
        "Your question appears outside the supported domains, so I cannot answer it here."
    )
    return {
        "tool": "RejectQuery",
        "args": {
            "reason":                   reason,
            "nearest_kb_distance":      nearest_kb_distance,
            "nearest_centroid_distance": nearest_centroid_distance,
            "confidence":               confidence,
        },
        "result": asdict(RejectQueryResult(decision="rejected", message=message)),
    }
