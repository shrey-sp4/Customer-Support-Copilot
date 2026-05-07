"""Tool schema definitions as Python dataclasses."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class RouteDomainArgs:
    query: str
    top_k_domains: int = 2


@dataclass
class DomainResult:
    domain: str
    centroid_similarity: float
    centroid_distance: float
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class RouteDomainResult:
    domains: List[DomainResult]
    route_confidence: float


@dataclass
class SearchKBArgs:
    query: str
    top_k: int = 5
    domain: Optional[str] = None


@dataclass
class PassageResult:
    doc_id: str
    chunk_id: str
    section_id: str
    span_start: int
    span_end: int
    text: str
    score: float
    domain: str = ""


@dataclass
class SearchKBResult:
    passages: List[PassageResult]


@dataclass
class GetPolicyArgs:
    doc_id: str
    section_id: str = ""


@dataclass
class GetPolicyResult:
    policy_text: str
    doc_id: str
    section_id: str


@dataclass
class CreateTicketArgs:
    summary: str
    category: str
    severity: str = "medium"  # low | medium | high


@dataclass
class CreateTicketResult:
    ticket_id: str
    status: str = "created"


@dataclass
class RejectQueryArgs:
    reason: str                   # out_of_domain | unsupported_domain | unsafe | too_far_from_kb
    nearest_kb_distance: float
    nearest_centroid_distance: float
    confidence: float


@dataclass
class RejectQueryResult:
    decision: str = "rejected"
    message: str = "Your query is outside the supported knowledge base."


# Tool name registry
TOOL_NAMES = ["RouteDomain", "SearchKB", "GetPolicy", "CreateTicket", "RejectQuery"]
