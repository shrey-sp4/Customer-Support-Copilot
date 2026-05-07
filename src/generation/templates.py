"""Response templates for REJECT and TICKET decisions."""


def format_reject_response() -> str:
    return (
        "I can only help with questions covered by this support knowledge base. "
        "Your question appears outside the supported domains, so I cannot answer it here."
    )


def format_ticket_response(ticket_id: str, query: str) -> str:
    return (
        f"I could not find enough knowledge base evidence to answer this confidently, "
        f"but your issue appears related to our support domain. "
        f"I have created a support ticket for human review.\n"
        f"Ticket ID: {ticket_id}"
    )


def format_answer_with_citation(answer_text: str, citations: list) -> str:
    """Append citation string to answer if not already present."""
    import re
    if citations and not re.search(r"\[doc_id=", answer_text):
        c = citations[0]
        citation_str = (
            f"[doc_id={c.get('doc_id','')}, "
            f"chunk_id={c.get('chunk_id','')}, "
            f"span={c.get('span_start',0)}-{c.get('span_end',0)}]"
        )
        answer_text = answer_text.rstrip() + " " + citation_str
    return answer_text
