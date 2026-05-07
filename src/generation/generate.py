"""Answer generation using Flan-T5 or template fallback.

Generator is optional. If not provided, template-based generation is used.
"""
import re
import os
import torch
from typing import List, Optional, Tuple


def format_evidence(passages: List[dict]) -> str:
    """Format top passages as evidence text for the prompt."""
    parts = []
    for i, p in enumerate(passages[:3]):
        parts.append(f"[{i+1}] {p['text'][:300]}")
    return "\n".join(parts)


def extract_citations(passages: List[dict]) -> List[dict]:
    """Extract citation metadata from passages."""
    citations = []
    for p in passages:
        citations.append({
            "doc_id":     p.get("doc_id",     ""),
            "chunk_id":   p.get("chunk_id",   ""),
            "section_id": p.get("section_id", ""),
            "span_start": p.get("span_start", 0),
            "span_end":   p.get("span_end",   0),
        })
    return citations


def template_answer(query: str, passages: List[dict]) -> Tuple[str, List[dict]]:
    """Generate a template-based answer with citations."""
    if not passages:
        return (
            "I could not find relevant information in the knowledge base for your query. "
            "Please contact support for further assistance.",
            [],
        )

    top_p = passages[0]
    evidence_snippet = top_p["text"][:300].strip()
    doc_id   = top_p.get("doc_id", "")
    chunk_id = top_p.get("chunk_id", "")
    span_s   = top_p.get("span_start", 0)
    span_e   = top_p.get("span_end", 0)

    answer = (
        f"Based on the knowledge base: {evidence_snippet} "
        f"[doc_id={doc_id}, chunk_id={chunk_id}, span={span_s}-{span_e}]"
    )
    citations = extract_citations(passages[:3])
    return answer, citations


def generate_answer(
    query: str,
    passages: List[dict],
    generator=None,
    preference_scorer=None,
    num_candidates: int = 3,
) -> Tuple[str, List[dict]]:
    """
    Generate final cited answer.
    If generator (Flan-T5) is available: generate candidates, score with preference ranker.
    Otherwise: use template-based generation.
    """
    citations = extract_citations(passages[:3])

    if generator is None:
        return template_answer(query, passages)

    # --- Generator-based answer ---
    evidence_text = format_evidence(passages)
    prompt = (
        f"Answer the following customer support question using only the provided evidence. "
        f"Include a citation in the format [doc_id=X, chunk_id=Y, span=A-B] at the end.\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Answer:"
    )

    candidates = generator.generate(prompt, num_return_sequences=num_candidates)

    # Ensure at least one candidate has a citation
    if citations:
        top_p = passages[0]
        citation_str = (
            f"[doc_id={top_p.get('doc_id','')}, "
            f"chunk_id={top_p.get('chunk_id','')}, "
            f"span={top_p.get('span_start',0)}-{top_p.get('span_end',0)}]"
        )
        # Append citation to last candidate if none present
        if not any(re.search(r"\[doc_id=", c) for c in candidates):
            candidates[-1] = candidates[-1].rstrip() + " " + citation_str

    # Prefer with scorer
    if preference_scorer is not None and len(candidates) > 1:
        best = preference_scorer.select_best(query, candidates, passages)
    else:
        best = candidates[0]

    return best, citations


class FlanT5Generator:
    """Wraps Flan-T5 for seq2seq generation."""

    def __init__(
        self,
        model_path: str = "google/flan-t5-small",
        device: torch.device = None,
        max_new_tokens: int = 150,
    ):
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        if device is None:
            from src.utils.device import get_device
            device = get_device("auto")
        self.device         = device
        self.max_new_tokens = max_new_tokens

        print(f"[generator] Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:
        """Generate answer candidates from a prompt."""
        enc = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            max_new_tokens  = self.max_new_tokens,
            num_beams       = max(num_return_sequences, 4),
            num_return_sequences = num_return_sequences,
            early_stopping  = True,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded


def load_generator(model_path: str, device=None) -> Optional[FlanT5Generator]:
    """Load generator. If path does not exist, return None (template mode)."""
    if not model_path or not os.path.isdir(model_path):
        # Also check if it's a known HF model name if we want to allow that
        if model_path and "/" in model_path: # simplified check for HF model name
            pass
        else:
            return None
    try:
        return FlanT5Generator(model_path, device=device)
    except Exception as e:
        print(f"[generator] Could not load generator: {e}. Using template mode.")
        return None
