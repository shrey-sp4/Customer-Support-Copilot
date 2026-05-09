"""Answer generation using Flan-T5 or template fallback.

Generator is optional. If not provided, template-based generation is used.
"""
import re
import os
import torch
from typing import List, Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)

def format_evidence(passages: List[dict]) -> str:
    """Format top passages as evidence text for the prompt."""
    parts = []
    for i, p in enumerate(passages): # Use all provided passages
        text = p.get("text", "")
        # Minimal cleaning for the prompt
        text = re.sub(r"\s+", " ", text).strip()
        parts.append(f"[{i+1}] {text}")
    return "\n".join(parts)


def extract_citations(passages: List[dict]) -> List[str]:
    """Extract formatted citation strings from passages."""
    citations = []
    for p in passages:
        doc_id = p.get("doc_id", "unknown")
        chunk_id = p.get("chunk_id", "unknown")
        s = p.get("span_start", 0)
        e = p.get("span_end", 0)
        if s or e:
            citations.append(f"[{doc_id}:{chunk_id} {s}-{e}]")
        else:
            citations.append(f"[{doc_id}:{chunk_id}]")
    return list(dict.fromkeys(citations)) # dedup


def clean_text_formatting(text: str) -> str:
    """Fix common spacing, punctuation, and apostrophe issues."""
    text = re.sub(r"\bLet s\b", "Let's", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou ll\b", "you'll", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou ve\b", "you've", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdon t\b", "don't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan t\b", "can't", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)\s+s\b", r"\1's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDepartment of Labor s\b", "Department of Labor's", text, flags=re.IGNORECASE)
    
    # Fix spacing and punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"([.,!?])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove common incomplete suffixes/broken words
    text = re.sub(r"\s*\.\.\.$", ".", text)
    
    return text

def _extract_clean_sentences(text: str, query: str, max_chars: int = 5000) -> str:
    """Extract the most relevant complete sentences from a passage."""
    text = clean_text_formatting(text)

    # Remove section headings (lines that are short and Title Case or ALL CAPS)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    filtered = []
    for line in lines:
        words = line.split()
        is_heading = (len(words) <= 6 and line.istitle()) or line.isupper() or line.endswith(":")
        if not is_heading:
            filtered.append(line)
    text = " ".join(filtered)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Score each sentence by query token overlap
    query_tokens = set(re.findall(r'\w+', query.lower()))
    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        overlap = len(query_tokens & set(re.findall(r'\w+', s.lower())))
        scored.append((overlap, s))
    scored.sort(key=lambda x: -x[0])

    # Take top sentences that fit within max_chars
    result_sents = []
    total = 0
    for _, s in scored[:4]:
        if total + len(s) <= max_chars:
            result_sents.append(s)
            total += len(s)

    if not result_sents:
        # Fallback: take first complete sentence
        for s in sentences:
            if len(s) >= 20 and s[0].isupper():
                result_sents.append(s)
                break

    result = " ".join(result_sents).strip()
    
    # Remove broken trailing words or ellipses
    # Remove only obvious broken endings, not every final word.
    result = re.sub(r"\s*\.\.\.$", ".", result)

    if result.endswith(" fin"):
        result = result[:-4] + "."

    # If text ends with an incomplete connector, remove only that connector.
    result = re.sub(
        r"\s+(and|or|with|for|to|of|in|on|at|by|from)$",
        "",
        result,
        flags=re.IGNORECASE
    )
        
    return result


def template_answer(query: str, passages: List[dict]) -> Tuple[str, List[str], bool]:
    """Generate a clean, support-style answer with citations from selected evidence."""
    if not passages:
        return (
            "I could not find enough evidence in the knowledge base to answer this query. "
            "I've created a support ticket for further investigation.",
            [],
            True
        )

    citations = extract_citations(passages)

    # Build answer body from passages
    body_parts = []
    for p in passages:
        snippet = _extract_clean_sentences(p.get("text", ""), query)
        if snippet:
            body_parts.append(snippet)

    body = " ".join(body_parts).strip()

    if not body:
        # Hard fallback: first 350 chars of top passage, trimmed to sentence boundary
        raw = passages[0].get("text", "").strip()[:350]
        last_period = raw.rfind(".")
        body = raw[:last_period + 1] if last_period > 50 else raw

    # Final support-style wrapping
    if len(body.split()) < 15:
        answer = f"The knowledge base states: {body}"
    else:
        answer = body

    # Ensure it ends with proper punctuation
    if answer and answer[-1] not in ".!?":
        answer += "."

    cit_str = " " + " ".join(citations[:2]) if citations else ""
    return answer + cit_str, citations, False


def verify_grounding(answer: str, passages: List[dict]) -> Tuple[bool, str]:
    """Verify that every sentence in the answer is supported by the evidence."""
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    evidence_text = " ".join([p.get("text", "") for p in passages]).lower()
    evidence_tokens = set(re.findall(r"\b\w{4,}\b", evidence_text)) # Use tokens with length >= 4
    
    for sent in sentences:
        if not sent.strip(): continue
        sent_tokens = set(re.findall(r"\b\w{4,}\b", sent.lower()))
        # Check if at least 30% of significant tokens exist in evidence
        if sent_tokens:
            overlap = len(sent_tokens.intersection(evidence_tokens))
            coverage = overlap / len(sent_tokens)
            if coverage < 0.3:
                return False, f"Unsupported claim: '{sent[:50]}...'"
                
    return True, "Verified"


def validate_answer_quality(answer: str, query: str, citations: List[str]) -> bool:
    """Perform quality gate checks on the generated answer."""
    if not answer or len(answer.strip()) < 15:
        logger.warning(f"[Quality] Rejected: Too short ({len(answer.strip())} chars)")
        return False
    
    words = answer.split()
    if len(words) > 250:
        logger.warning(f"[Quality] Rejected: Too long ({len(words)} words)")
        return False
    
    # Check for excessive repetition
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) > 2:
        for i in range(len(sentences)-1):
            if sentences[i].strip().lower() == sentences[i+1].strip().lower():
                logger.warning(f"[Quality] Rejected: Repetition detected")
                return False
                
    return True


def generate_answer(
    query: str,
    passages: List[dict],
    generator=None,
    preference_scorer=None,
    num_candidates: int = 3,
) -> Tuple[str, List[str], bool]:
    """
    Generate final cited answer.
    Returns (answer, citations, is_insufficient).
    """
    citations = extract_citations(passages)

    if generator is None:
        return template_answer(query, passages)

    # --- Generator-based answer ---
    evidence_text = format_evidence(passages)
    
    def run_generation(strict=False):
        style_instr = "Provide a comprehensive, detailed, and to-the-point answer."
        if strict:
            style_instr = "Provide a concise, direct answer focusing on the most relevant facts."
            
        prompt = (
            "Answer the following question using ONLY the provided evidence.\n"
            f"{style_instr}\n"
            "If the evidence is not sufficient to answer, write: INSUFFICIENT_EVIDENCE.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Answer:"
        )
        candidates = generator.generate(prompt, num_return_sequences=num_candidates)
        
        if preference_scorer is not None and len(candidates) > 1:
            best = preference_scorer.select_best(query, candidates, passages)
        else:
            best = candidates[0]
        return best

    best = run_generation(strict=False)
    
    # Retry logic (Part D)
    if not best or "INSUFFICIENT_EVIDENCE" in best or len(best.split()) < 5:
        logger.info("[Generation] Triggering retry with strict prompt...")
        best = run_generation(strict=True)

    # Final quality gate (Part E)
        
    is_grounded, grounding_reason = verify_grounding(best, passages)

    if (
        not best
        or "INSUFFICIENT_EVIDENCE" in best
        or not validate_answer_quality(best, query, citations)
        or not is_grounded
    ):
        logger.warning(
            f"[Generation] Quality/Grounding gate failed. "
            f"Reason: {grounding_reason if not is_grounded else 'Quality'}. "
            f"Result: {best[:50] if best else 'EMPTY'}..."
        )

        # Critical fix:
        # If passages exist, retrieval/evidence worked.
        # A bad neural generation should fall back to template answer,
        # not trigger ticket creation.
        if passages:
            logger.warning(
                "[Generation] Neural generation failed despite available evidence; "
                "falling back to template answer."
            )
            return template_answer(query, passages)

        return (
            "I could not find enough evidence in the knowledge base to answer this query.",
            citations,
            True
        )

    # Post-process for common tokenizer issues
    best = clean_text_formatting(best)

    # Ensure punctuation before citation
    if best and best[-1] not in ".!?":
        best += "."

    # Programmatically append the primary citation if not already mentioned
    if citations and not any(c in best for c in citations):
        best = best.rstrip() + f" {citations[0]}"

    return best, citations, False


class FlanT5Generator:
    """Wraps Flan-T5 for seq2seq generation."""

    def __init__(
        self,
        model_path: str = "google/flan-t5-base",
        device: torch.device = None,
        max_new_tokens: int = 120,
        num_beams: int = 4,
        temperature: float = 0.0,
    ):
        from transformers import T5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
        if device is None:
            from src.utils.device import get_device
            device = get_device("auto")
        self.device         = device
        self.max_new_tokens = max_new_tokens
        self.num_beams       = num_beams
        self.temperature    = temperature

        # 4-bit configuration for hardware efficiency (as per README)
        bnb_config = None
        if "cuda" in str(device).lower():
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("[generator] 4-bit quantization enabled.")
            except Exception:
                print("[generator] bitsandbytes not found or incompatible. Loading in full precision...")

        print(f"[generator] Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            if bnb_config:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_path, 
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                self.model.to(device)
        except Exception as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower() or "paging file" in str(e).lower():
                print(f"[warning] Failed to load generator on {device}. Falling back to CPU...")
                self.device = torch.device("cpu")
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                self.model.to(self.device)
            else:
                raise e
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, num_return_sequences: int = 1) -> List[str]:
        """Generate answer candidates from a prompt."""
        enc = self.tokenizer(
            prompt,
            max_length=768,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        gen_config = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": max(num_return_sequences, self.num_beams),
            "num_return_sequences": num_return_sequences,
            "early_stopping": True,
        }
        if self.temperature > 0:
            gen_config["do_sample"] = True
            gen_config["temperature"] = self.temperature
        else:
            gen_config["do_sample"] = False

        outputs = self.model.generate(**gen_config)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded


def load_generator(model_path: str, device=None, cfg: dict = None) -> Optional[FlanT5Generator]:
    """Load generator. If path does not exist, return None (template mode)."""
    if cfg is None: cfg = {}
    
    # Priority: 1. Passed model_path, 2. Config model_name, 3. Config generator_model_name
    model_name = model_path or cfg.get("generator_model_name") or cfg.get("generator_model") or "google/flan-t5-base"
    fallback_name = cfg.get("generator_fallback_model_name") or "google/flan-t5-small"

    try:
        return FlanT5Generator(
            model_path=model_name,
            device=device,
            max_new_tokens=cfg.get("generator_max_new_tokens", 120),
            num_beams=cfg.get("generator_num_beams", 4),
            temperature=cfg.get("generator_temperature", 0.0),
        )
    except Exception as e:
        err_msg = str(e).split("\n")[0]
        print(f"[warning] Could not load generator '{model_name}' ({err_msg}). Trying fallback...")
        try:
            return FlanT5Generator(
                model_path=fallback_name,
                device=device,
                max_new_tokens=cfg.get("generator_max_new_tokens", 120),
                num_beams=cfg.get("generator_num_beams", 4),
                temperature=cfg.get("generator_temperature", 0.0),
            )
        except Exception as e2:
            err_msg2 = str(e2).split("\n")[0]
            print(f"[error] Fallback failed ({err_msg2}). Generator mode: template fallback")
            return None
