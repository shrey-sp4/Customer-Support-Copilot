import re
import numpy as np
from typing import List, Dict

def compute_answer_quality_metrics(results: List[dict]) -> dict:
    """
    Compute automated quality metrics for ANSWER predictions.
    
    Metrics:
    - AnswerQualityScore
    - DirectAnswerRate
    - FragmentRate
    - RepetitionRate
    - BadGrammarRate
    - WrongDomainCitationRate
    - IncoherentMultiEvidenceRate
    - AnswerTooLongRate
    - AnswerHasCitationRate
    - UnsupportedAnswerRate
    """
    answers = [r for r in results if r.get("decision") == "ANSWER"]
    if not answers:
        return {}

    has_citations = []
    lengths = []
    too_long = []
    fragments = []
    bad_grammar = []
    unsupported = []
    repetition = []
    direct_answer = []
    wrong_domain = []
    incoherent = []
    rubric_scores = []

    bad_punct_pattern = re.compile(r"\b(it s|you ll|don t|can t|Labor s|Department s|government s|religious or|they re|i m)\b", re.IGNORECASE)
    action_terms = {"renew", "apply", "update", "check", "submit", "eligibility", "documents", "contact", "status"}

    for res in answers:
        text = res.get("final_answer", "")
        citations = res.get("citations", [])
        query = res.get("query", "").lower()
        gold_domain = (res.get("gold_domain") or "").lower()
        
        # 1. Has citation
        has_cit = 1 if citations else 0
        has_citations.append(has_cit)
        
        # 2. Length
        words = text.split()
        length = len(words)
        lengths.append(length)
        
        # 3. Too long (>150 words)
        is_too_long = 1 if length > 150 else 0
        too_long.append(is_too_long)
        
        # 4. Fragment rate
        is_fragment = 0
        if text:
            clean_text = text.strip()
            while True:
                clean_text = clean_text.strip()
                if clean_text.endswith("]"):
                    last_bracket = clean_text.rfind("[")
                    if last_bracket != -1:
                        clean_text = clean_text[:last_bracket].strip()
                    else:
                        break
                else:
                    break
            
            if clean_text and clean_text[-1] not in ".!?":
                is_fragment = 1
            if text.strip().endswith(" or.") or text.strip().endswith(" and.") or text.strip().endswith(" with."):
                is_fragment = 1
        fragments.append(is_fragment)
        
        # 5. Bad Grammar / Punctuation
        has_bad_grammar = 1 if bad_punct_pattern.search(text) else 0
        bad_grammar.append(has_bad_grammar)
        
        # 6. Unsupported
        is_unsupported = 1 if not citations or "I could not find enough evidence" in text or "I could not generate a high-quality answer" in text else 0
        unsupported.append(is_unsupported)
        
        # 7. Repetition
        sentences = re.split(r'(?<=[.!?])\s+', text)
        has_repetition = 0
        if len(sentences) > 2:
            seen = set()
            for s in sentences:
                s_clean = s.strip().lower()
                if s_clean in seen and len(s_clean) > 20:
                    has_repetition = 1
                    break
                seen.add(s_clean)
        repetition.append(has_repetition)
        
        # 8. Direct Answer (contains action term if query does)
        query_actions = [a for a in action_terms if a in query]
        is_direct = 1
        if query_actions and not any(a in text.lower() for a in query_actions):
            is_direct = 0
        direct_answer.append(is_direct)
        
        # 9. Wrong Domain Citation
        is_wrong_domain = 0
        if citations and gold_domain:
            if not any(gold_domain in c.lower() for c in citations):
                is_wrong_domain = 1
        wrong_domain.append(is_wrong_domain)
        
        # 10. Incoherent Multi-Evidence
        is_incoherent = 0
        if len(citations) > 1:
            # Check if domains match in citations
            citation_domains = [c.split(":")[0].strip("[ ").lower() for c in citations]
            if len(set(citation_domains)) > 1:
                is_incoherent = 1
        incoherent.append(is_incoherent)

        # 11. Rubric Score (0 to 1)
        # Weights: supported(0.2), not_too_long(0.1), no_fragment(0.1), no_bad_grammar(0.1), 
        # no_repetition(0.1), direct_answer(0.2), right_domain(0.2)
        score = 0
        if not is_unsupported: score += 0.2
        if not is_too_long: score += 0.1
        if not is_fragment: score += 0.1
        if not has_bad_grammar: score += 0.1
        if not has_repetition: score += 0.1
        if is_direct: score += 0.2
        if not is_wrong_domain: score += 0.2
        rubric_scores.append(score)

    return {
        "AnswerQualityScore": np.mean(rubric_scores),
        "DirectAnswerRate": np.mean(direct_answer),
        "FragmentRate": np.mean(fragments),
        "RepetitionRate": np.mean(repetition),
        "BadGrammarRate": np.mean(bad_grammar),
        "WrongDomainCitationRate": np.mean(wrong_domain),
        "IncoherentMultiEvidenceRate": np.mean(incoherent),
        "AnswerTooLongRate": np.mean(too_long),
        "AnswerHasCitationRate": np.mean(has_citations),
        "UnsupportedAnswerRate": np.mean(unsupported),
        "avg_answer_length": np.mean(lengths)
    }
