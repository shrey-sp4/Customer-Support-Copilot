import os
import json
import re
import numpy as np
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ESACalculator:
    def __init__(self, model_path: str, kb_path: str, device: str = "cpu"):
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)
        
        logger.info(f"Loading KB for ESA from {kb_path}")
        kb_chunks = read_jsonl(kb_path)
        self.chunk_by_id = {ch["chunk_id"]: ch.get("text", "") for ch in kb_chunks}
        
        # Regex for malformed check
        self.malformed_patterns = [
            r"^\s*$",                # Empty
            r"^\?+$",               # Only question marks
            r"^[a-z ]+\?$",         # Just a question back to user
            r"could not find enough", # Refusal
            r"i apologize",          # Apology fragment
        ]

    def calculate_esa(self, results: List[dict], label: str = "proposed") -> dict:
        esa_rows = []
        scores = []
        
        # Filter for answerable examples (where system decided to ANSWER)
        answer_samples = [r for r in results if r.get("decision") == "ANSWER"]
        
        if not answer_samples:
            return {"label": label, "ESA": 0.0, "n_samples": 0}

        for res in answer_samples:
            query = res.get("query", "")
            answer = res.get("final_answer", "")
            citations = res.get("citations", [])
            
            row = {
                "query": query,
                "answer": answer,
                "esa_pass": False,
                "esa_failure_reason": None,
                "query_citation_sim": 0.0,
                "answer_citation_sim": 0.0,
                "query_answer_sim": 0.0
            }

            # 1. Citation exists
            if not citations:
                row["esa_failure_reason"] = "missing_citation"
                esa_rows.append(row)
                continue

            # 2. Extract first citation text (ESA uses best evidence)
            # Format: [doc_id:chunk_id] or [doc_id:chunk_id 100-200]
            cit_match = re.search(r"\[([^:\]]+):\s?([^\]\s]+)", citations[0])
            if not cit_match:
                row["esa_failure_reason"] = "malformed_citation_format"
                esa_rows.append(row)
                continue
                
            cit_chunk_id = cit_match.group(2).strip()
            cit_text = self.chunk_by_id.get(cit_chunk_id)
            
            if not cit_text:
                row["esa_failure_reason"] = "citation_text_not_found"
                esa_rows.append(row)
                continue

            # 3. Compute Embeddings
            try:
                q_emb = self.model.encode(query, convert_to_tensor=True)
                a_emb = self.model.encode(answer, convert_to_tensor=True)
                c_emb = self.model.encode(cit_text, convert_to_tensor=True)
                
                q_c_sim = float(util.cos_sim(q_emb, c_emb))
                a_c_sim = float(util.cos_sim(a_emb, c_emb))
                q_a_sim = float(util.cos_sim(q_emb, a_emb))
                
                row["query_citation_sim"] = q_c_sim
                row["answer_citation_sim"] = a_c_sim
                row["query_answer_sim"] = q_a_sim
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                # Lexical fallback
                q_c_sim = self._lexical_overlap(query, cit_text)
                a_c_sim = self._lexical_overlap(answer, cit_text)
                q_a_sim = self._lexical_overlap(query, answer)
                row["query_citation_sim"] = q_c_sim
                row["answer_citation_sim"] = a_c_sim
                row["query_answer_sim"] = q_a_sim

            # 4. Apply ESA conditions
            pass_all = True
            
            # Condition 3: Relevance to query
            if q_c_sim < 0.35:
                row["esa_failure_reason"] = "citation_not_relevant_to_query"
                pass_all = False
            
            # Condition 4: Answer supported by citation
            elif a_c_sim < 0.40:
                row["esa_failure_reason"] = "answer_not_supported_by_citation"
                pass_all = False
                
            # Condition 5: Answer direct to query
            elif q_a_sim < 0.30:
                row["esa_failure_reason"] = "answer_not_direct_to_query"
                pass_all = False
                
            # Condition 6: Malformed / Fragmentary
            elif self._is_malformed(answer):
                row["esa_failure_reason"] = "malformed_answer"
                pass_all = False

            if pass_all:
                row["esa_pass"] = True
                scores.append(1)
            else:
                scores.append(0)
            
            esa_rows.append(row)

        final_esa = np.mean(scores) if scores else 0.0
        return {
            "label": label,
            "ESA": final_esa,
            "n_samples": len(answer_samples),
            "rows": esa_rows
        }

    def _is_malformed(self, text: str) -> bool:
        if not text or len(text.split()) < 5:
            return True
        for p in self.malformed_patterns:
            if re.search(p, text.lower()):
                return True
        return False

    def _lexical_overlap(self, text1: str, text2: str) -> float:
        t1 = set(re.findall(r"\w+", text1.lower()))
        t2 = set(re.findall(r"\w+", text2.lower()))
        if not t1 or not t2: return 0.0
        return len(t1.intersection(t2)) / len(t1)

def run_esa_audit(results_path: str, kb_path: str, model_path: str, output_path: str, label: str):
    with open(results_path, 'r') as f:
        results = json.load(f)
        if isinstance(results, dict) and "all_results" in results:
            results = results["all_results"]
            
    calc = ESACalculator(model_path, kb_path)
    report = calc.calculate_esa(results, label=label)
    
    # Save per-row breakdown
    rows = report.pop("rows")
    write_jsonl(rows, output_path.replace(".json", "_rows.jsonl"))
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"ESA for {label}: {report['ESA']:.4f}")

if __name__ == "__main__":
    import sys
    # Example usage: python src/evaluation/esa.py outputs/reports/proposed_results.json ...
    pass
