import unittest
import os
import sys
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.routing.router import load_router
from src.tools.executor import validate_answerability
from src.generation.generate import verify_grounding

class TestPipeline(unittest.TestCase):
    
    def test_router_loading(self):
        # Verify router can load centroids
        data_dir = "data/processed"
        if os.path.exists(os.path.join(data_dir, "domain_centroids.json")):
            router = load_router(
                os.path.join(data_dir, "domain_centroids.json"),
                os.path.join(data_dir, "domain_keywords.json")
            )
            self.assertIsNotNone(router)
            self.assertTrue(len(router.domains) > 0)

    def test_grounding_verifier(self):
        # Test the sentence-level grounding logic
        passages = [{"text": "The Social Security office is open from 9am to 4pm."}]
        answer = "The Social Security office opens at 9am."
        is_grounded, reason = verify_grounding(answer, passages)
        self.assertTrue(is_grounded)
        
        answer_bad = "The Social Security office is in Paris."
        is_grounded, reason = verify_grounding(answer_bad, passages)
        self.assertFalse(is_grounded)

    def test_answerability_gate(self):
        # Test keyword-based answerability
        query = "how to renew health insurance"
        evidence = [{"text": "renew your health insurance plan by...", "doc_id": "doc1", "section_id": "sec1", "chunk_id": "c1", "score": 0.5}]
        res = validate_answerability(query, evidence, ["healthcare"])
        self.assertTrue(res["answerable"])

if __name__ == "__main__":
    unittest.main()
