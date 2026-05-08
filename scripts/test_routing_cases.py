import argparse
import os
import sys
import torch
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.retrieval.search_kb import load_searcher
from src.routing.router import load_router
from src.triage.predict import load_predictor
from src.reranking.rerank import load_reranker
from src.generation.generate import load_generator
from src.tools.executor import ToolExecutor
from src.utils.io import read_jsonl

def test_cases(executor):
    cases = [
        # StudentAid
        ("how to get a scholarship", "studentaid"),
        ("what documents do I need for a scholarship application", "studentaid"),
        ("how do I apply for FAFSA", "studentaid"),
        ("can I get student aid for college", "studentaid"),
        
        # SSA
        ("how do I get a social security card", "ssa"),
        ("what documents do I need for SSI", "ssa"),
        ("how do I apply for disability benefits", "ssa"),
        
        # VA
        ("how do I update VA direct deposit", "va"),
        ("how do I apply for VA pension", "va"),
        ("can I use VA health care with private insurance", "va"),
        
        # DMV
        ("how do I renew my driver's license", "dmv"),
        ("what documents do I need for vehicle registration", "dmv"),
        
        # REJECT
        ("what films do you recommend", "REJECT"),
        ("how do I bake a cake", "REJECT"),
        ("why are we here", "REJECT"),
    ]
    
    print("\n" + "="*80)
    print(f"{'Query':<50} | {'Expected':<10} | {'Actual':<10} | {'Decision':<10}")
    print("-" * 87)
    
    passed = 0
    for query, expected in cases:
        res = executor.run(query)
        actual_domain = res.get("tool_trace", [])[0].get("top_domain", "none")
        decision = res["decision"]
        
        if expected == "REJECT":
            match = (decision == "REJECT")
        else:
            match = (actual_domain == expected and decision != "REJECT")
            
        status = "PASS" if match else "FAIL"
        if match: passed += 1
        
        print(f"{query[:50]:<50} | {expected:<10} | {actual_domain:<10} | {decision:<10} -> {status}")

    print("="*80)
    print(f"Passed: {passed}/{len(cases)}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_local.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    data_dir = cfg.get("data_dir", "data/processed")
    output_dir = cfg.get("output_dir", "outputs")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2"), device=device)
    searcher = load_searcher(cfg.get("index_dir", "data/indexes"), os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
    triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    
    gen_path = os.path.join(output_dir, "generator")
    if not os.path.isdir(gen_path): gen_path = None
    generator = load_generator(gen_path, device=device, cfg=cfg)
    
    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    
    executor = ToolExecutor(encoder, searcher, router, triage, reranker, generator, None, chunk_by_id, cfg)
    
    test_cases(executor)

if __name__ == "__main__":
    main()
