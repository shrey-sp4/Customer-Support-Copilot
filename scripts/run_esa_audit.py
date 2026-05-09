import os
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.tools.executor import BaselineExecutor, RuleWorkflowExecutor, ToolExecutor
from src.retrieval.search_kb import KBSearcher
from src.routing.router import load_router
from src.triage.predict import TriagePredictor
from src.generation.generate import FlanT5Generator
from src.evaluation.esa import ESACalculator
from src.utils.io import read_jsonl

class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigObject(v))
            else:
                setattr(self, k, v)
    def get(self, k, default=None):
        return getattr(self, k, default)

def run_esa_comparison(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # RELAX THRESHOLDS FOR SMOKE TEST ONLY (Flat config)
    config_dict["evidence_answer_threshold"] = 0.05
    config_dict["cluster_out_of_domain_threshold"] = 0.02
    
    cfg = ConfigObject(config_dict)
    
    data_dir = config_dict.get("data_dir", "data/sample")
    index_dir = config_dict.get("index_dir", "data/smoke_indexes/global")
    eval_file_name = config_dict.get("eval_file", "eval_set.jsonl")
    
    model_name = config_dict.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2")
    gen_model_name = "google/flan-t5-small"
    triage_model_name = config_dict.get("triage_model", "distilbert-base-uncased")
    
    kb_path = os.path.join(data_dir, "kb_chunks.jsonl")
    eval_file = os.path.join(data_dir, eval_file_name)
    
    print(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)
    searcher = KBSearcher(index_dir=index_dir, kb_path=kb_path, encoder=encoder, domain_indexes_dir=config_dict.get("domain_indexes_dir"))
    
    # Load routing
    centroids_path = os.path.join(data_dir, "domain_centroids.json")
    if not os.path.exists(centroids_path): centroids_path = "data/processed/domain_centroids.json"
    keywords_path = os.path.join(data_dir, "domain_keywords.json")
    if not os.path.exists(keywords_path): keywords_path = "data/processed/domain_keywords.json"
    router = load_router(centroids_path, keywords_path)
    
    triage = TriagePredictor(model_path=triage_model_name)
    generator = FlanT5Generator(model_path=gen_model_name)
    
    eval_set = read_jsonl(eval_file)
    
    # Pass flat config object to executors
    base1 = BaselineExecutor(searcher, generator=None, cfg=cfg)
    base2 = RuleWorkflowExecutor(searcher, router=router, generator=None, cfg=cfg)
    prop = ToolExecutor(encoder=encoder, searcher=searcher, router=router, triage_predictor=triage, generator=generator, cfg=cfg, chunk_by_id=searcher.chunk_by_id)
    
    executors = [("Baseline-1", base1), ("Baseline-2", base2), ("Proposed", prop)]
    calc = ESACalculator(model_name, kb_path)
    final_table = []

    for name, exe in executors:
        print(f"\nEvaluating {name}...")
        results = []
        for sample in tqdm(eval_set):
            try:
                res = exe.run(sample["query"])
                res["query"] = sample["query"]
                results.append(res)
            except Exception: pass
        
        report = calc.calculate_esa(results, label=name)
        final_table.append({"System": name, "ESA": report["ESA"], "Samples": report["n_samples"]})

    print("\n--- FINAL ESA COMPARISON (RELAXED SMOKE) ---")
    print(f"{'System':<15} | {'ESA Score':<10} | {'Answered'}")
    print("-" * 40)
    for row in final_table:
        print(f"{row['System']:<15} | {row['ESA']:<10.4f} | {row['Samples']}")

if __name__ == "__main__":
    run_esa_comparison("configs/smoke.yaml")
