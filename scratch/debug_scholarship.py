import os
import sys
sys.path.insert(0, os.path.abspath(os.getcwd()))
from src.utils.config import load_config
from src.retrieval.search_kb import load_searcher
from sentence_transformers import SentenceTransformer

cfg = load_config('configs/smoke.yaml')
encoder = SentenceTransformer(cfg.get('retriever_model'))
searcher = load_searcher(cfg.get('index_dir'), 'data/processed/kb_chunks.jsonl', encoder)

query = "What documents do I need to have if I am applying for scholarship"
results = searcher.search(query, top_k=5, domain='studentaid')

print(f"Query: {query}")
for i, r in enumerate(results):
    print(f"\n--- Result {i+1} (Score: {r['score']:.4f}) ---")
    print(f"Doc ID: {r['doc_id']}")
    print(f"Text: {r['text']}")
