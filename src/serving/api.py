"""Optional FastAPI serving demo."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys

# Simplified global executor placeholder
executor = None

app = FastAPI(title="Support Copilot API")

class QueryRequest(BaseModel):
    query: str
    history: str = ""

class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    span_start: int
    span_end: int

class QueryResponse(BaseModel):
    query: str
    decision: str
    confidence: float
    final_answer: str
    citations: List[Citation]
    latency_ms: float
    tool_trace: List[Dict[str, Any]]

@app.on_event("startup")
def startup_event():
    global executor
    try:
        from src.utils.config import load_config
        from src.utils.device import get_device
        from sentence_transformers import SentenceTransformer
        from src.retrieval.search_kb import load_searcher
        from src.routing.router import load_router
        from src.triage.predict import load_predictor
        from src.reranking.rerank import load_reranker
        from src.generation.generate import load_generator
        from src.preference.score_candidates import load_preference_scorer
        from src.tools.executor import ToolExecutor
        from src.utils.io import read_jsonl

        cfg = load_config("configs/full_local.yaml")
        device = get_device("auto")

        data_dir   = cfg.get("data_dir", "data/processed")
        index_dir  = cfg.get("index_dir", "data/indexes")
        output_dir = cfg.get("output_dir", "outputs")

        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
        searcher = load_searcher(index_dir, os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
        router = load_router(os.path.join(data_dir, "domain_centroids.json"), os.path.join(data_dir, "domain_keywords.json"))
        triage = load_predictor(os.path.join(output_dir, "triage"), device=device)
        reranker = load_reranker(os.path.join(output_dir, "reranker"), device=device)
        pref = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
        generator = load_generator(None, device=device)

        kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
        chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

        executor = ToolExecutor(encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load models at startup: {e}")

@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    if not executor:
        return {"error": "System not initialized"}
    
    result = executor.run(req.query, history=req.history)
    return result
