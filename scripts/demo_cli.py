"""scripts/demo_cli.py — Run the interactive or single-query demo."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Copilot CLI Demo")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device_str = getattr(cfg, "device", "auto")
    from src.utils.device import get_device
    device = get_device(device_str)

    data_dir   = cfg.get("data_dir", "data/processed")
    index_dir  = cfg.get("index_dir", "data/indexes")
    output_dir = cfg.get("output_dir", "outputs")

    # Load components
    from sentence_transformers import SentenceTransformer
    from src.retrieval.search_kb import load_searcher
    from src.routing.router import load_router
    from src.triage.predict import load_predictor
    from src.reranking.rerank import load_reranker
    from src.generation.generate import load_generator
    from src.preference.score_candidates import load_preference_scorer
    from src.tools.executor import ToolExecutor
    from src.utils.io import read_jsonl

    logger.info("Loading system components...")
    
    model_path = os.path.join(output_dir, "retriever")
    if not os.path.isdir(model_path):
        model_path = cfg.get("retriever_model", "sentence-transformers/all-MiniLM-L6-v2")
    encoder = SentenceTransformer(model_path, device=str(device))
    searcher = load_searcher(index_dir, os.path.join(data_dir, "kb_chunks.jsonl"), encoder)
    
    router = load_router(
        os.path.join(data_dir, "domain_centroids.json"),
        os.path.join(data_dir, "domain_keywords.json")
    )
    
    triage     = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker   = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    pref       = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
    
    gen_path   = os.path.join(output_dir, "generator")
    if not os.path.isdir(gen_path):
        gen_path = None if cfg.get("generator_epochs", 0) == 0 else cfg.get("generator_model", "google/flan-t5-small")
    generator  = load_generator(gen_path, device=device)

    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    executor = ToolExecutor(
        encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg
    )

    from src.serving.cli_demo import interactive_loop, print_tool_trace
    from rich.console import Console

    console = Console()

    if args.query:
        console.print(f"\n[bold yellow]User:[/bold yellow] {args.query}")
        result = executor.run(args.query)
        console.print("\n[bold magenta]--- Tool Trace ---[/bold magenta]")
        print_tool_trace(result.get("tool_trace", []))
        console.print("\n[bold cyan]--- Final Answer ---[/bold cyan]")
        console.print(result.get("final_answer", ""))
        if result.get("citations"):
            console.print("\n[bold]Citations:[/bold]")
            for c in result["citations"]:
                console.print(f"  [doc_id={c['doc_id']}, chunk_id={c['chunk_id']}, span={c['span_start']}-{c['span_end']}]")
    else:
        interactive_loop(executor)


if __name__ == "__main__":
    main()
