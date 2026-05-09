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
    parser.add_argument("--debug", action="store_true", help="Show raw internal INFO logs")
    parser.add_argument("--hide-trace", action="store_true", help="Hide the pretty Tool Trace boxes")
    args = parser.parse_args()

    # Suppress internal pipeline logs unless --debug is passed
    import logging
    if not args.debug:
        logging.getLogger("src").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

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
    searcher = load_searcher(
        index_dir, 
        os.path.join(data_dir, "kb_chunks.jsonl"), 
        encoder,
        domain_indexes_dir="data/indexes_by_domain"
    )
    
    router = load_router(
        os.path.join(data_dir, "domain_centroids.json"),
        os.path.join(data_dir, "domain_keywords.json")
    )
    
    triage     = load_predictor(os.path.join(output_dir, "triage"), device=device)
    reranker   = load_reranker(os.path.join(output_dir, "reranker"), device=device)
    pref       = load_preference_scorer(os.path.join(output_dir, "preference"), device=device)
    
    gen_path   = os.path.join(output_dir, "generator")
    if not os.path.isdir(gen_path):
        gen_path = None
    generator  = load_generator(gen_path, device=device, cfg=cfg)

    kb_chunks = read_jsonl(os.path.join(data_dir, "kb_chunks.jsonl"))
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    executor = ToolExecutor(
        encoder, searcher, router, triage, reranker, generator, pref, chunk_by_id, cfg
    )

    from src.serving.cli_demo import interactive_loop, print_tool_trace
    from rich.console import Console

    console = Console()
    gen_mode = "LLM" if executor.generator is not None else "TEMPLATE FALLBACK"
    console.print(f"[bold blue]System initialized. Generator mode: {gen_mode}[/bold blue]")

    if args.query:
        console.print(f"\n[bold yellow]User:[/bold yellow] {args.query}")
        result = executor.run(args.query)
        if not args.hide_trace:
            console.print("\n[bold magenta]--- Tool Trace ---[/bold magenta]")
            print_tool_trace(result.get("tool_trace", []))
        console.print("\n[bold cyan]--- Final Answer ---[/bold cyan]")
        console.print(result.get("final_answer", ""))
        console.print(f"\n[dim]Latency: {result.get('latency_ms', 0):.0f} ms | Decision: {result.get('decision')}[/dim]")
    else:
        interactive_loop(executor, show_trace=not args.hide_trace)


if __name__ == "__main__":
    main()
