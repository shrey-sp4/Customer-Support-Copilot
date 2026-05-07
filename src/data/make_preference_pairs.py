"""Create preference pairs for rubric/preference alignment training.

Each pair has a preferred answer (has citation, correct triage, grounded) and
a rejected answer (missing citation, wrong triage, hallucinated, or verbose).

Output: data/processed/preference_pairs.jsonl
"""
import argparse
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_preferred_answer(query: str, chunk: dict) -> str:
    doc_id    = chunk.get("doc_id", "unk")
    chunk_id  = chunk.get("chunk_id", "unk")
    span_s    = chunk.get("span_start", 0)
    span_e    = chunk.get("span_end", 0)
    evidence  = chunk.get("text", "")[:200]
    return (
        f"Based on the knowledge base: {evidence} "
        f"[doc_id={doc_id}, chunk_id={chunk_id}, span={span_s}-{span_e}]"
    )


def make_rejected_variants(query: str, chunk: dict) -> List[str]:
    """Generate several rejected candidate answer templates."""
    evidence = chunk.get("text", "")[:200]
    variants = [
        # No citation
        f"Based on the information available: {evidence}",
        # Hallucinated answer
        f"You can definitely do this easily online. Just visit the official website and follow the steps.",
        # Wrong triage — answers when should reject
        f"I can help with that. {evidence}",
        # Verbose and unhelpful
        (
            f"That is a great question! There are many aspects to consider when thinking about your query. "
            f"First, let me explain the background. The system works in a variety of ways. "
            f"You should consult the documentation thoroughly before proceeding."
        ),
        # Missing evidence
        f"Please check the official portal for more information about your query.",
    ]
    return variants


def make_preference_pairs(
    dialogue_turns: List[dict],
    kb_chunks: List[dict],
    out_dir: str,
    max_samples: int = None,
) -> List[dict]:
    ensure_dir(out_dir)
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}

    pairs = []
    answer_turns = [t for t in dialogue_turns if t.get("gold_triage") == "ANSWER" and t.get("gold_chunk_id")]

    for t in answer_turns:
        cid = t.get("gold_chunk_id", "")
        if cid not in chunk_by_id:
            continue
        chunk = chunk_by_id[cid]
        preferred = make_preferred_answer(t["query"], chunk)
        rejected_list = make_rejected_variants(t["query"], chunk)
        for rej in rejected_list:
            pairs.append({
                "query_id":         t["query_id"],
                "query":            t["query"],
                "preferred_answer": preferred,
                "rejected_answer":  rej,
                "gold_triage":      t["gold_triage"],
                "gold_doc_id":      t.get("gold_doc_id", ""),
                "gold_chunk_id":    cid,
            })
        if max_samples and len(pairs) >= max_samples:
            break

    # Shuffle and limit
    random.shuffle(pairs)
    if max_samples:
        pairs = pairs[:max_samples]

    out_path = os.path.join(out_dir, "preference_pairs.jsonl")
    write_jsonl(pairs, out_path)
    logger.info(f"Created {len(pairs)} preference pairs -> {out_path}")
    return pairs


def main(args):
    dial_path = os.path.join(args.out_dir, "dialogue_turns.jsonl")
    kb_path   = os.path.join(args.out_dir, "kb_chunks.jsonl")
    if not os.path.exists(dial_path):
        raise FileNotFoundError(f"dialogue_turns.jsonl not found.")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"kb_chunks.jsonl not found.")

    turns  = read_jsonl(dial_path)
    chunks = read_jsonl(kb_path)
    make_preference_pairs(turns, chunks, args.out_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create preference pairs")
    parser.add_argument("--out_dir",     default="data/processed")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
