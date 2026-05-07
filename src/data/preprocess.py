"""Preprocess multidoc2dial dialogues into structured turn objects.

Output: data/processed/dialogue_turns.jsonl
        data/processed/retriever_train.jsonl
        data/processed/reranker_train.jsonl
        data/processed/eval_set.jsonl
"""
import argparse
import os
import sys
import random
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_dial_turns(raw_dial_path: str, kb_chunks: List[dict], max_samples: int = None) -> List[dict]:
    """Parse SQuAD-formatted multidoc2dial records into flat query-turn objects."""
    records = read_jsonl(raw_dial_path)
    turns = []
    
    # Group chunks by doc_id for fast matching
    chunks_by_doc = {}
    for ch in kb_chunks:
        chunks_by_doc.setdefault(ch["doc_id"], []).append(ch)

    for i, rec in enumerate(records):
        question_text = rec.get("question", "")
        parts = question_text.split("[SEP]")
        query = parts[0].strip()
        history = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        
        gold_doc_id = rec.get("title", "")
        gold_domain = rec.get("domain", "")
        
        # multidoc2dial HF format stores answer in 'answers' dict
        answers = rec.get("answers", {})
        answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
        
        gold_chunk_id = ""
        # Find which chunk contains the answer_text
        if gold_doc_id and answer_text and gold_doc_id in chunks_by_doc:
            # simple substring match
            for ch in chunks_by_doc[gold_doc_id]:
                # If answer text is inside chunk text, or vice versa (chunk inside answer)
                if answer_text in ch["text"] or ch["text"] in answer_text:
                    gold_chunk_id = ch["chunk_id"]
                    break

        turns.append({
            "query_id":       f"q{i:06d}",
            "dial_id":        rec.get("id", f"unk_{i}"),
            "turn_idx":       0,
            "query":          query,
            "history":        history[-300:],
            "gold_doc_id":    gold_doc_id,
            "gold_chunk_id":  gold_chunk_id,
            "gold_domain":    gold_domain,
            "gold_triage":    "ANSWER" if gold_chunk_id else "TICKET",
        })

        if max_samples and len(turns) >= max_samples:
            return turns

    logger.info(f"Extracted {len(turns)} dialogue turns.")
    return turns


def make_retriever_train(turns: List[dict], kb_chunks: List[dict], max_samples: int = None) -> List[dict]:
    """Create (query, positive_chunk, negative_chunk) triples for retriever training."""
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    # Group chunks by domain for in-domain negatives
    domain_chunks: Dict[str, List[dict]] = {}
    for ch in kb_chunks:
        domain_chunks.setdefault(ch["domain"], []).append(ch)

    records = []
    for turn in turns:
        if not turn["gold_chunk_id"] or turn["gold_chunk_id"] not in chunk_by_id:
            continue
        pos_chunk = chunk_by_id[turn["gold_chunk_id"]]
        domain = pos_chunk.get("domain", "")
        # Pick random negative from same domain (or any domain if not enough)
        neg_pool = [c for c in domain_chunks.get(domain, kb_chunks) if c["chunk_id"] != pos_chunk["chunk_id"]]
        if not neg_pool:
            neg_pool = [c for c in kb_chunks if c["chunk_id"] != pos_chunk["chunk_id"]]
        if not neg_pool:
            continue
        neg_chunk = random.choice(neg_pool)
        records.append({
            "query":          turn["query"],
            "query_id":       turn["query_id"],
            "pos_chunk_id":   pos_chunk["chunk_id"],
            "pos_text":       pos_chunk["text"],
            "neg_chunk_id":   neg_chunk["chunk_id"],
            "neg_text":       neg_chunk["text"],
            "domain":         domain,
        })
        if max_samples and len(records) >= max_samples:
            break

    logger.info(f"Created {len(records)} retriever training pairs.")
    return records


def make_reranker_train(turns: List[dict], kb_chunks: List[dict], max_samples: int = None) -> List[dict]:
    """Create (query, passage, label) records for reranker cross-encoder."""
    chunk_by_id = {ch["chunk_id"]: ch for ch in kb_chunks}
    records = []

    for turn in turns:
        if not turn["gold_chunk_id"] or turn["gold_chunk_id"] not in chunk_by_id:
            continue
        pos_chunk = chunk_by_id[turn["gold_chunk_id"]]
        # Positive
        records.append({
            "query":    turn["query"],
            "query_id": turn["query_id"],
            "chunk_id": pos_chunk["chunk_id"],
            "text":     pos_chunk["text"],
            "label":    1,
        })
        # Hard negative from same domain
        domain = pos_chunk.get("domain", "")
        neg_pool = [c for c in kb_chunks
                    if c["chunk_id"] != pos_chunk["chunk_id"] and c.get("domain") == domain]
        if not neg_pool:
            neg_pool = [c for c in kb_chunks if c["chunk_id"] != pos_chunk["chunk_id"]]
        if neg_pool:
            neg_chunk = random.choice(neg_pool[:min(20, len(neg_pool))])
            records.append({
                "query":    turn["query"],
                "query_id": turn["query_id"],
                "chunk_id": neg_chunk["chunk_id"],
                "text":     neg_chunk["text"],
                "label":    0,
            })

        if max_samples and len(records) >= max_samples * 2:
            break

    logger.info(f"Created {len(records)} reranker training samples.")
    return records


def make_eval_set(turns: List[dict], max_samples: int = None) -> List[dict]:
    """Create evaluation set from turns with gold labels."""
    eval_turns = [t for t in turns if t["gold_chunk_id"]]
    if max_samples:
        eval_turns = eval_turns[:max_samples]
    logger.info(f"Eval set: {len(eval_turns)} turns.")
    return eval_turns


def main(args):
    ensure_dir(args.out_dir)

    # Load KB chunks
    kb_path = os.path.join(args.out_dir, "kb_chunks.jsonl")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"KB chunks not found at {kb_path}. Run build_kb.py first.")
    kb_chunks = read_jsonl(kb_path)
    logger.info(f"Loaded {len(kb_chunks)} KB chunks.")

    # Parse dialogues from all available raw splits
    all_turns = []
    for split in ["train", "validation"]:
        raw_path = os.path.join(args.raw_dir, f"dialogues_{split}.jsonl")
        if os.path.exists(raw_path):
            t = parse_dial_turns(raw_path, kb_chunks, max_samples=args.max_train_samples)
            all_turns.extend(t)
        else:
            logger.warning(f"No raw dialogue file found for split '{split}'.")

    write_jsonl(all_turns, os.path.join(args.out_dir, "dialogue_turns.jsonl"))

    # Train/eval split
    random.seed(42)
    random.shuffle(all_turns)
    n_eval = min(args.max_eval_samples or 500, max(100, len(all_turns) // 10))
    eval_turns  = all_turns[:n_eval]
    train_turns = all_turns[n_eval:]

    # Retriever training data
    ret_train = make_retriever_train(train_turns, kb_chunks, max_samples=args.max_train_samples)
    write_jsonl(ret_train, os.path.join(args.out_dir, "retriever_train.jsonl"))

    # Reranker training data
    rer_train = make_reranker_train(train_turns, kb_chunks, max_samples=args.max_train_samples)
    write_jsonl(rer_train, os.path.join(args.out_dir, "reranker_train.jsonl"))

    # Eval set
    eval_set = make_eval_set(eval_turns, max_samples=args.max_eval_samples)
    write_jsonl(eval_set, os.path.join(args.out_dir, "eval_set.jsonl"))

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess multidoc2dial dialogues")
    parser.add_argument("--raw_dir",           default="data/raw")
    parser.add_argument("--out_dir",           default="data/processed")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples",  type=int, default=None)
    args = parser.parse_args()
    main(args)
