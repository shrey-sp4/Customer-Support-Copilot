"""Fine-tune sentence-transformer retriever on MultiDoc2Dial query-passage pairs.

Uses MultipleNegativesRankingLoss (in-batch negatives) + hard negatives.
Saves fine-tuned model to outputs/retriever/
"""
import argparse
import os
import sys
import random
from typing import List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device, use_fp16

logger = get_logger(__name__)


def build_train_examples(records: List[dict]):
    """Build InputExample pairs for MNRL training."""
    from sentence_transformers import InputExample
    examples = []
    for rec in records:
        examples.append(InputExample(
            texts=[rec["query"], rec["pos_text"]],
        ))
        # Also add query -> negative as a hard negative pair (reversed label, not needed in MNRL
        # but we add positive-negative pairs for margin loss via TripletLoss if desired)
    return examples


def train_retriever(
    train_path: str,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 1,
    batch_size: int = 8,
    max_samples: int = None,
    fp16: bool = True,
    seed: int = 42,
):
    """Fine-tune the retriever using MultipleNegativesRankingLoss."""
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    set_seed(seed)
    device = get_device("auto")
    ensure_dir(output_dir)

    # Load training data
    records = read_jsonl(train_path)
    if max_samples:
        random.shuffle(records)
        records = records[:max_samples]
    logger.info(f"Retriever train samples: {len(records)}")

    # Load model
    logger.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name, device=str(device))

    # Build examples
    examples = build_train_examples(records)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)

    # Loss
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Train
    warmup_steps = max(1, len(loader) // 10)
    logger.info(f"Training for {epochs} epochs, {len(loader)} steps/epoch, warmup={warmup_steps}")

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
        use_amp=use_fp16(fp16, device),
    )

    logger.info(f"Retriever saved to {output_dir}")
    return model


def evaluate_retriever(model, records: List[dict], top_k: int = 5):
    """Compute Recall@k and MRR@10 on a held-out set."""
    logger.info("Evaluating retriever on eval set ...")
    queries   = {r["query_id"]: r["query"]    for r in records}
    corpus    = {r["pos_chunk_id"]: r["pos_text"] for r in records}
    relevant  = {r["query_id"]: {r["pos_chunk_id"]} for r in records}

    query_embs  = model.encode(list(queries.values()),  convert_to_numpy=True, show_progress_bar=False)
    corpus_embs = model.encode(list(corpus.values()),   convert_to_numpy=True, show_progress_bar=False)

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    query_ids   = list(queries.keys())
    corpus_ids  = list(corpus.keys())
    sim_matrix  = cosine_similarity(query_embs, corpus_embs)

    recall_at_1 = recall_at_5 = mrr = ndcg = evidence_hit = 0.0
    n = len(query_ids)

    for i, qid in enumerate(query_ids):
        ranked = np.argsort(sim_matrix[i])[::-1]
        gold_cids = relevant.get(qid, set())
        hit_1 = any(corpus_ids[j] in gold_cids for j in ranked[:1])
        hit_5 = any(corpus_ids[j] in gold_cids for j in ranked[:top_k])
        recall_at_1 += float(hit_1)
        recall_at_5 += float(hit_5)
        evidence_hit += float(hit_5)
        for rank, j in enumerate(ranked[:10], start=1):
            if corpus_ids[j] in gold_cids:
                mrr += 1.0 / rank
                break

    metrics = {
        "Recall@1":      recall_at_1 / n,
        "Recall@5":      recall_at_5 / n,
        "MRR@10":        mrr / n,
        "EvidenceHit@5": evidence_hit / n,
    }
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retriever")
    parser.add_argument("--train_path",  default="data/processed/retriever_train.jsonl")
    parser.add_argument("--output_dir",  default="outputs/retriever")
    parser.add_argument("--model_name",  default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--fp16",        action="store_true", default=True)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    train_retriever(
        train_path=args.train_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        fp16=args.fp16,
        seed=args.seed,
    )
