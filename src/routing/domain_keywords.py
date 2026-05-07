"""Compute TF-IDF domain keywords for each KB domain.

Output: data/processed/domain_keywords.json
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_json, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Common English stopwords (no NLTK dependency required)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "if",
    "as", "so", "that", "this", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "which", "who", "what", "when", "where", "how", "all",
    "any", "each", "more", "also", "not", "no", "up", "out", "about",
    "than", "then", "there", "their", "into", "through", "during", "before",
    "after", "between", "under", "while", "own", "same", "other", "such",
}


def tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer."""
    return [w.lower() for w in re.findall(r"[a-z][a-z'-]*[a-z]", text.lower()) if len(w) > 2]


def compute_domain_keywords(
    kb_chunks: List[dict],
    top_n: int = 50,
    min_df: int = 2,
) -> Dict[str, List[str]]:
    """Compute per-domain exclusive top keywords using TF-IDF-like domain exclusivity."""

    # Collect per-domain token counts and doc frequencies
    domain_tf: Dict[str, Counter] = defaultdict(Counter)
    domain_df: Dict[str, Counter] = defaultdict(Counter)  # token in how many *chunks* per domain

    for chunk in kb_chunks:
        domain = chunk.get("domain", "unknown")
        tokens = tokenize(chunk.get("text", ""))
        # filter stopwords
        tokens = [t for t in tokens if t not in STOPWORDS]
        domain_tf[domain].update(tokens)
        domain_df[domain].update(set(tokens))  # unique per chunk

    # Global df across all domains
    global_df: Counter = Counter()
    for dom_df in domain_df.values():
        global_df.update(dom_df.keys())

    n_domains = len(domain_tf)

    # Compute domain exclusivity score for each token:
    #   score = tf_domain / global_df (rewards tokens frequent in one domain but not all)
    domain_keywords: Dict[str, List[str]] = {}

    for domain, tf in domain_tf.items():
        scores = {}
        for token, freq in tf.items():
            if domain_df[domain][token] < min_df:
                continue
            gdf = global_df.get(token, 1)
            # Exclusivity: penalize tokens that appear in many domains
            domain_count = sum(1 for d, df in domain_df.items() if token in df)
            exclusivity = 1.0 / domain_count if domain_count > 0 else 0.0
            scores[token] = freq * exclusivity

        top_tokens = sorted(scores, key=lambda t: scores[t], reverse=True)[:top_n]
        domain_keywords[domain] = top_tokens
        logger.info(f"Domain '{domain}': top keywords = {top_tokens[:10]}")

    return domain_keywords


def main(args):
    ensure_dir(os.path.dirname(args.out_path) if os.path.dirname(args.out_path) else ".")
    kb_path = os.path.join(args.data_dir, "kb_chunks.jsonl")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"KB chunks not found at {kb_path}.")
    kb_chunks = read_jsonl(kb_path)
    if args.max_chunks:
        kb_chunks = kb_chunks[:args.max_chunks]
    kw = compute_domain_keywords(kb_chunks, top_n=args.top_n)
    write_json(kw, args.out_path)
    logger.info(f"Domain keywords saved to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-domain TF-IDF keywords")
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--out_path",   default="data/processed/domain_keywords.json")
    parser.add_argument("--top_n",      type=int, default=50)
    parser.add_argument("--max_chunks", type=int, default=None)
    args = parser.parse_args()
    main(args)
