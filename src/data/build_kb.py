"""Build KB chunks from raw multidoc2dial documents.

Each document is chunked into overlapping fixed-size spans.
Output: data/processed/kb_chunks.jsonl
"""
import argparse
import os
import re
import sys
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 250        # characters per chunk
CHUNK_OVERLAP = 50      # character overlap
MIN_CHUNK_LEN = 40      # discard very short chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append({"text": chunk, "span_start": start, "span_end": end})
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_domain(doc_id: str, title: str = "") -> str:
    """Heuristic domain extraction from doc_id or title."""
    doc_lower = (doc_id + " " + title).lower()
    if "ssa" in doc_lower or "social security" in doc_lower:
        return "ssa"
    elif "dmv" in doc_lower or "vehicle" in doc_lower or "driver" in doc_lower:
        return "dmv"
    elif "studentaid" in doc_lower or "student" in doc_lower or "fafsa" in doc_lower:
        return "studentaid"
    elif "va" in doc_lower or "veteran" in doc_lower:
        return "va"
    else:
        # Fall back: use first path segment
        parts = re.split(r"[/_\-]", doc_id)
        return parts[0] if parts else "unknown"


def build_kb_chunks(raw_dir: str, out_dir: str, max_chunks: int = None) -> List[dict]:
    """Read raw documents and produce KB chunks."""
    ensure_dir(out_dir)
    doc_path = os.path.join(raw_dir, "documents.jsonl")
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Raw documents not found at {doc_path}. Run load_multidoc2dial.py first.")

    raw_docs = read_jsonl(doc_path)
    logger.info(f"Loaded {len(raw_docs)} raw document records.")

    kb_chunks = []
    for rec in raw_docs:
        # multidoc2dial doc structure has 'doc_id', 'title', 'domain', 'doc_text' or 'spans'
        doc_id = rec.get("doc_id") or rec.get("id", "unk")
        title  = rec.get("title", "")
        domain = rec.get("domain") or extract_domain(doc_id, title)

        # Try to get full text — multidoc2dial provides 'doc_text' at top level
        doc_text = rec.get("doc_text") or rec.get("text") or ""

        # If the doc has 'spans' dict, use that as pre-segmented chunks
        spans_dict = rec.get("spans")
        if spans_dict and isinstance(spans_dict, dict):
            for span_id, span_info in spans_dict.items():
                span_text = span_info.get("text_sp") or span_info.get("text", "")
                if not span_text or len(span_text.strip()) < MIN_CHUNK_LEN:
                    continue
                sec_id = span_info.get("id_sec") or span_info.get("sec_id", "")
                chunk_id = f"{doc_id}_{span_id}"
                kb_chunks.append({
                    "chunk_id":   chunk_id,
                    "doc_id":     doc_id,
                    "domain":     domain,
                    "title":      title,
                    "section_id": sec_id,
                    "span_start": span_info.get("start_sp", 0),
                    "span_end":   span_info.get("end_sp", len(span_text)),
                    "text":       span_text.strip(),
                })
        elif doc_text:
            for i, ch in enumerate(chunk_text(doc_text)):
                chunk_id = f"{doc_id}_span{i:04d}"
                kb_chunks.append({
                    "chunk_id":   chunk_id,
                    "doc_id":     doc_id,
                    "domain":     domain,
                    "title":      title,
                    "section_id": f"section_{i}",
                    "span_start": ch["span_start"],
                    "span_end":   ch["span_end"],
                    "text":       ch["text"],
                })

        if max_chunks and len(kb_chunks) >= max_chunks:
            logger.info(f"Reached max_kb_chunks={max_chunks}. Stopping early.")
            break

    if max_chunks:
        kb_chunks = kb_chunks[:max_chunks]

    out_path = os.path.join(out_dir, "kb_chunks.jsonl")
    write_jsonl(kb_chunks, out_path)
    logger.info(f"Built {len(kb_chunks)} KB chunks -> {out_path}")
    return kb_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KB chunks from raw documents")
    parser.add_argument("--raw_dir",    default="data/raw")
    parser.add_argument("--out_dir",    default="data/processed")
    parser.add_argument("--max_chunks", type=int, default=None)
    args = parser.parse_args()
    build_kb_chunks(args.raw_dir, args.out_dir, args.max_chunks)
