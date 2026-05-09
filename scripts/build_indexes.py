import argparse
import subprocess
import sys


def main():
    # Note: These defaults are mirrored in configs/smoke.yaml and can be overridden via CLI.
    # Dataset-specific canonicalization for the four support domains.
    parser = argparse.ArgumentParser(
        description="Convenience wrapper to build FAISS indexes"
    )

    parser.add_argument(
        "--mode",
        choices=["global", "raw", "domain"],
        required=True,
        help=(
            "raw: Baseline-1 raw MiniLM index, "
            "global: trained/global index, "
            "domain: proposed per-domain indexes"
        ),
    )

    parser.add_argument(
        "--kb_path",
        default="data/processed/kb_chunks.jsonl",
        help="Path to processed KB chunks JSONL file.",
    )

    parser.add_argument(
        "--index_dir",
        default="data/indexes",
        help="Output directory for global FAISS index.",
    )

    parser.add_argument(
        "--raw_index_dir",
        default="data/indexes_raw",
        help="Output directory for raw baseline FAISS index.",
    )

    parser.add_argument(
        "--domain_indexes_dir",
        default="data/indexes_by_domain",
        help="Output directory for per-domain FAISS indexes.",
    )

    parser.add_argument(
        "--model_path",
        default="outputs/retriever",
        help="Fine-tuned retriever path or Hugging Face model ID for global/domain indexes.",
    )

    parser.add_argument(
        "--raw_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Raw retriever model for Baseline-1.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Optional max number of chunks to index.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, cuda, etc.",
    )

    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.retrieval.build_faiss",
        "--mode",
        args.mode,
        "--kb_path",
        args.kb_path,
        "--index_dir",
        args.index_dir,
        "--raw_index_dir",
        args.raw_index_dir,
        "--domain_indexes_dir",
        args.domain_indexes_dir,
        "--model_path",
        args.model_path,
        "--raw_model",
        args.raw_model,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]

    if args.max_chunks is not None:
        cmd.extend(["--max_chunks", str(args.max_chunks)])

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()