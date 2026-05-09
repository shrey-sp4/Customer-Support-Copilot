import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Convenience wrapper to build FAISS indexes")
    parser.add_argument("--mode", choices=["global", "raw", "domain"], required=True,
                        help="raw: Baseline-1 index, domain: Proposed indexes, global: Trained global index")
    parser.add_argument("--kb_path", default="data/processed/kb_chunks.jsonl")
    args = parser.parse_args()

    cmd = [
        "python", "-m", "src.retrieval.build_faiss",
        "--mode", args.mode,
        "--kb_path", args.kb_path
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
