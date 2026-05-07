"""run_all.py — Orchestrates the full training and evaluation pipeline."""
import argparse
import os
import subprocess
import sys

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"Executing: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        print(f"Failed command: {cmd}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full Support Copilot pipeline")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Prepare Data
    run_command(f"{sys.executable} scripts/prepare_data.py --config {args.config}", "Prepare Data")
    
    # 2. Train Retriever
    run_command(f"{sys.executable} scripts/train_retriever.py --config {args.config}", "Train Retriever")
    
    # 3. Build Index
    run_command(f"{sys.executable} scripts/build_index.py --config {args.config}", "Build FAISS Index")
    
    # 4. Train Reranker
    run_command(f"{sys.executable} scripts/train_reranker.py --config {args.config}", "Train Reranker")
    
    # 5. Train Triage Model
    run_command(f"{sys.executable} scripts/train_triage.py --config {args.config}", "Train Triage Model")
    
    # 6. Train Preference Ranker
    run_command(f"{sys.executable} scripts/train_preference.py --config {args.config}", "Train Preference Ranker")
    
    # 7. Evaluate
    run_command(f"{sys.executable} scripts/evaluate.py --config {args.config}", "Evaluate End-to-End")

    print(f"\n{'='*60}")
    print("Full pipeline execution completed successfully!")
    print(f"Check the outputs/reports directory for evaluation metrics.")
    print("Run `python scripts/demo_cli.py --config configs/smoke.yaml` to try the interactive demo.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
