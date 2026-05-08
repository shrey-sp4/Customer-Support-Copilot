import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import read_jsonl, ensure_dir

def run_sanity_check():
    data_dir = "data/processed"
    output_dir = "outputs/reports"
    ensure_dir(output_dir)
    
    eval_files = {
        "Natural (1000)": "eval_md2d_natural_1000.jsonl",
        "Balanced (300)": "eval_workflow_balanced_300.jsonl",
        "Robustness (150)": "eval_robustness_150.jsonl"
    }
    
    report = "# Evaluation Sanity Check Report\n\n"
    report += "## 1. Label Distributions\n\n"
    report += "| Set | Total | ANSWER | TICKET | REJECT |\n"
    report += "| --- | --- | --- | --- | --- |\n"
    
    label_stats = {}
    
    for name, filename in eval_files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        
        data = read_jsonl(path)
        labels = [d.get("gold_triage", d.get("gold_label", "ANSWER")) for d in data]
        counts = Counter(labels)
        
        report += f"| {name} | {len(data)} | {counts.get('ANSWER', 0)} | {counts.get('TICKET', 0)} | {counts.get('REJECT', 0)} |\n"
        label_stats[name] = {"total": len(data), "counts": counts}
        
    report += "\n## 2. Baseline Accuracy Validation\n\n"
    report += "Expected accuracy if a system always predicts **ANSWER**:\n\n"
    report += "| Set | Expected Baseline-1 Accuracy |\n"
    report += "| --- | --- |\n"
    
    for name, stats in label_stats.items():
        expected_acc = stats["counts"].get("ANSWER", 0) / stats["total"]
        report += f"| {name} | {expected_acc:.3f} |\n"
        
    report += "\n> [!IMPORTANT]\n"
    report += "> If the reported Baseline-1 accuracy in the main evaluation does not match the 'Expected' values above, there is a logging or data-loading mismatch.\n\n"
    
    # Check if a results file exists to do a confusion matrix
    results_path = "outputs/reports/final_robust_metrics.csv"
    if os.path.exists(results_path):
        report += "## 3. Current Performance Snapshot\n\n"
        df = pd.read_csv(results_path)
        report += df.to_markdown(index=False)
        report += "\n"

    # Write report
    report_path = os.path.join(output_dir, "eval_sanity_check.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Sanity check report generated at {report_path}")

if __name__ == "__main__":
    run_sanity_check()
