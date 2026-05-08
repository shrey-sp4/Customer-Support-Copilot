import json
from collections import Counter
import os

def main():
    path = "data/processed/triage_train_balanced.jsonl"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    data = [json.loads(line) for line in open(path, encoding='utf-8')]
    
    counts = Counter(d.get("gold_triage") for d in data)
    domains = Counter(d.get("gold_domain") for d in data)
    sources = Counter(d.get("source") for d in data)
    
    report = {
        "label_distribution": dict(counts),
        "domain_distribution": dict(domains),
        "source_distribution": dict(sources)
    }
    
    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/triage_training_distribution.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n=== TRIAGE TRAINING DISTRIBUTION ===")
    print(f"Total Samples: {len(data)}")
    print("\nLabels:")
    for l, c in counts.items():
        print(f"  {l}: {c}")
    print("\nDomains:")
    for d, c in domains.items():
        print(f"  {d}: {c}")
    print("\nSources:")
    for s, c in sources.items():
        print(f"  {s}: {c}")

if __name__ == "__main__":
    main()
