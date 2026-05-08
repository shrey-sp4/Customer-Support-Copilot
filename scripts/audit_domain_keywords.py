import json
import os
import sys
from collections import defaultdict

def audit_keywords(keywords_path, output_json, output_md):
    if not os.path.exists(keywords_path):
        print(f"Error: {keywords_path} not found.")
        return

    with open(keywords_path, 'r') as f:
        data = json.load(f)

    # 1. Keywords per domain
    domain_counts = {domain: len(keywords) for domain, keywords in data.items()}

    # 2. Duplicate keywords across domains
    keyword_to_domains = defaultdict(list)
    for domain, keywords in data.items():
        for kw in keywords:
            keyword_to_domains[kw.lower()].append(domain)

    duplicates = {kw: domains for kw, domains in keyword_to_domains.items() if len(domains) > 1}

    # 3. Suspicious generic keywords (examples)
    generic_candidates = [
        "need", "apply", "must", "information", "documents", "document", 
        "get", "receive", "help", "find", "use", "one", "see", "show",
        "current", "time", "application", "form", "online", "number",
        "status", "contact", "help", "amount", "make", "eligible"
    ]
    suspicious = [kw for kw in generic_candidates if kw in keyword_to_domains]

    # 4. Keyword collisions (subset of duplicates)
    collisions = duplicates

    # Prepare report
    report = {
        "domain_counts": domain_counts,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "suspicious_generics": suspicious,
        "all_keywords_by_domain": data
    }

    # Save JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)

    # Save Markdown
    with open(output_md, 'w') as f:
        f.write("# Domain Keyword Audit Report\n\n")
        f.write("## Domain Keyword Counts\n")
        for domain, count in domain_counts.items():
            f.write(f"- **{domain}**: {count} keywords\n")
        
        f.write("\n## Duplicate Keywords (Collisions)\n")
        f.write(f"Total duplicates: {len(duplicates)}\n\n")
        f.write("| Keyword | Domains |\n")
        f.write("| :--- | :--- |\n")
        for kw, domains in sorted(duplicates.items()):
            f.write(f"| {kw} | {', '.join(domains)} |\n")

        f.write("\n## Suspicious Generic Keywords\n")
        for kw in sorted(suspicious):
            f.write(f"- {kw} (found in: {', '.join(keyword_to_domains[kw])})\n")

    print(f"Audit report saved to {output_json} and {output_md}")

if __name__ == "__main__":
    audit_keywords(
        'data/processed/domain_keywords.json',
        'outputs/reports/domain_keyword_audit.json',
        'outputs/reports/domain_keyword_audit.md'
    )
