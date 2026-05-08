import json
import random
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import read_jsonl, write_jsonl, ensure_dir

def generate_ticket_queries():
    templates = [
        "Can you check my {domain} status?",
        "Where is my {domain} application?",
        "Why was my {domain} payment delayed?",
        "What is the status of my {domain} claim?",
        "Can you update my {domain} direct deposit information?",
        "Can you change my address on my {domain} account?",
        "Can you submit this {domain} form for me?",
        "Can you guarantee that my {domain} application will be approved?",
        "Can you tell me my exact {domain} benefit amount?",
        "I haven't received my {domain} check this month, why?",
        "How do I track my {domain} file?",
        "Can you reset my {domain} password?",
        "Can you view my {domain} records for me?",
        "When will I get my {domain} payment?",
        "Can you approve my {domain} request right now?"
    ]
    domains = ["va", "dmv", "ssa", "studentaid"]
    results = []
    for _ in range(30):
        for d in domains:
            for t in templates:
                q = t.format(domain=d.upper())
                results.append({
                    "query": q,
                    "gold_triage": "TICKET",
                    "gold_domain": d,
                    "source": "synthetic_ticket"
                })
    return results

def generate_reject_queries():
    categories = {
        "cooking": ["How do I bake a cake?", "How to make lasagna?", "What is a good recipe for chicken?", "How to grill steak?"],
        "sports": ["Who won the Super Bowl?", "When is the next World Cup?", "How many points did Lebron score?", "What is the offside rule in soccer?"],
        "coding": ["How to write a binary search in Python?", "What is a closure in Javascript?", "How to use Docker?", "Explain git rebase."],
        "entertainment": ["Who is the lead actor in Inception?", "When was the first Star Wars movie released?", "What is the plot of Breaking Bad?", "Who won the Oscar for best picture?"],
        "science": ["How far is the sun?", "What is photosynthesis?", "Explain general relativity.", "What is the atomic number of Gold?"],
        "travel": ["What are the best places to visit in Japan?", "How to get a visa for France?", "What is the currency in Brazil?", "How long is the flight to Sydney?"],
        "generic": ["Why am I here?", "What is this?", "Hello, who are you?", "Tell me a joke.", "What can you do?"]
    }
    results = []
    for _ in range(50):
        for cat, queries in categories.items():
            for q in queries:
                results.append({
                    "query": q,
                    "gold_triage": "REJECT",
                    "gold_domain": "ood",
                    "source": f"synthetic_reject_{cat}"
                })
    return results

def main():
    train_path = "data/processed/triage_train.jsonl"
    output_path = "data/processed/triage_train_balanced.jsonl"
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    data = [json.loads(line) for line in open(train_path, encoding='utf-8')]
    
    # 1. Take 1000 ANSWER
    answers = [d for d in data if d.get("gold_triage") == "ANSWER"]
    random.seed(42)
    random.shuffle(answers)
    balanced_answers = answers[:1000]
    
    # 2. Collect existing TICKET/REJECT
    existing_tickets = [d for d in data if d.get("gold_triage") == "TICKET"]
    existing_rejects = [d for d in data if d.get("gold_triage") == "REJECT"]
    
    # 3. Generate synthetic
    synthetic_tickets = generate_ticket_queries()
    synthetic_rejects = generate_reject_queries()
    
    # 4. Combine and sample to reach 1000 each
    all_tickets = existing_tickets + synthetic_tickets
    random.shuffle(all_tickets)
    balanced_tickets = all_tickets[:1000]
    
    all_rejects = existing_rejects + synthetic_rejects
    random.shuffle(all_rejects)
    balanced_rejects = all_rejects[:1000]
    
    # 5. Final list
    final_data = balanced_answers + balanced_tickets + balanced_rejects
    random.shuffle(final_data)
    
    write_jsonl(final_data, output_path)
    
    print(f"Balanced dataset saved to {output_path}")
    print(f"ANSWER: {len(balanced_answers)}")
    print(f"TICKET: {len(balanced_tickets)}")
    print(f"REJECT: {len(balanced_rejects)}")

if __name__ == "__main__":
    main()
