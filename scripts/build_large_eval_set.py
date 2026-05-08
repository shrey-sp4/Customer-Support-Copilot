import os
import json
import random
import re

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

TICKET_TEMPLATES = [
    "Can you check the status of my {domain} application?",
    "Can you update my personal information for my {domain} account?",
    "Why was my specific {domain} payment delayed this month?",
    "Can you tell me my exact {domain} benefit amount?",
    "What is the private phone number of a {domain} officer?",
    "Can you submit my {domain} form on my behalf?",
    "Can you guarantee my {domain} approval?",
    "Can you tell me whether my personal {domain} appeal was accepted?",
    "I need to speak to a human about my {domain} case status.",
    "Can you reset my password for the {domain} portal?",
    "Where is my {domain} check? I haven't received it yet.",
    "Can you fix the error in my {domain} file?",
    "What is the status of my claim #{id}?",
    "Can you schedule an appointment for me at the local {domain} office?",
    "I want to complain about a specific {domain} employee.",
]

REJECT_TEMPLATES = [
    "How do I bake a {food}?",
    "Write me a {genre} poem.",
    "What {product} should I buy?",
    "How do I invest in {topic}?",
    "Explain {scientific_topic}.",
    "Who will win the {sport} match?",
    "How do I learn {skill}?",
    "Make me a {activity} plan.",
    "What is the recipe for {food}?",
    "Tell me a joke about {subject}.",
    "What is the weather in {city}?",
    "Who is the CEO of {company}?",
    "Translate '{phrase}' to Spanish.",
    "How do I fix a leaky faucet?",
    "What is the capital of {country}?",
]

FOODS = ["cake", "biryani", "pizza", "cookie", "sourdough bread"]
GENRES = ["love", "sad", "nature", "epic", "haiku"]
PRODUCTS = ["gaming laptop", "smartphone", "electric car", "mechanical keyboard"]
TOPICS = ["crypto", "real estate", "stocks", "gold"]
SCIENCE = ["quantum mechanics", "general relativity", "photosynthesis", "black holes"]
SPORTS = ["cricket", "soccer", "basketball", "tennis"]
SKILLS = ["guitar", "coding", "painting", "swimming"]
ACTIVITIES = ["gym workout", "diet", "study", "travel"]
SUBJECTS = ["AI", "lawyers", "cats", "programming"]
CITIES = ["New York", "London", "Tokyo", "Mumbai"]
COMPANIES = ["Apple", "Google", "Tesla", "Microsoft"]
COUNTRIES = ["France", "Japan", "India", "Canada"]
PHRASES = ["Hello", "Goodbye", "Thank you", "Where is the library?"]

DOMAINS = ["dmv", "ssa", "va", "studentaid"]

def generate_synthetic(templates, count, triage_label, seed=42):
    random.seed(seed)
    results = []
    for _ in range(count):
        tpl = random.choice(templates)
        # Fill placeholders
        query = tpl
        if "{domain}" in query:
            query = query.replace("{domain}", random.choice(DOMAINS).upper())
        if "{id}" in query:
            query = query.replace("{id}", str(random.randint(100000, 999999)))
        if "{food}" in query:
            query = query.replace("{food}", random.choice(FOODS))
        if "{genre}" in query:
            query = query.replace("{genre}", random.choice(GENRES))
        if "{product}" in query:
            query = query.replace("{product}", random.choice(PRODUCTS))
        if "{topic}" in query:
            query = query.replace("{topic}", random.choice(TOPICS))
        if "{scientific_topic}" in query:
            query = query.replace("{scientific_topic}", random.choice(SCIENCE))
        if "{sport}" in query:
            query = query.replace("{sport}", random.choice(SPORTS))
        if "{skill}" in query:
            query = query.replace("{skill}", random.choice(SKILLS))
        if "{activity}" in query:
            query = query.replace("{activity}", random.choice(ACTIVITIES))
        if "{subject}" in query:
            query = query.replace("{subject}", random.choice(SUBJECTS))
        if "{city}" in query:
            query = query.replace("{city}", random.choice(CITIES))
        if "{company}" in query:
            query = query.replace("{company}", random.choice(COMPANIES))
        if "{country}" in query:
            query = query.replace("{country}", random.choice(COUNTRIES))
        if "{phrase}" in query:
            query = query.replace("{phrase}", random.choice(PHRASES))
            
        results.append({
            "query": query,
            "gold_decision": triage_label,
            "gold_triage": triage_label, # back compat
            "gold_domain": random.choice(DOMAINS) if triage_label == "TICKET" else None,
            "gold_doc_id": None,
            "gold_chunk_id": None,
            "source": f"synthetic_{triage_label.lower()}",
            "notes": "synthetic example for evaluation"
        })
    return results

def main():
    data_dir = "data/processed"
    dialogue_path = os.path.join(data_dir, "dialogue_turns.jsonl")
    
    logger_info = lambda x: print(f"INFO: {x}")
    
    logger_info("Loading dialogue turns...")
    all_turns = read_jsonl(dialogue_path)
    
    # Filter for ANSWERs
    answers = [t for t in all_turns if t.get("gold_triage") == "ANSWER"]
    random.seed(42)
    random.shuffle(answers)
    
    logger_info(f"Available real ANSWERs: {len(answers)}")
    
    # 1. Build Natural 1000
    # 800 ANSWER, 120 TICKET, 80 REJECT
    natural_answers = answers[:800]
    for a in natural_answers:
        a["source"] = "multidoc2dial_real"
        a["gold_decision"] = "ANSWER"
        
    natural_tickets = generate_synthetic(TICKET_TEMPLATES, 120, "TICKET", seed=42)
    natural_rejects = generate_synthetic(REJECT_TEMPLATES, 80, "REJECT", seed=43)
    
    natural_1000 = natural_answers + natural_tickets + natural_rejects
    random.shuffle(natural_1000)
    
    write_jsonl(natural_1000, os.path.join(data_dir, "eval_md2d_natural_1000.jsonl"))
    logger_info("Saved eval_md2d_natural_1000.jsonl")
    
    # 2. Build Balanced 300
    # 100 ANSWER, 100 TICKET, 100 REJECT
    balanced_answers = answers[800:900]
    for a in balanced_answers:
        a["source"] = "multidoc2dial_real"
        a["gold_decision"] = "ANSWER"
        
    balanced_tickets = generate_synthetic(TICKET_TEMPLATES, 100, "TICKET", seed=44)
    balanced_rejects = generate_synthetic(REJECT_TEMPLATES, 100, "REJECT", seed=45)
    
    balanced_300 = balanced_answers + balanced_tickets + balanced_rejects
    random.shuffle(balanced_300)
    
    write_jsonl(balanced_300, os.path.join(data_dir, "eval_workflow_balanced_300.jsonl"))
    logger_info("Saved eval_workflow_balanced_300.jsonl")

if __name__ == "__main__":
    main()
