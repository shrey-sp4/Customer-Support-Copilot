import json
import os
import random

def main():
    input_path = "data/processed/eval_md2d_natural_1000.jsonl"
    output_path = "data/processed/eval_answer_quality_100.jsonl"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    # Filter only ANSWER samples
    answers = [r for r in records if r.get("gold_triage") == "ANSWER"]
    
    if len(answers) < 100:
        print(f"Warning: Only {len(answers)} ANSWER samples found. Using all of them.")
        selected = answers
    else:
        random.seed(42)
        selected = random.sample(answers, 100)
        
    with open(output_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
            
    print(f"Created {output_path} with {len(selected)} samples.")

if __name__ == "__main__":
    main()
