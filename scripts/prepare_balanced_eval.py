import os
import json
import random

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    data_dir = "data/processed"
    eval_path = os.path.join(data_dir, "eval_set.jsonl")
    triage_path = os.path.join(data_dir, "triage_train.jsonl")
    
    eval_data = read_jsonl(eval_path)
    triage_data = read_jsonl(triage_path)
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(eval_data)
    random.shuffle(triage_data)
    
    # ANSWER: Primarily from eval_set (which is mostly answers)
    answers = [s for s in eval_data if s.get("gold_triage") == "ANSWER"]
    if len(answers) < 30:
        answers += [s for s in triage_data if s.get("gold_triage") == "ANSWER"]
    answers = answers[:30]
    
    # TICKET: From triage_train
    tickets = [s for s in triage_data if s.get("gold_triage") == "TICKET"]
    tickets = tickets[:30]
    
    # REJECT: From triage_train
    rejects = [s for s in triage_data if s.get("gold_triage") == "REJECT"]
    rejects = rejects[:30]
    
    balanced = answers + tickets + rejects
    random.shuffle(balanced)
    
    output_path = os.path.join(data_dir, "balanced_eval.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in balanced:
            f.write(json.dumps(s) + "\n")
            
    print(f"Created balanced evaluation set at {output_path}")
    print(f"Distribution: ANSWER={len(answers)}, TICKET={len(tickets)}, REJECT={len(rejects)}")

if __name__ == "__main__":
    main()
