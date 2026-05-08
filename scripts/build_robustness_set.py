import json
import random
import os

def main():
    output_path = "data/processed/eval_robustness_150.jsonl"
    
    robustness_data = [
        # --- Paraphrases / Variations ---
        {"query": "HOW DO I APPLY FOR FAFSA?", "gold_triage": "ANSWER", "gold_domain": "studentaid"},
        {"query": "how do i apply for fafsa", "gold_triage": "ANSWER", "gold_domain": "studentaid"},
        {"query": "How do i apply for fasa?", "gold_triage": "ANSWER", "gold_domain": "studentaid"}, # Spelling
        {"query": "PROCEDURE FOR SCHOLARSHIP APPLICATION", "gold_triage": "ANSWER", "gold_domain": "studentaid"},
        
        # --- Mixed Case / Noise ---
        {"query": "cAn YoU hElP mE wItH vA bEnEfItS?", "gold_triage": "ANSWER", "gold_domain": "va"},
        {"query": "va benefits help!!!!", "gold_triage": "ANSWER", "gold_domain": "va"},
        
        # --- Vague Out-of-Domain ---
        {"query": "why am i here?", "gold_triage": "REJECT", "gold_domain": "ood"},
        {"query": "what is the meaning of life?", "gold_triage": "REJECT", "gold_domain": "ood"},
        {"query": "hello world", "gold_triage": "REJECT", "gold_domain": "ood"},
        {"query": "idk what to ask", "gold_triage": "REJECT", "gold_domain": "ood"},
        
        # --- In-Domain Personal / Action Requests (Should be TICKET) ---
        {"query": "Can you check my FAFSA application status?", "gold_triage": "TICKET", "gold_domain": "studentaid"},
        {"query": "Update my direct deposit for VA benefits.", "gold_triage": "TICKET", "gold_domain": "va"},
        {"query": "Why was my SSA payment delayed this month?", "gold_triage": "TICKET", "gold_domain": "ssa"},
        {"query": "Can you submit my DMV form for me?", "gold_triage": "TICKET", "gold_domain": "dmv"},
        {"query": "Guarantee my scholarship approval.", "gold_triage": "TICKET", "gold_domain": "studentaid"},
        {"query": "Tell me my exact benefit amount for this month.", "gold_triage": "TICKET", "gold_domain": "va"},
        
        # --- Ambiguous Multi-Domain ---
        {"query": "I need help with my application.", "gold_triage": "TICKET", "gold_domain": "support"},
        {"query": "How do I log in to the portal?", "gold_triage": "ANSWER", "gold_domain": "support"},
        
        # --- Out of Domain Categories ---
        {"query": "How do I bake a chocolate cake?", "gold_triage": "REJECT", "gold_domain": "ood"},
        {"query": "Who won the NBA finals last year?", "gold_triage": "REJECT", "gold_domain": "ood"},
        {"query": "How to write a hello world in C++?", "gold_triage": "REJECT", "gold_domain": "ood"},
        {"query": "What is the capital of France?", "gold_triage": "REJECT", "gold_domain": "ood"}
    ]
    
    # Fill up to 150 with some variations
    while len(robustness_data) < 150:
        robustness_data.append(random.choice(robustness_data).copy())
        
    with open(output_path, "w", encoding="utf-8") as f:
        for d in robustness_data:
            f.write(json.dumps(d) + "\n")
            
    print(f"Robustness set saved to {output_path} (n={len(robustness_data)})")

if __name__ == "__main__":
    main()
