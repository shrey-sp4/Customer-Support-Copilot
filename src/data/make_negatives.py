"""Create synthetic REJECT and TICKET examples for triage training.

Output: data/processed/triage_train.jsonl
"""
import argparse
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Synthetic REJECT queries
# ---------------------------------------------------------------------------
EASY_REJECT = [
    "Who won the IPL match yesterday?",
    "Write Python code for merge sort.",
    "Give me a pasta recipe.",
    "What is the capital of France?",
    "Explain black holes.",
    "Tell me a joke about cats.",
    "What movies are showing this weekend?",
    "Who is the current Prime Minister of the UK?",
    "How do I bake sourdough bread?",
    "What is the weather forecast for tomorrow?",
    "Can you recommend a good novel to read?",
    "How do I fix a leaking faucet?",
    "What is Bitcoin worth right now?",
    "Explain quantum entanglement.",
    "Who invented the telephone?",
]

HARD_REJECT = [
    "How do I reset my Netflix password?",
    "Can I upgrade my iPhone warranty?",
    "Where is my Amazon package?",
    "Can I cancel my airline ticket?",
    "How do I claim health insurance reimbursement?",
    "Can I renew my passport here?",
    "How do I dispute my credit card transaction?",
    "Can I update my bank KYC using this portal?",
    "Can I transfer college credits through this support page?",
    "How do I get a replacement SIM card?",
    "How do I upgrade my internet plan?",
    "Can I return a product I bought on your e-commerce site?",
    "What are the tax filing deadlines for 2024?",
    "How do I reset my email password?",
    "Can I get a refund for my gym membership?",
]

NEAR_BOUNDARY_REJECT = [
    "Can I update my bank KYC using this government portal?",
    "How do I dispute my tax credit transaction on the portal?",
    "Can I transfer my college credits through this support page?",
    "How do I get a replacement document for my account?",
    "What documents do I need for a benefits application at another agency?",
    "How do I appeal a decision made by another government department?",
]

# ---------------------------------------------------------------------------
# Synthetic TICKET queries (in-domain but insufficient KB evidence)
# ---------------------------------------------------------------------------
TICKET_EXAMPLES = [
    "I followed the renewal instructions but my account still shows pending. Can someone check my case?",
    "The document says I can update my address, but the website gives an error. What should I do?",
    "My application was rejected but the KB does not explain the reason. Can support review it?",
    "I submitted my benefits claim 3 weeks ago and have not heard back. Can I get a status update?",
    "The online portal keeps timing out when I try to upload documents. Can someone help?",
    "I was told my case is under review, but I need an urgent update. Can support escalate?",
    "My account is locked and I cannot access any services. Who can unlock it?",
    "I received a notice that my benefits were terminated, but I do not know why.",
    "I need to speak to a caseworker about my specific situation — the online guide is not enough.",
    "I uploaded my verification documents but the system says they are missing.",
    "The portal shows my application as incomplete but I filled everything in.",
    "I have a complex situation involving both SSA and VA benefits. Who handles these?",
    "My representative payee changed, but the system did not update. Can support fix this?",
    "I believe there is an error in my benefits calculation. Can someone review it?",
    "I sent in the required forms by mail but there is no acknowledgment in the system.",
    "I cannot log into the portal because my identity verification failed, but I have the documents.",
    "My case was closed in error. How can I request it be reopened?",
    "I need to update my direct deposit information but the page is not loading.",
    "My dependent's information was removed from my account and I do not know why.",
    "The deadline to submit additional documents is tomorrow, but the upload button is grayed out.",
]


def make_negatives(
    dialogue_turns: List[dict],
    out_dir: str,
    max_samples: int = None,
) -> List[dict]:
    """Combine ANSWER turns with synthetic REJECT/TICKET to build triage_train.jsonl."""
    ensure_dir(out_dir)

    triage_records = []

    # --- ANSWER examples from real dialogue turns ---
    answer_turns = [t for t in dialogue_turns if t.get("gold_triage") == "ANSWER"]
    if max_samples:
        answer_turns = answer_turns[:max_samples // 3]

    for t in answer_turns:
        triage_records.append({
            "query_id":        t["query_id"],
            "query":           t["query"],
            "history":         t.get("history", ""),
            "gold_doc_id":     t.get("gold_doc_id", ""),
            "gold_chunk_id":   t.get("gold_chunk_id", ""),
            "gold_domain":     t.get("gold_domain", ""),
            "gold_triage":     "ANSWER",
            "source":          "multidoc2dial",
            # These will be filled at training time from retrieval features
            "keyword_gate":    "pass",
            "centroid_domain": t.get("gold_domain", ""),
            "centroid_sim_top1": 0.70,
            "centroid_margin":   0.20,
            "nearest_chunk_sim": 0.68,
            "retrieval_score_gap": 0.10,
        })

    # --- TICKET examples ---
    ticket_pool = TICKET_EXAMPLES * 3  # repeat to get more examples
    if max_samples:
        ticket_pool = ticket_pool[:max_samples // 3]

    for i, q in enumerate(ticket_pool):
        triage_records.append({
            "query_id":        f"ticket_{i:04d}",
            "query":           q,
            "history":         "",
            "gold_doc_id":     "",
            "gold_chunk_id":   "",
            "gold_domain":     "support",
            "gold_triage":     "TICKET",
            "source":          "synthetic",
            "keyword_gate":    "pass",
            "centroid_domain": "support",
            "centroid_sim_top1": 0.55,
            "centroid_margin":   0.10,
            "nearest_chunk_sim": 0.45,
            "retrieval_score_gap": 0.05,
        })

    # --- REJECT examples ---
    all_reject = EASY_REJECT + HARD_REJECT + NEAR_BOUNDARY_REJECT
    reject_pool = all_reject * 3
    if max_samples:
        reject_pool = reject_pool[:max_samples // 3]

    for i, q in enumerate(reject_pool):
        is_easy = q in EASY_REJECT
        keyword_gate = "reject" if is_easy else "pass"
        triage_records.append({
            "query_id":        f"reject_{i:04d}",
            "query":           q,
            "history":         "",
            "gold_doc_id":     "",
            "gold_chunk_id":   "",
            "gold_domain":     "ood",
            "gold_triage":     "REJECT",
            "source":          "synthetic",
            "keyword_gate":    keyword_gate,
            "centroid_domain": "ood",
            "centroid_sim_top1": 0.25 if is_easy else 0.42,
            "centroid_margin":   0.05,
            "nearest_chunk_sim": 0.20 if is_easy else 0.38,
            "retrieval_score_gap": 0.02,
        })

    random.shuffle(triage_records)
    if max_samples:
        triage_records = triage_records[:max_samples]

    out_path = os.path.join(out_dir, "triage_train.jsonl")
    write_jsonl(triage_records, out_path)
    logger.info(f"Created {len(triage_records)} triage training examples -> {out_path}")
    return triage_records


def main(args):
    dial_path = os.path.join(args.out_dir, "dialogue_turns.jsonl")
    if not os.path.exists(dial_path):
        raise FileNotFoundError(f"dialogue_turns.jsonl not found at {dial_path}.")
    dialogue_turns = read_jsonl(dial_path)
    make_negatives(dialogue_turns, args.out_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic REJECT/TICKET triage examples")
    parser.add_argument("--out_dir",     default="data/processed")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
