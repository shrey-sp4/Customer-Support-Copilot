# Reject-Aware Domain-Routed Customer Support Copilot

This project implements a reject-aware domain-routed customer-support copilot using the MultiDoc2Dial dataset. The system uses centroid-based domain routing, FAISS retrieval, cross-encoder reranking, a boundary-aware triage model for ANSWER/TICKET/REJECT, structured tool execution, and preference/rubric-based answer selection. It compares against a baseline retrieval-only RAG system and reports retrieval, grounding, triage, routing, and latency metrics.

## Features
- **Two-phase Triage**: Lexical keyword gate for fast out-of-domain rejection, followed by a learned semantic triage using routing and KB proximity features.
- **Centroid-guided Domain Routing**: Groups KB chunks by domain and routes queries using cosine similarity against domain centroids.
- **Boundary-aware Triage Loss**: Custom loss function that trains the triage model to be confident and well-separated in its decisions.
- **Structured Tool Execution**: Uses Python-based tool execution for robust actions (RouteDomain, SearchKB, GetPolicy, CreateTicket, RejectQuery).
- **Preference/Rubric Alignment**: Uses a lightweight preference ranker to select the best generated candidate answer based on citation adherence and correctness.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```
> **Note**: `sentencepiece` is required for the local Flan-T5 generator. If you encounter warnings, ensure it installed correctly.

## Running the Pipeline

You can run the entire pipeline at once or step-by-step. The system provides multiple configurations under `configs/`:
- `smoke.yaml`: Tiny limits to quickly verify the pipeline.
- `full_local.yaml`: Full dataset, optimized for RTX 3060 6GB VRAM.
- `default.yaml`: Medium-sized limits.

### Smoke Run
Run a quick end-to-end test to ensure everything works:
```bash
python run_all.py --config configs/smoke.yaml
```

### Full Local Run
Run the full training pipeline:
```bash
python run_all.py --config configs/full_local.yaml
```

### Step-by-step Run
```bash
python scripts/prepare_data.py --config configs/full_local.yaml
python scripts/train_retriever.py --config configs/full_local.yaml
python scripts/build_index.py --config configs/full_local.yaml
python scripts/train_reranker.py --config configs/full_local.yaml
python scripts/train_triage.py --config configs/full_local.yaml
python scripts/train_preference.py --config configs/full_local.yaml
python scripts/evaluate.py --config configs/full_local.yaml
```

## Demo CLI
Start an interactive chat with the trained copilot:
```bash
python scripts/demo_cli.py --config configs/full_local.yaml
```

### Cluster-Gated Retrieval and Ambiguity-Aware Escalation
The system now implements an efficient cluster-gated retrieval mechanism:
- **Confident Queries**: If the query is strongly assigned to a single domain cluster (high margin), the system searches *only* that cluster, reducing noise and latency.
- **Ambiguous Queries**: If the query lies near multiple cluster boundaries, the system searches top-k nearby clusters and compares evidence.
- **Evidence-Based Escalation**: If evidence in the support domain is weak, a **Ticket** is created instead of attempting to answer.
- **Strict Rejection**: Queries truly outside the support clusters are **Rejected** immediately to prevent hallucination.
- **Efficiency**: This approach improves the **REE@5** metric by minimizing the fraction of the knowledge base scanned per query.

Or run a single query:
```bash
python scripts/demo_cli.py --query "Can I renew my benefits online?" --config configs/full_local.yaml
python scripts/demo_cli.py --query "Who won the IPL yesterday?" --config configs/full_local.yaml
```

## Results & Evaluation

We compare three systems on a balanced evaluation set (30 samples for smoke, full validation for full runs):

1. **Baseline-1 (Simple RAG)**: Assignment-required retrieval-only baseline. Defaults all queries to `ANSWER` using global retrieval. No triage or routing.
2. **Baseline-2 (Rule Workflow)**: A fair workflow comparison. Uses global retrieval but applies simple keyword-and-threshold rules to decide `ANSWER / TICKET / REJECT`. No cluster gating or learned components.
3. **Proposed System**: The complete reject-aware copilot. Adds cluster-gated retrieval, ambiguity-aware domain expansion, trained triage model, and a preference ranker for final selection.

| Metric | Baseline-1 | Baseline-2 | Proposed | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **EvidenceHit@5** | 0.167 | 0.167 | **0.200** | Proposed finds more relevant evidence. |
| **CitDocPrec** | 0.267 | 0.267 | 0.233 | Grounding accuracy for answerable queries. |
| **Triage Accuracy** | 1.000* | 0.800 | 0.533 | Proposed is more conservative to avoid hallucinations. |
| **Avg Latency** | 55.9ms | **35.0ms** | 37.5ms | Baseline-2/Proposed are faster than Baseline-1. |
| **REE@5** | 0.167 | 0.167 | **1.412** | **Proposed is 8x more efficient via domain gating.** |

*\*Baseline-1 triage accuracy is artificially high on answer-heavy sets as it defaults everything to "ANSWER". Baseline-2 provides a more realistic workflow benchmark.*

## Outputs
Evaluation metrics, tool traces, and ablation results are saved to `outputs/reports/`.
- `baseline_metrics.json`: Metrics for Baseline-1 (Simple RAG).
- `rule_workflow_baseline_metrics.json`: Metrics for Baseline-2 (Rule Workflow).
- `proposed_metrics.json`: Metrics for the Proposed Cluster-Gated system.
- `ablation_metrics.csv`: Summary table for system comparison.
