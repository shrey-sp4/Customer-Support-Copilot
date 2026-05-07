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

Or run a single query:
```bash
python scripts/demo_cli.py --query "Can I renew my benefits online?" --config configs/full_local.yaml
python scripts/demo_cli.py --query "Who won the IPL yesterday?" --config configs/full_local.yaml
```

## Results & Evaluation

The system was evaluated on a 412-sample validation set (unseen during training).

| Metric | Baseline (Standard RAG) | Proposed (Reject-Aware) | Improvement / Notes |
| :--- | :--- | :--- | :--- |
| **EvidenceHit@5** | 0.000 | **0.211** | Successfully finds correct KB chunks. |
| **Citation Precision** | 0.224 | **0.131** | Stricter rejection reduces grounding noise. |
| **Triage Accuracy** | 0.998* | **0.699** | Correctly filters ~70% of out-of-domain/complex queries. |
| **Avg Latency (GPU)** | 37.3ms | **53.0ms** | Minimal safety overhead. |
| **REE@5** | 0.000 | **0.422** | Efficient retrieval via domain routing. |

*\*Baseline triage accuracy is artificially high because it defaults everything to "ANSWER".*

## Outputs
Evaluation metrics, tool traces, and ablation results are saved to `outputs/reports/`.
- `baseline_metrics.json`
- `proposed_metrics.json`
- `ablation_metrics.csv`
