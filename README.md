# Support Copilot: Reject-Aware RAG for Domain-Routed Customer Support

A robust, local-first RAG copilot optimized for low-resource environments (4GB-6GB VRAM), featuring domain routing, boundary-aware triage, and grounded answer synthesis.

## 🚀 Reproducibility: Fresh Clone Guide

To run the complete pipeline from scratch (including data prep and training):

```bash
# 1. Clone and Setup
git clone https://github.com/shrey-sp4/Customer-Support-Copilot
cd Customer-Support-Copilot
pip install -r requirements.txt

# 2. Run Full Pipeline (Smoke Mode)
# This will prepare data, train all models (retriever/reranker/triage), 
# build FAISS index, and run final evaluation.
python run_all.py --config configs/smoke.yaml

# 3. Try the Interactive Demo
python scripts/demo_cli.py --config configs/smoke.yaml
```

## 🛠 Project Components (Honest Status)

| Component | Status | Implementation Detail |
| :--- | :--- | :--- |
| **Data Pipeline** | ✅ Implemented | Full `src/data` package for loading MD2D, building KB, and creating training pairs. |
| **Domain Routing** | ✅ Implemented | Centroid-based routing with keyword-driven lexical gating. |
| **Retrieval** | ✅ Implemented | FAISS dense retrieval using `sentence-transformers`. |
| **Reranking** | ✅ Trained | Cross-encoder model fine-tuned on MD2D hard negatives. |
| **Triage Model** | ✅ Trained | BERT-based classifier for ANSWER/TICKET/REJECT decisions. |
| **Generation** | ✅ Grounded | `flan-t5-base` with a sentence-level citation verifier. |
| **Tool Loop** | ✅ Structured | Structured traces for RouteDomain, SearchKB, GetPolicy, and CreateTicket. |

## 📊 Evaluation Results (Summary)

The system is evaluated on a 200-sample subset of the MD2D Natural set.

| Metric | Baseline RAG | Proposed System |
| :--- | :---: | :---: |
| **EvidenceHit@5** | 0.22 | **0.31** |
| **UnsupportedClaimRate** | 0.50 | **0.00** |
| **Triage Macro-F1** | 0.30 | **0.41** |
| **REE@5 (Efficiency)** | 0.83 | **1.39** |

*See `outputs/reports/full_comparison_report.md` for the complete 17-metric table and ablation study.*

## 💻 Hardware & Dataset
- **Hardware**: Tested on NVIDIA RTX 3050 (4.3GB VRAM) / 16GB RAM.
- **Dataset**: IBM/multidoc2dial (processed into 11,694 KB chunks across 4 domains: DMV, SSA, VA, StudentAid).
- **Generator**: `google/flan-t5-base` (local execution).
