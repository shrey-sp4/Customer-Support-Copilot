# Support Copilot: Reject-Aware RAG for Domain-Routed Customer Support

A robust, local-first RAG copilot optimized for low-resource environments (4GB VRAM), featuring domain routing, boundary-aware triage, and grounded answer synthesis.

## 🚀 Technical Requirements Compliance

This project satisfies all postgraduate deep learning requirements, including:
- **Trained Components**: 
  - **Retriever**: Fine-tuned `all-MiniLM-L6-v2` using `scripts/train_retriever.py` with Triplet Loss.
  - **Reranker**: Cross-encoder fine-tuned on MD2D hard negatives.
  - **Generator**: `google/flan-t5-large` fine-tuned using **PEFT/QLoRA (4-bit)** (`scripts/train_generator_peft.py`).
  - **Preference Alignment**: Aligned via **DPO (Direct Preference Optimization)** (`scripts/train_preference.py`).
  - **Triage Model**: BERT-based classifier (`distilbert-base-uncased`) for decision boundaries.
- **Structured Tool Loop**: Implements a structured execution trace with dynamic tool calls (`RouteDomain`, `SearchKB`, `GetPolicy`, `CreateTicket`).
- **Grounded Generation**: Integrated **sentence-level citation verifier** and template fallback for ungrounded neural outputs.

## 🛠 Reproducibility (Fresh Clone)

To run the full pipeline immediately on a tiny sample (50 chunks, 10 queries):

```bash
# 1. Setup
git clone https://github.com/shrey-sp4/Customer-Support-Copilot
cd Customer-Support-Copilot
pip install -r requirements.txt

# 2. Run Sample Pipeline (Uses data/sample/)
# Build Indexes
python scripts/build_indexes.py --mode raw --kb_path data/sample/kb_chunks.jsonl
python scripts/build_indexes.py --mode domain --kb_path data/sample/kb_chunks.jsonl
python scripts/build_indexes.py --mode global --kb_path data/sample/kb_chunks.jsonl

# Run Evaluation
python scripts/evaluate.py --config configs/smoke.yaml
```

## 📊 Final Evaluation Results

| Metric | Baseline-1 (RAW) | Baseline-2 (Raw Workflow) | Proposed (Novelty) |
| :--- | :---: | :---: | :---: |
| **EvidenceHit@5** | 0.133 | 0.133 | 0.133 |
| **Triage Accuracy** | 0.333 | 0.333 | **0.444** |
| **Triage Macro-F1** | 0.167 | 0.167 | **0.426** |
| **Search Latency** | 17.6 ms | 10.9 ms | **7.9 ms** |
| **Avg Fraction KB Scanned** | 1.000 | 1.000 | **0.386** |
| **REE@5 (Efficiency)** | 0.133 | 0.133 | **0.345** |

**Technical Honesty**: The Proposed system achieves a **~50% reduction in search latency** and a **2.6x improvement in knowledge efficiency (REE@5)** by intelligently partitioning the knowledge base.

## 📈 Training Evidence & Artifacts

### 1. Retriever (Triplet Loss)
- **Train/Val Size**: 1,200 / 300 pairs (MD2D)
- **Final Loss**: 0.042
- **Path**: `outputs/retriever/`

### 2. Triage Classifier (BERT)
- **Train/Val Size**: 900 / 300 (Balanced ANSWER/TICKET/REJECT)
- **Metrics**: Accuracy: 0.62, Macro-F1: 0.50
- **Path**: `outputs/triage/`

### 3. Generator (QLoRA + DPO)
- **Model**: `google/flan-t5-large` (780M)
- **Training**: 1 Epoch, PEFT/LoRA (rank=8, alpha=32)
- **DPO Alignment**: Trained on 500 pairwise preference samples.
- **Path**: `outputs/generator/` (Adapter) / `outputs/preference/` (DPO)

## 🧪 Failure Analysis & Pivots
1. **Model Scaling**: Switched from 7B to `flan-t5-large` with **4-bit QLoRA** to fit within 4.3GB VRAM.
2. **Intent Penalty**: Added manual penalties for cross-intent retrieval (e.g., Driver License vs Non-Driver ID) to fix precision issues in RAG.
3. **Template Fallback**: Implemented a robust fallback to prevent "TICKET" creation when evidence is valid but the LLM fails to generate a grounded answer.

## 💻 Hardware
- **Target**: NVIDIA RTX 3050 (4.3GB VRAM) / 16GB RAM.
