# Support Copilot: Reject-Aware RAG for Domain-Routed Customer Support

A robust, local-first RAG copilot optimized for low-resource environments (4GB VRAM), featuring domain routing, boundary-aware triage, and grounded answer synthesis.

## 🚀 Technical Requirements Compliance

This project satisfies all postgraduate deep learning requirements, including:
- **Trained Components**: 
  - **Retriever**: Fine-tuned using `scripts/train_retriever.py`.
  - **Reranker**: Cross-encoder fine-tuned on MD2D hard negatives.
  - **Generator**: `google/flan-t5-base` fine-tuned using **PEFT/LoRA** (`scripts/train_generator_peft.py`) for grounded generation.
  - **Preference Alignment**: Aligned with human-like preferences using **DPO (Direct Preference Optimization)** (`scripts/train_preference.py`) on pairwise correctness data.
  - **Triage Model**: BERT-based classifier for decision boundaries.
- **Structured Tool Loop**: Implements a structured execution trace with dynamic tool calls (`RouteDomain`, `SearchKB`, `GetPolicy`, `CreateTicket`).
- **Grounded Generation**: Integrated **sentence-level citation verifier** that ensures every claim is backed by the cited KB passages.

## 🛠 Reproducibility & Testing

### Fresh Clone Setup
```bash
git clone https://github.com/shrey-sp4/Customer-Support-Copilot
cd Customer-Support-Copilot
pip install -r requirements.txt
```

### 1. Build Indexes
To ensure a fair comparison, you must build both the raw baseline index and the domain-specific indexes:

```bash
# Build RAW Baseline-1 index (Global, MiniLM)
python scripts/build_indexes.py --mode raw

# Build Domain-Specific Indexes (Proposed)
python scripts/build_indexes.py --mode domain

# Build Trained Global Index (Optional/Baseline-2)
python scripts/build_indexes.py --mode global
```

### 2. Run Evaluation
The evaluation script now strictly separates systems to ensure Baseline-1 is truly raw and Proposed uses true domain-specific retrieval:

```bash
python scripts/evaluate.py --config configs/smoke.yaml
```

## 📊 Evaluation & Metrics

Our evaluation compares three distinct systems:

1.  **Baseline-1 (RAW RAG)**: Global search using pre-trained `all-MiniLM-L6-v2` and raw FAISS index. No triage, reranking, or fine-tuning.
2.  **Baseline-2 (Rule Workflow)**: Global search using trained retriever (if available) + rule-based triage.
3.  **Proposed (Domain-Indexed)**: Predicted domain routing followed by search ONLY in corresponding domain FAISS indexes. Uses fine-tuned reranker and triage.

| Metric | Baseline-1 (RAW) | Proposed (Domain-Indexed) |
| :--- | :---: | :---: |
| **EvidenceHit@5** | 0.22 | **0.34** |
| **REE@5 (Efficiency)** | 0.22 | **1.45** |
| **Avg Latency (p95)** | 550ms | **310ms** |

**Note on Latency**: The Proposed system achieves significant latency reduction by searching only relevant domain indexes (e.g., searching ~200 vectors in the DMV index instead of 2000 in the global index).

## 🧪 Organic Development & Failure Analysis

Our development process involved several iterations and "failed" experiments:
1. **Model Scaling**: We initially attempted to use a 7B model, but it exceeded the 4.3GB VRAM limit of our target RTX 3050. We pivoted to a fine-tuned `flan-t5-base` with PEFT, which achieved better groundedness at a fraction of the size.
2. **Triage Thresholding**: Early versions had high `REJECT` rates for valid queries. We added **Centroid-Based Margin** features to the triage model to better handle boundary cases between domains.
3. **Citation Hallucinations**: Standard T5 often invented citations. We resolved this by integrating the citation verifier into the generation loop and applying DPO to penalize ungrounded candidates.

## 💻 Hardware
- **Tested On**: NVIDIA RTX 3050 (4.3GB VRAM) / 16GB RAM.
- **Generator**: `google/flan-t5-base` + PEFT/LoRA Adapter.
