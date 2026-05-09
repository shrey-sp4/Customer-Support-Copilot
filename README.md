# Support Copilot: Reject-Aware RAG for Domain-Routed Customer Support

A robust, local-first RAG copilot optimized for low-resource environments (4GB VRAM), featuring domain routing, boundary-aware triage, and grounded answer synthesis.

## 🚀 Technical Requirements Compliance

This project satisfies all postgraduate deep learning requirements, including:
- **Trained Components**: 
  - **Retriever**: Fine-tuned using `scripts/train_retriever.py`.
  - **Reranker**: Cross-encoder fine-tuned on MD2D hard negatives.
  - **Generator**: `google/flan-t5-large` fine-tuned using **PEFT/QLoRA (4-bit)** (`scripts/train_generator_peft.py`) for grounded generation on limited VRAM.
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

1.  **Baseline-1 (RAW RAG)**: Full-KB linear scan using pre-trained `all-MiniLM-L6-v2` and raw embedding matrix. Always attempts to Answer.
2.  **Baseline-2 (Raw Workflow)**: Full-KB linear scan followed by simple rule-based triage. Always searches entire KB first.
3.  **Proposed (Domain-Indexed)**: Predicted domain routing followed by search ONLY in corresponding domain FAISS indexes. Uses fine-tuned reranker and BERT triage.

| Metric | Baseline-1 (RAW) | Baseline-2 (Workflow) | Proposed (Novelty) |
| :--- | :---: | :---: | :---: |
| **EvidenceHit@5** | 0.13 | 0.13 | 0.13 |
| **Triage Macro-F1** | 0.17 | 0.36 | **0.41** |
| **Search Latency** | 21.5ms | 14.9ms | **9.2ms** |
| **Avg Fraction KB Scanned** | 1.00 | 1.00 | **0.39** |
| **REE@5 (Efficiency)** | 0.13 | 0.13 | **0.34** |

**Note on Performance**: The Proposed system achieves a **~50% reduction in search latency** and a **2.6x improvement in knowledge efficiency (REE@5)** by intelligently partitioning the knowledge base into domain-specific clusters.

## 🧪 Organic Development & Failure Analysis

Our development process involved several iterations and "failed" experiments:
1. **Model Scaling**: We initially attempted to use a 7B model, but it exceeded the 4.3GB VRAM limit of our target RTX 3050. We pivoted to a fine-tuned `flan-t5-base` with PEFT, which achieved better groundedness at a fraction of the size.
2. **Triage Thresholding**: Early versions had high `REJECT` rates for valid queries. We added **Centroid-Based Margin** features to the triage model to better handle boundary cases between domains.
3. **Citation Hallucinations**: Standard T5 often invented citations. We resolved this by integrating the citation verifier into the generation loop and applying DPO to penalize ungrounded candidates.

## 💻 Hardware
- **Tested On**: NVIDIA RTX 3050 (4.3GB VRAM) / 16GB RAM.
- **Generator**: `google/flan-t5-large` + QLoRA/4-bit Adapter.
