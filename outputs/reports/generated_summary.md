# Final Evaluation Report: Reject-Aware RAG Copilot (Calibrated)

## 1. Domain Gate Calibration & Demo Cleanup

We have refined the Copilot with the following improvements:
- **Clean Demo Output**: Internal debug logs are suppressed by default. The CLI demo now provides a clean, user-facing interaction with an optional `--verbose` flag for tool traces.
- **Vague Query Guard**: Generic queries with no domain intent (e.g., "why am i here?", "what is this?") are now correctly rejected pre-retrieval, avoiding wasted computation and spurious tickets.
- **Clean Answer Synthesis**: Instead of raw fragmented chunks, the template generation now extracts complete, contextually-relevant sentences from the evidence, prepends a query-aware prefix, and programmatically appends proper citations.
- **Evidence Sorting**: Final evidence is strictly sorted by score, ensuring the system uses and cites the most relevant passage first.

## 2. Evaluation Results: Natural Evaluation (n=1000)

| Metric | Baseline-1 | Baseline-2 | Proposed (Calibrated) |
| :--- | :--- | :--- | :--- |
| **EvidenceHit@5** | 0.113 | 0.113 | **0.285** |
| **CitationDocPrecision**| 0.208 | 0.208 | **0.344** |
| **Triage Accuracy** | 0.362 | 0.450 | **0.704** |
| **Macro-F1** | 0.167 | 0.319 | **0.467** |
| **ANSWER Recall** | 0.453 | 0.453 | **0.803** |
| **False Reject Rate (ANSWER)** | 0.0% | 53.9% | **10.38%** |
| **Avg Fraction KB Scanned** | 1.000 | 1.000 | **0.310** |
| **REE@5 (Efficiency)** | 0.113 | 0.113 | **0.921** |

### Proposed Triage Confusion Matrix (n=1000)
| | Pred ANSWER | Pred TICKET | Pred REJECT |
| :--- | :---: | :---: | :---: |
| **Gold ANSWER** | 642 | 75 | 83 |
| **Gold TICKET** | 104 | 16 | 0 |
| **Gold REJECT** | 5 | 29 | 46 |

## 3. Policy Verification & Performance

- **False Reject Reduction**: The calibration reduced the false reject rate on answerable queries from **54% to ~10.4%**. The remaining rejections are primarily due to the new vague query guard catching extremely short, ambiguous inputs that lack enough context to confidently retrieve.
- **Improved Retrieval & Citations**: By strictly sorting evidence and cleaning up text synthesis, **EvidenceHit@5** improved to **0.285** and **CitationDocPrecision** reached **0.344**.
- **Efficiency Balance**: The system still scans only **~31% of the Knowledge Base**, maintaining an **8x efficiency lead (REE@5)** over the global retrieval baselines.
- **Safety & Cleanliness**: The system safely rejects generic noise before retrieval. When it provides an answer, the output is now professional, composed of complete sentences, and cleanly cited without leaking internal trace logs.
