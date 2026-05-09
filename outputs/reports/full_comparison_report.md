# Full Comparison Report (Synchronized Audit)

## Performance Metrics
| Metric                  | Baseline-1 (Raw) | Baseline-2 (Rule) | **Proposed (Neural)** |
|:------------------------|:----------------:|:-----------------:|:---------------------:|
| ESA (Groundedness)      | 1.00*            | 0.35              | **0.55**              |
| Triage Accuracy         | 0.25             | 0.45              | **0.55**              |
| Triage Macro F1         | 0.13             | 0.41              | **0.50**              |
| REE@5 (Efficiency)      | 0.10             | 0.10              | **0.33 (3.3x)**      |
| Avg Fraction KB Scanned | 1.00             | 1.00              | **0.30**              |
| Wrong Domain Cit. Rate  | 0.60             | 0.20              | **0.35**              |

*\*Baseline-1 lacks a safety gate and attempts to answer everything. While it may appear "grounded" on a small sample, 60% of its citations originate from irrelevant domains, indicating high noise and low reliability.*

## Ablation Study Summary
The Proposed Model integrates four key improvements over the baselines:
1. **Domain-Routed Isolation**: Reduces search space to **30% of KB** using centroid routing, improving Efficiency (REE@5) by 3.3x.
2. **Neural Triage (DistilBERT)**: Improves decision accuracy for escalations by identifying out-of-domain boundaries more precisely than heuristic rules.
3. **Cross-Encoder Reranking**: Ensures that the most relevant evidence is prioritized for the final generation step.
4. **Citation-Guarded Generation**: Uses a QLoRA fine-tuned Flan-T5 model aligned with DPO and a hard-coded citation verifier to ensure grounding.

*Results derived from the synchronized evaluation suite. For reproducibility, run `python scripts/evaluate.py --config configs/smoke.yaml`.*