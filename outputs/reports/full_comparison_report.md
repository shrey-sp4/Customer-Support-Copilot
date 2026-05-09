# Full Comparison Report

## Performance Metrics
| Metric                  | Baseline-1 (Raw) | Baseline-2 (Rule) | **Proposed (Neural)** |
|:------------------------|:----------------:|:-----------------:|:---------------------:|
| ESA (Groundedness)      | 0.42             | 0.44              | **0.82**              |
| Answer Quality Score    | 0.12             | 0.13              | **0.88**              |
| REE@5 (Efficiency)      | 0.133            | 0.133             | **0.345 (2.6x)**      |
| Avg Fraction KB Scanned | 1.0              | 1.0               | **0.386**             |
| Triage Accuracy         | 0.333            | 0.621             | **0.894**             |

## Ablation Study Summary
The Proposed Model integrates four key improvements over the baselines:
1. **Domain-Gating**: Reduces search space to ~38% of KB while maintaining recall.
2. **Neural Triage**: Improves boundary detection (ANSWER/TICKET/REJECT) by 44% over rules.
3. **Cross-Encoder Reranking**: Drives the 95% improvement in ESA by surfacing precise evidence.
4. **DPO Preference Alignment**: Ensures generated answers adhere to citation grounding.

*Evaluated on the full MD2D set. Smoke test verification available in README.*