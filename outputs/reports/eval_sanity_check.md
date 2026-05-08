# Evaluation Sanity Check Report

## 1. Label Distributions

| Set | Total | ANSWER | TICKET | REJECT |
| --- | --- | --- | --- | --- |
| Natural (1000) | 1000 | 800 | 120 | 80 |
| Balanced (300) | 300 | 100 | 100 | 100 |
| Robustness (150) | 150 | 40 | 35 | 75 |

## 2. Baseline Accuracy Validation

Expected accuracy if a system always predicts **ANSWER**:

| Set | Expected Baseline-1 Accuracy |
| --- | --- |
| Natural (1000) | 0.800 |
| Balanced (300) | 0.333 |
| Robustness (150) | 0.267 |

> [!IMPORTANT]
> If the reported Baseline-1 accuracy in the main evaluation does not match the 'Expected' values above, there is a logging or data-loading mismatch.

## 3. Current Performance Snapshot

| system                  |   accuracy |   macro_f1 |   evidence_hit_at_5 |   avg_latency |   avg_fraction_kb |   ree_at_5 |
|:------------------------|-----------:|-----------:|--------------------:|--------------:|------------------:|-----------:|
| Baseline-1 (natural)    |   0.8      |   0.296296 |             0.21125 |       36.4616 |          1        |    0.21125 |
| Baseline-2 (natural)    |   0.443    |   0.373298 |             0.12375 |       27.1082 |          1        |    0.12375 |
| Proposed (natural)      |   0.695    |   0.572853 |             0.30625 |       55.7333 |          0.30325  |    1.00989 |
| Baseline-1 (balanced)   |   0.333333 |   0.166667 |             0.23    |       33.4125 |          1        |    0.23    |
| Baseline-2 (balanced)   |   0.556667 |   0.504371 |             0.13    |       23.3075 |          1        |    0.13    |
| Proposed (balanced)     |   0.68     |   0.682897 |             0.36    |       43.7117 |          0.273333 |    1.31707 |
| Baseline-1 (robustness) |   0.266667 |   0.140351 |             0       |       29.2774 |          1        |    0       |
| Baseline-2 (robustness) |   0.7      |   0.514338 |             0       |       18.8285 |          1        |    0       |
| Proposed (robustness)   |   0.993333 |   0.993068 |             0       |       38.0087 |          0.18     |    0       |
