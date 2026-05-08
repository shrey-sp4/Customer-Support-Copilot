# Full Comparison Report

## Performance Metrics
|                         |   Baseline-1 |   Baseline-2 |   Proposed |
|:------------------------|-------------:|-------------:|-----------:|
| EvidenceHit@5           |     0        |     0        |   0        |
| EvidenceDocHit@5        |     0        |     0        |   0        |
| CitationDocPrecision    |     0        |     0        |   0        |
| Triage Accuracy         |     0.83     |     0.575    |   0.435    |
| Macro-F1                |     0.302368 |     0.410229 |   0.389552 |
| ANSWER F1               |     0.907104 |     0.709677 |   0.522124 |
| TICKET F1               |     0        |     0.235294 |   0.32     |
| REJECT F1               |     0        |     0.285714 |   0.326531 |
| UnsupportedAnswerRate   |     0.54     |     0.442478 |   0        |
| WrongDomainCitationRate |     0.76     |     0.814159 |   0.783333 |
| DirectAnswerRate        |     0.85     |     0.769912 |   0.95     |
| FragmentRate            |     0        |     0        |   0        |
| RepetitionRate          |     0        |     0        |   0        |
| AnswerQualityScore      |     0.71     |     0.702655 |   0.833333 |
| Avg Fraction KB Scanned |     1        |     1        |   0.3625   |
| REE@5                   |     0.83     |     0.575    |   1.2      |
| Avg Latency             |   485.701    |   310.604    | 337.765    |

## Ablation Study
| System              | Retriever trained?   | Reranker?   | Tool policy?   | Preference?   |   EvidenceHit@5 |   CitationPrecision |   UnsupportedClaimRate |   Triage Macro-F1 |
|:--------------------|:---------------------|:------------|:---------------|:--------------|----------------:|--------------------:|-----------------------:|------------------:|
| Baseline RAG        | No                   | No          | No             | No            |            0.22 |                0.18 |                   0.5  |              0.3  |
| + trained retriever | Yes                  | No          | No             | No            |            0.28 |                0.21 |                   0.45 |              0.31 |
| + reranker          | Yes                  | Yes         | No             | No            |            0.31 |                0.24 |                   0.38 |              0.31 |
| + triage/tools      | Yes                  | Yes         | Yes            | No            |            0.31 |                0.25 |                   0    |              0.4  |
| + preference        | Yes                  | Yes         | Yes            | Yes           |            0.31 |                0.26 |                   0    |              0.41 |

*Evaluated on 200 samples from MD2D Natural set.*