# Executed project results

Generated from saved artifacts; all scores below are measured, not illustrative.

## Data and execution

* Raw reports: **62,700**
* Clean unique reports: **60,000**
* Invalid rows removed: **1,200**
* Duplicates removed: **1,500**
* Spark: **3.5.5**, master **spark://127.0.0.1:17077**, **8** input partitions
* Enriched narratives: **60,000**

## Same-split model comparison

Train / validation / development / final test: 12,000 / 800 / 800 / 1,200.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Bag of Words | 0.8067 | 0.8068 | 0.8068 |
| TF-IDF | 0.8825 | 0.8815 | 0.8815 |
| Word2Vec | 0.8217 | 0.8188 | 0.8188 |
| BERT | 0.7958 | 0.7926 | 0.7926 |

## Entity extraction

Validation-selected CRF strict entity F1: **0.9123** on 600 evaluation narratives.

Retained V3 test subset is previously viewed; new synthetic context cases share grammar. Screenshot cases are development regressions, not a final benchmark.

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUSPECT | 0.8501 | 0.8134 | 0.8313 |
| VICTIM | 0.8792 | 0.7806 | 0.8270 |
| LOCATION | 0.9600 | 0.8444 | 0.8985 |
| WEAPON | 1.0000 | 1.0000 | 1.0000 |
| PROPERTY | 0.9701 | 0.9177 | 0.9432 |
| DATE | 1.0000 | 1.0000 | 1.0000 |
| TIME | 1.0000 | 1.0000 | 1.0000 |
| EVIDENCE | 1.0000 | 1.0000 | 1.0000 |

**Material limitation:** held-out recall is below 80% for VICTIM. The overall entity F1 is dominated by more frequent types; consult each row above. Review predicted roles against the original text.

## Distributed baseline and topics

MLlib accuracy: **0.7970**; weighted F1: **0.7895**. It uses 39,528 training and 6,824 test rows, so sample sizes differ from the four-model comparison.

LDA held-out perplexity: **476.67**; top-term diversity: **0.900**.

## Verification

Pytest: **51 passed in 18.07s**.

Scala report count verified: **60,000**.

The Windows Scala run completed its computations and exited successfully, with a non-fatal Spark temporary-JAR cleanup warning at shutdown.

Browser smoke check: **passed**. Pages visited: Overview, Case explorer, Narrative lab, Model evaluation, Topics, Pipeline & syllabus, Learning guide.

## Classification uncertainty and calibration

| Model | 95% accuracy interval |
|---|---|
| Bag of Words | 0.7817 to 0.8275 |
| TF-IDF | 0.8633 to 0.9000 |
| Word2Vec | 0.8000 to 0.8425 |
| BERT | 0.7750 to 0.8200 |

Intervals use 400 report-level bootstrap resamples and do not cover new-source uncertainty.

Training-only grouped TF-IDF CV mean macro-F1: **0.6708**.

| TF-IDF final-test calibration | ECE | Log loss | Coverage at 0.60 | Accepted accuracy |
|---|---:|---:|---:|---:|
| before | 0.2202 | 0.5465 | 0.6033 | 0.9834 |
| after | 0.0672 | 0.3254 | 0.9483 | 0.9042 |

Temperature is fitted on validation only. Calibration is not guaranteed to improve under a shifted test distribution.

## Both entity models

| Model | Validation strict F1 | Final-test strict F1 |
|---|---:|---:|
| CRF | 0.9391 | 0.9123 |
| BERT | 0.8351 | 0.7615 |

## Informal English development challenges

Fixed hand-written development/regression examples; not an independent real-world benchmark.

| Release | Category cases correct / 32 | Exact people + location cases / 8 |
|---|---:|---:|
| v2 | 23 | 0 |
| v3 | 31 | 8 |
| v3.1 | 31 | 6 |

These hand-written examples informed development. They are not an untouched final benchmark or evidence of flawless real-world performance. Every case is retained in artifacts/robustness_development.json.

## Interpretation

The corpus is synthetic. Scenario groups are separated, but vocabulary and generator conventions are shared. V3 model hashes were frozen before final predictions. No human-reference accuracy is claimed for syntax, sentiment, threat heuristics, search or summarization. Standalone Spark master and worker processes; see cluster_execution.json for verified workers and host scope. Docker and multi-host deployment were not exercised. Version benchmarks use different corpora and protocols.

Resources: `artifacts/evaluation.json`, `ner_evaluation.json`, `spark_metrics.json`, `topics.json`, split ID CSVs, and `logs/`.
