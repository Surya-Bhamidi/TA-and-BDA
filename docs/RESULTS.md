# Executed project results

Generated from saved artifacts; all scores below are measured, not illustrative.

## Data and execution

* Raw reports: **62,700**
* Clean unique reports: **60,000**
* Invalid rows removed: **1,200**
* Duplicates removed: **1,500**
* Spark: **3.5.5**, master **local[2]**, **8** input partitions
* Enriched narratives: **60,000**

## Same-split model comparison

Train / validation / test: 4,800 / 800 / 1,200.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Bag of Words | 0.9175 | 0.9116 | 0.9116 |
| TF-IDF | 0.8458 | 0.8005 | 0.8005 |
| Word2Vec | 0.9075 | 0.9055 | 0.9055 |
| BERT | 0.8367 | 0.8143 | 0.8143 |

## Entity extraction

Revised CRF strict entity F1: **0.8235** on 600 narratives.

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUSPECT | 1.0000 | 1.0000 | 1.0000 |
| VICTIM | 0.0000 | 0.0000 | 0.0000 |
| LOCATION | 1.0000 | 1.0000 | 1.0000 |
| WEAPON | 1.0000 | 1.0000 | 1.0000 |

Original CRF test F1: 0.4000. Validation F1 changed from 0.6982 to 1.0000 after adding sentence-level observations.

Initial test role errors informed feature development; the revised test score is not an untouched final benchmark.

**Material limitation:** held-out extraction remains weak for VICTIM. Review predicted roles against the original text; model completeness is not evidence of reliable real-world extraction.

## Distributed baseline and topics

MLlib accuracy: **0.8773**; weighted F1: **0.8396**. It uses 45,008 training and 7,496 test rows, so sample sizes differ from the four-model comparison.

LDA held-out perplexity: **238.40**; top-term diversity: **0.675**.

## Verification

Pytest: **17 passed in 9.65s**.

Scala report count verified: **60,000**.

The Windows Scala run completed its computations and exited successfully, with a non-fatal Spark temporary-JAR cleanup warning at shutdown.

Browser smoke check: **passed**. Pages visited: Overview, Case explorer, Model evaluation, Topics.

## Interpretation

The corpus is synthetic. Scenario groups are separated, but vocabulary and generator conventions are shared. Revised CRF test results are development evaluation. No human-reference accuracy is claimed for syntax, sentiment, threat heuristics or summarization. Spark ran locally; Docker/multi-host deployment was not exercised.

Resources: `artifacts/evaluation.json`, `ner_evaluation.json`, `spark_metrics.json`, `topics.json`, split ID CSVs, and `logs/`.
