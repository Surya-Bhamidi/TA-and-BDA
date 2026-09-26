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

Train / validation / development / final test: 6,400 / 800 / 800 / 1,200.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Bag of Words | 0.7842 | 0.7393 | 0.7393 |
| TF-IDF | 0.8367 | 0.7934 | 0.7934 |
| Word2Vec | 0.9517 | 0.9505 | 0.9505 |
| BERT | 0.8325 | 0.7906 | 0.7906 |

## Entity extraction

Validation-selected CRF strict entity F1: **0.9846** on 600 final-test narratives.

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUSPECT | 1.0000 | 1.0000 | 1.0000 |
| VICTIM | 1.0000 | 1.0000 | 1.0000 |
| LOCATION | 1.0000 | 1.0000 | 1.0000 |
| WEAPON | 1.0000 | 1.0000 | 1.0000 |
| PROPERTY | 1.0000 | 0.4400 | 0.6111 |
| DATE | 1.0000 | 1.0000 | 1.0000 |
| TIME | 1.0000 | 1.0000 | 1.0000 |
| EVIDENCE | 1.0000 | 1.0000 | 1.0000 |

**Material limitation:** held-out recall is below 80% for PROPERTY. The overall entity F1 is dominated by more frequent types; consult each row above. Review predicted roles against the original text.

## Distributed baseline and topics

MLlib accuracy: **0.8749**; weighted F1: **0.8335**. It uses 37,512 training and 7,496 test rows, so sample sizes differ from the four-model comparison.

LDA held-out perplexity: **126.40**; top-term diversity: **0.662**.

## Verification

Pytest: **31 passed in 12.43s**.

Scala report count verified: **60,000**.

The Windows Scala run completed its computations and exited successfully, with a non-fatal Spark temporary-JAR cleanup warning at shutdown.

Browser smoke check: **passed**. Pages visited: Overview, Case explorer, Narrative lab, Model evaluation, Topics, Pipeline & syllabus, Learning guide.

## Classification uncertainty and calibration

| Model | 95% accuracy interval |
|---|---|
| Bag of Words | 0.7633 to 0.8092 |
| TF-IDF | 0.8175 to 0.8575 |
| Word2Vec | 0.9400 to 0.9625 |
| BERT | 0.8125 to 0.8517 |

Intervals use 400 report-level bootstrap resamples and do not cover new-source uncertainty.

Training-only grouped TF-IDF CV mean macro-F1: **0.2502**.

| TF-IDF final-test calibration | ECE | Log loss | Coverage at 0.60 | Accepted accuracy |
|---|---:|---:|---:|---:|
| before | 0.3830 | 0.9476 | 0.2033 | 1.0000 |
| after | 0.0757 | 0.4668 | 0.9217 | 0.8580 |

Temperature is fitted on validation only. Calibration is not guaranteed to improve under a shifted test distribution.

## Both entity models

| Model | Validation strict F1 | Final-test strict F1 |
|---|---:|---:|
| CRF | 1.0000 | 0.9846 |
| BERT | 0.9414 | 0.8741 |

## Interpretation

The corpus is synthetic. Scenario groups are separated, but vocabulary and generator conventions are shared. V2 model hashes were frozen before final predictions. No human-reference accuracy is claimed for syntax, sentiment, threat heuristics, search or summarization. Spark ran with a standalone master and two workers on one physical host; Docker and multi-host deployment were not exercised. V1 and V2 scores use different corpora and protocols.

Resources: `artifacts/evaluation.json`, `ner_evaluation.json`, `spark_metrics.json`, `topics.json`, split ID CSVs, and `logs/`.
