# Demonstration and viva guide

## Eight-minute demonstration

1. Double-click `START_DASHBOARD.cmd`; open `http://localhost:8501`.
2. **Overview:** explain the 60,000 fictional reports and inspect a district/date filter. Point out that generated trends are not actual crime rates.
3. **Case explorer:** search `unauthorized account access` or `stolen phone`. Open a report and identify the source label versus TF-IDF prediction.
4. **Entities & summary:** explain CRF predictions and the colors for suspects, victims, locations and weapons. Read the two-sentence extractive summary.
5. **Syntax & morphology:** select a sentence. Show a dependency arc, its POS tags and corresponding morphology table.
6. **Evidence search:** enter `witness` or `weapon`; show the retrieved source sentence. Explain lexical similarity and the no-evidence response.
7. **Narrative lab:** paste the sample below, leave BERT enabled, and click Analyze. Explain potential differences between BERT and TF-IDF predictions.
8. **Model evaluation:** compare four models and inspect the confusion matrix. Discuss macro-F1 and strict CRF entity F1, and why the held-out-template protocol matters.
9. **Topics:** choose a theme, inspect words and high-assignment examples. Topic IDs are arbitrary.
10. **Pipeline & syllabus:** show raw/clean row counts, local Spark execution mode, and the unit mapping. Open `docs/RESULTS.md` for recorded run results.

## Sample input

```text
An offender threatened a pedestrian and demanded a mobile phone before escaping on foot. The incident occurred at Cedar Market, Northgate. The reporting victim was Mira Vale. A witness identified Ari Sen as the alleged suspect. The witness described a knife carried by the assailant. Officers preserved camera footage and interviewed two witnesses. The case remains under investigation.
```

## Suggested viva questions

**Why use Spark for 60,000 rows?** This corpus can fit in memory, but Spark demonstrates actual distributed operators, partitioning, lazy execution, shuffle, SQL, storage and MLlib. It is a teaching-scale example, not proof that Spark is necessary at this volume.

**How do you prevent data leakage?** Split templates before fitting; deduplicate content; train vectorizers and embeddings only on the training split; use validation only for BERT checkpoint selection; keep ground-truth metadata out of model inputs. Shared generator vocabulary remains a limitation.

**What is neural about Word2Vec?** It learns dense embedding weights by predicting contextual words with a neural skip-gram objective and negative sampling. Document averaging loses order; BERT adds contextual self-attention.

**Why use a small BERT?** It is a real pretrained transformer with two layers and 128 hidden units that can be fine-tuned on CPU. It demonstrates sequence modeling while keeping the project runnable on a laptop.

**Why CRF instead of keyword matching?** The CRF learns observation and label-transition weights and predicts a coherent BIO label sequence. It can use the context around names to distinguish roles. The measured strict F1 shows how well this generalizes to the held-out templates.

**Does sentiment predict danger?** No. VADER is lexical polarity. The displayed threat-language score is a separate transparent heuristic with limited negation handling; neither has a validated real-world risk interpretation.

**What makes TextRank extractive?** It selects existing sentences based on a similarity graph rather than generating new sentences. It can still omit important information.

**What was actually distributed?** Spark cleaning, aggregation, Parquet writing and MLlib fitting ran as Spark tasks in local mode. General NLP training is bounded and local; a cluster configuration is supplied separately.

**Why not report summary ROUGE or parsing accuracy?** No human reference summaries or gold syntax annotations were created. Assigning such scores without a reference would be misleading.

## Artifacts to show an examiner

* `data/raw/reports.jsonl` and `artifacts/dataset_manifest.json` for provenance.
* `data/processed/reports/` for Spark partitioned Parquet.
* `artifacts/spark_metrics.json` for counts and MLlib metrics.
* `artifacts/evaluation.json`, `ner_evaluation.json`, `topics.json` for measured NLP results.
* `artifacts/train_ids.csv`, `validation_ids.csv`, `test_ids.csv` for split reproducibility.
* `artifacts/models/bert_classifier/` for trained transformer weights/config/tokenizer.
* `scala/CrimeAnalytics.scala` and `logs/scala.log` for functional programming and Spark evidence.
* `logs/tests.log` and `docs/RESULTS.md` for validation evidence.
