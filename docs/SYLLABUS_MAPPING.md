# Syllabus mapping

## Text Analytics / NLP

| Unit | Syllabus requirement | Implemented in | Demonstration |
|---|---|---|---|
| 1 | Computational linguistics, syntax, morphology, NLP problems | `crime_nlp/inference.py`, `app.py` | Token, lemma, POS, morphology and dependency table; document classification and role extraction |
| 2 | BoW / TF-IDF; language representation | `crime_nlp/train.py`, `crime_nlp/features.py` | Word unigram/bigram counts; word and character-fragment TF-IDF feeding the same logistic regression; shared-split comparison |
| 2 | Neural word embeddings | `crime_nlp/train.py` | Gensim neural Word2Vec skip-gram, document averaging; 100-dimensional vectors |
| 3 | Sequence models: BERT/GPT alongside HMM/CRF | `crime_nlp/bert_model.py`, `crime_nlp/train.py` | Pretrained BERT fine-tuning for classification; CRF BIO sequence labeling. BERT and CRF are the selected alternatives. |
| 3 | Topic modeling | `crime_nlp/train.py`, `crime_nlp/topics.py` | LDA with lemmatized content words, perplexity/coherence/diversity, NMF top-term comparison |
| 4 | POS tagging, named entity recognition, dependency parsing | `crime_nlp/inference.py`, `crime_nlp/ner_model.py`, `app.py` | CRF and fine-tuned BERT NER with eight entity types; spaCy tags and dependencies |
| 4 | Sentiment analysis | `crime_nlp/text.py`, `crime_nlp/enrich.py` | NLTK VADER polarity plus explicitly separate threat-language heuristic |
| 4 | Summarization, text classification, evaluation | `crime_nlp/text.py`, `crime_nlp/train.py` | TextRank, four classifiers, precision/recall/F1, confusion matrices, strict span F1 |

The project requirements select particular examples from the syllabus. Machine translation, GPT, HMM, GloVe and probabilistic n-gram language generation are not additional implemented models. Bigrams are vectorizer features, not a separately trained generative language model. No claim of full coverage of every elective example is made.

## Big Data Analytics

| Unit | Syllabus requirement | Implemented / documented in | Scope |
|---|---|---|---|
| 1 | Big Data properties, architectures, Hadoop / map / reduce | `docs/REPORT.md`, Spark pipeline | Actual partitioned Spark execution; Hadoop ecosystem and MapReduce architecture explained |
| 2 | Scala functional programming, immutable data, collections, traits, patterns | `scala/CrimeAnalytics.scala`, `scripts/run_scala.py` | Executable Scala case classes, trait/object, pattern matching, map/flatMap/filter, closure, list/set/map |
| 3 | Spark architecture, RDD/DataFrame operations, transformations/actions, relational data | `crime_nlp/spark_pipeline.py`, Scala example | Explicit schema, repartition, filter, SHA hash, window deduplication, SQL, count, Parquet writes |
| 3 | Distributed filesystem concepts | `docs/REPORT.md`, `compose.yaml` | HDFS/GFS concepts and shared-path cluster configuration; verified storage is local Parquet |
| 4 | MLlib, classification, pipelines and evaluation | `crime_nlp/spark_pipeline.py` | Actual Spark MLlib HashingTF/IDF/LogisticRegression pipeline with held-out metrics |

HDFS, YARN, GFS and MongoDB are not installed or claimed as running services. Version 2 was executed on a standalone Spark master and two worker processes on one Windows host. See `artifacts/cluster_execution.json` and `spark_metrics.json`. Multi-host performance has not been benchmarked.
