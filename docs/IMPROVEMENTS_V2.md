# Version 2 delivery and remaining work

## Implemented

| Area | Delivered change | Evidence |
|---|---|---|
| Corpus | 384 composite groups, varied layouts/participants, unknown and multiple participants, eight entity types | generate.py, corpus_patterns.py, dataset_manifest.json |
| Spark | Two standalone workers on one host, quality report and physical plan | run_cluster.py, cluster_execution.json, data_quality.json |
| Evaluation | Four splits, grouped training CV, frozen model hashes, bootstrap accuracy intervals | evaluation_plan.json, frozen_models.json, evaluation.json |
| Confidence | Validation-fit temperature, before/after calibration, review threshold | evaluation.py, calibration metrics, dashboard |
| Entities | CRF and fine-tuned BERT token classification, validation selection, mention negation | ner_model.py, ner_evaluation.json |
| Topics | Content-word lemmatization, LDA coherence/diversity and NMF top terms | topics.py, topics.json |
| Retrieval | Local MiniLM embeddings, keyword/semantic/hybrid ranking, cited source sentences | semantic.py, index_manifest.json |
| Dashboard | Entity filters, pagination, SQLite case lookup, explanations, learning page | app.py, store.py |
| Privacy | Optional password, redacted text export, explicit correction with text hash | privacy.py, store.py |
| Ingestion | JSONL validation, quarantine and source/license staging receipt | import_data.py, import_jsonl.py |
| Reproduction | Central settings, stage hashes, append-only run receipts, tests and CI | settings.toml, audit.py, tests/, workflow |
| Teaching | Rewritten report, beginner guide, demo guide, model/data cards | docs/ |

## Deliberate limits

The new synthetic benchmark differs from V1; score changes are not controlled ablations. V1 metrics are archived under artifacts/baselines/v1. Real reports are not included, and staging an external file does not automatically train a new benchmark.

No physical multi-host or cloud benchmark, HDFS/YARN service, Kafka stream, SSO/RBAC, production audit system, multilingual evaluation, human-reference summary evaluation, or human-judged search benchmark is claimed. Those require additional infrastructure or independently labeled data. CI configuration is included; its remote execution must be checked on GitHub.

## Recommended next experiments

Obtain permissioned real text and independent annotations. Measure entity agreement, retrieval relevance, summary coverage and redaction misses. Keep a source-separated test set. Then benchmark memory and throughput before deciding whether approximate vector search, a distributed database or streaming infrastructure is needed.
