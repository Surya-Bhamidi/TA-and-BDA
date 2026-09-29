# Decoding Crime Narratives using NLP and Big Data Analytics

**Version 3.1** improves informal English, misspellings, unfamiliar names, ownership roles and FIR-style administrative context using the project's existing algorithms. It includes 60,000 fictional reports, Apache Spark, four document classifiers, CRF and BERT entity extraction, LDA/NMF topics, semantic search, spaCy syntax, VADER polarity, TextRank summaries and a seven-page Streamlit dashboard.

The [entity and text-fidelity revision](docs/ENTITY_CONTEXT_V3_1.md) addresses the reported short-sentence and long-FIR failures, including relatives/officers versus actual parties, punctuation, dotted dates and property lists. Model weights are in `artifacts/models/v3_1`; prior releases are retained.

Open **Narrative lab** and try `sir my moblie stoln from pocket in bus ystrday` or `José García hit Wei Zhang with iron rod near Nairobi`. Original spelling and entity offsets are preserved. See [the robustness update](docs/ROBUSTNESS_V3.md) for the constraints, evidence and remaining limits.

## Start here

Read the [beginner guide](docs/BEGINNER_GUIDE.md), then the [complete project report PDF](docs/PROJECT_REPORT.pdf). The report explains the problem, follows a narrative through every stage, teaches each model with examples, explains metrics and includes setup, troubleshooting, a code tour and viva questions.

- [Editable report](docs/REPORT.md) and [standalone HTML report](docs/PROJECT_REPORT.html)
- [Measured results](docs/RESULTS.md)
- [Demonstration walkthrough](docs/DEMO_GUIDE.md)
- [Delivered improvements and remaining work](docs/IMPROVEMENTS_V2.md)
- [Syllabus mapping](docs/SYLLABUS_MAPPING.md), [model card](docs/MODEL_CARD.md), [data card](docs/DATA_CARD.md)

**.pytest_cache/README.md is a pytest cache note, not the project guide.**

## Open the completed local project

Run **START_DASHBOARD.cmd**, then open **http://localhost:8502**. Alternatively, from this folder:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

The pages are Overview, Case explorer, Narrative lab, Model evaluation, Topics, Pipeline & syllabus, and Learning guide. Models and data already exist in the completed local workspace. Inference uses local resources after the initial downloads.

## Get the source and rebuild

```bash
git clone https://github.com/Surya-Bhamidi/TA-and-BDA.git
cd TA-and-BDA
```

Source, tests, documentation, screenshots and measured results are tracked. Large datasets, weights, search indexes, virtual environments and downloaded runtimes are excluded. A fresh clone must run the pipeline.

Use **Python 3.12 and Java 17**. Allow several GB of disk and at least 8 GB RAM. From PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe scripts\download_resources.py
.\.venv312\Scripts\python.exe run_pipeline.py
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

scripts/setup.ps1 automates Windows setup. The build computer uses project-local Java and existing compatible Hadoop native libraries under C:\hadoop\bin. On another Windows machine, set HADOOP_HOME to a trusted compatible distribution. Linux avoids that native-library requirement: create/activate a Python 3.12 venv, install Java 17 and requirements, then run the same scripts with python. Resolved package versions are in requirements-lock.txt.

## Pipeline and reproducibility

```text
Generate JSONL -> Spark clean/deduplicate -> partitioned Parquet
 -> train/validation/development/final-test protocol
 -> model fitting and validation selection -> freeze hashes -> final evaluation
 -> full-corpus enrichment -> SQLite + keyword/semantic indexes -> dashboard
```

Spark processes all 60,000 clean reports. NLP training uses bounded driver-side samples, and enrichment runs in Python batches. The overview reads metadata; selected full narratives come from SQLite.

Version 3 was verified using a standalone master and **two worker processes on one physical host**. It is not a multi-machine scalability benchmark. The default pipeline uses local[2].

```powershell
# Separate stages
.\.venv312\Scripts\python.exe run_pipeline.py --stage generate
.\.venv312\Scripts\python.exe run_pipeline.py --stage spark
.\.venv312\Scripts\python.exe run_pipeline.py --stage train
.\.venv312\Scripts\python.exe run_pipeline.py --stage enrich

# Instead of the local Spark stage: Windows standalone master + two workers
.\.venv312\Scripts\python.exe scripts\run_cluster.py

# Skip only stages whose source/input/output hashes still match
.\.venv312\Scripts\python.exe run_pipeline.py --resume

# Scala functional-programming and Spark Dataset example
.\.venv312\Scripts\python.exe scripts\run_scala.py
```

Stop the dashboard before rebuilding. Changing an upstream stage requires downstream rebuilding. Resume is conservative: source changes invalidate recorded stages. settings.toml centralizes defaults; run receipts and stage hashes document execution.

## Included capabilities

| Area | Implementation |
|---|---|
| Data | 528 scenario groups; formal and informal English, real typo edits, international names/places, eight entity types |
| Document models | BoW, word + character TF-IDF, neural Word2Vec averaging and fine-tuned two-layer BERT |
| Entity models | CRF and BERT BIO tagging; validation-selected default |
| Topics | Lemmatized content words, LDA perplexity/coherence/diversity and NMF top terms |
| Evaluation | Four splits, grouped training CV, frozen hashes, bootstrap intervals, calibration and review flags |
| Search | Keyword, MiniLM semantic and reciprocal-rank hybrid retrieval; entity filters and source-sentence citations |
| Dashboard | Pagination, disk-backed case lookup, feature explanations and a beginner Learning Guide |
| Privacy | Optional shared password, redacted text export, explicit local corrections stored with a text hash |
| Ingestion | Validated external JSONL staging, quarantine and source/license receipt |
| Engineering | Data-quality reports, physical Spark plan, unit/integration/UI tests and CI configuration |

## Read the scores correctly

Document models use 12,000 training, 800 validation, 800 development and 1,200 final-test examples. NER uses 8,000 training, 400 validation and 600 final-test examples. Event variants and name/place pools are split before fitting; grammar and vocabulary remain shared. See the manifest for separate formal and informal split policies.

Vectorizers and Word2Vec fit on training only. Validation selects checkpoints and fits temperature. Model hashes are frozen before final predictions. TF-IDF remains the dashboard category model for speed and explanations. The Automatic entity model comes from the saved validation comparison.

V1 and V2 artifacts are archived under artifacts/baselines. Versions use different data/protocols, so their benchmark scores are not direct improvement estimates. `python scripts/check_robustness.py` compares the retained V2 models and V3 on the same hand-written development examples. Those examples informed development and are not an independent final benchmark. No human-reference accuracy is claimed for syntax, polarity, threat cues, retrieval or summaries. Synthetic trends are not real crime rates.

## External JSONL staging

Each line must contain report_id, narrative, reported_at (ISO date/time), and district. Optional crime_type must be one of the eight categories; optional entities contain start/end/text/label with exact non-overlapping character spans.

```powershell
.\.venv312\Scripts\python.exe scripts\import_jsonl.py examples.jsonl --source-url 'https://source.example/dataset' --license 'permission or license reference'
```

This creates a separate ignored data/external directory with validated rows, rejection reasons and provenance. It does not automatically merge external text into the synthetic benchmark.

## Optional access gate and feedback

Set CRIME_DASHBOARD_PASSWORD in the process environment before starting Streamlit to enable a shared password. This is not SSO or per-user authorization. Submitted text stays in session memory; saving a correction is explicit and stores a hash and categories locally. Redaction is heuristic and must be reviewed before sharing.

## Optional Linux containers

Docker configurations are supplied but were not executed on this machine.

```bash
docker compose build dashboard
docker compose run --rm dashboard python scripts/download_resources.py
docker compose run --rm dashboard python run_pipeline.py
docker compose up -d dashboard
docker compose --profile cluster up -d spark-master spark-worker
docker compose --profile cluster run --rm pipeline
```

Multi-host deployment requires shared storage, reachable driver networking and executor dependencies. HDFS, Kafka, cloud deployment and production identity management are future extensions.

## Verification and report export

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe scripts\export_results.py
```

The export script renders REPORT.md plus actual artifact metrics into RESULTS.md and PROJECT_REPORT.html. Optional scripts/browser_check.py uses Playwright and installed Edge to check the running dashboard, capture screenshots and print the HTML to PROJECT_REPORT.pdf. Install Playwright separately if using that optional tool.

To print an updated HTML report without repeating browser checks, run `python scripts/print_report.py` from an environment with Playwright and Edge.

GitHub Actions runs correctness lint and lightweight unit tests. Full integration tests require generated models/data and run locally. See the measured appendix for the completed run.

## Troubleshooting

- Use the project venv when a package is missing.
- Check Java 17, JAVA_HOME and Windows Hadoop native libraries for Spark gateway failures.
- Run download_resources.py if local model resources are missing.
- Finish all stages if the dashboard shows setup instructions.
- Open the existing dashboard or choose another port if 8501 is occupied.
- Close and reopen PROJECT_REPORT.pdf in the IDE after regeneration.
- Read the report's glossary and code map before editing model internals.
