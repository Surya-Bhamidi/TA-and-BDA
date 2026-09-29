# Decoding Crime Narratives using NLP and Big Data Analytics
## A complete beginner's project report — Version 3.1

The 3.1 entity revision adds generated ownership, active/passive violence and FIR-style administrative contexts to CRF/BERT training, fixes shared punctuation boundaries and preserves exact displayed characters. See ENTITY_CONTEXT_V3_1.md. The NER evaluation now mixes a previously viewed V3 subset with generated context examples; its scope differs from the unchanged document-model benchmark.

Version 3 adds informal English and spelling variation within the original algorithms. Read ROBUSTNESS_V3.md for the precise changes, development challenges and limits. The measured appendix contains the current results; the explicitly labeled V2 discussion is historical.

### How to read this report

Start at Chapter 1 even if you have never used machine learning. Chapters 1–4 explain the problem and the data. Chapters 5–9 explain every NLP component with examples. Chapters 10–13 explain evaluation, the dashboard and how to run the code. Chapters 14–17 cover the syllabus, limitations, improvements and glossary. The measured-results appendix is generated from the actual saved experiment outputs, rather than manually typed scores.

This is an English-language university research prototype. Its 60,000 clean reports describe fictional events. It demonstrates engineering and NLP methods; it does not establish that its predictions are accurate on real police files.

## 1. The problem in ordinary language

A narrative is a description written as sentences. Reading thousands of descriptions manually makes it difficult to count incident categories, find mentions of a weapon, or discover repeated themes.

This project turns a narrative into searchable, structured information. For example, consider this invented illustration:

> On 2025-03-18 at 21:15, Alex Lee reported a stolen phone near Cedar Market. A witness identified Morgan Reed as a suspect. No knife was reported. Officers reviewed camera footage.

The system can suggest a crime category, highlight participants and locations, identify dates and evidence, display sentence grammar, retrieve relevant cases, and create a short extractive summary.

The example illustrates concepts; it is not a claim that every model returns these exact answers. A suspect mention represents a role in an allegation, not a finding of guilt. The phrase “No knife” must not become evidence that a knife was used.

### What goes in and what comes out?

| Input | Operation | Output |
|---|---|---|
| Text descriptions | Spark validation and cleaning | Unique valid reports |
| Clean training text and category labels | Train document classifiers | Suggested category |
| Training text with marked character spans | Train sequence taggers | Entity mentions and roles |
| Unlabeled training text | Topic modeling | Recurring word themes |
| An individual report | spaCy, VADER and TextRank | Grammar, polarity and summary |
| A search query | Keyword or embedding retrieval | Ranked reports and source sentences |

## 2. A map of the entire system

```text
Fictional report generator
    -> raw JSONL: one JSON object per line
    -> Apache Spark: validate, normalize, deduplicate
    -> partitioned Parquet: clean analytical storage
    -> fixed train / validation / development / final-test groups
    -> fit models on training examples
    -> select checkpoints and calibrate on validation
    -> freeze model hashes
    -> evaluate the final synthetic test
    -> enrich all reports using saved models
    -> SQLite case store + keyword index + semantic index
    -> Streamlit dashboard
```

A **pipeline** is this ordered sequence of steps. Each step produces files that the next step reads. Training teaches a model; inference uses a trained model to analyze new text. Opening the dashboard performs inference, not training.

Most expensive processing happens before the browser opens. This is a batch system, not a live police-feed service.

## 3. Dataset: where the reports come from

The project deliberately generates its own dataset, which is allowed by the project requirements. It does not silently substitute synthetic text for an alleged real-world download.

There are eight balanced categories: Assault, Arson, Burglary, Cybercrime, Fraud, Robbery, Theft and Vandalism. Events and roles are fictional. Place vocabulary includes real international cities; no event is a claim about those cities or any actual person. The generator produces 60,000 valid unique examples, then injects known faults so cleaning can be checked:

| Injected condition | Rows | Why include it? |
|---|---:|---|
| Duplicate reports | 1,500 | Check deduplication |
| Null narratives | 600 | Check missing-value rejection |
| Blank narratives | 300 | Check whitespace validation |
| Invalid dates | 300 | Check date parsing |
| Total raw rows | 62,700 | Input to Spark |

Version 3 includes **528 scenario groups**: 384 formal composite groups and 144 informal event groups. There are 12,000 formal reports and 48,000 informal reports. Informal inputs include short fragments, Indian English phrasings, actual misspellings, omitted function words, irregular spacing and varied casing. These are generated groups, not independently authored stories. Many words and grammar patterns remain shared.

Formal participant phrases and first-name pools differ by split. Informal event patterns, full-name pools and place pools differ by split; randomly invented names add unfamiliar surfaces. All noise variants of an event remain in its assigned split. Normalized duplicates are rejected across all splits. Some formal stories also have unidentified suspects, multiple participants and negated weapons. Original source spans remain exact through every training-text edit.

### A record and its fields

A JSON object is a set of named values. JSONL stores one such object per line. The generated record includes:

| Field | Meaning |
|---|---|
| report_id | Stable identifier used to find a case |
| narrative | Original text, preserved for display and offsets |
| crime_type | Generator's document-category label |
| district, reported_at | Fictional location grouping and timestamp |
| template_group, split | Scenario family and assigned evaluation partition |
| entities_json | Gold entity labels and character positions |

“Gold” means the reference annotation used for scoring. Here it is created by the generator, not a human annotator. For “Alex Lee,” a span could be start 0, end 8. Python uses an exclusive end: `text[0:8]` returns the eight characters.

Preserving original text matters: removing characters before NER would shift these positions and highlight the wrong words.

### External data import

`scripts/import_jsonl.py` validates a separately supplied JSONL file, checks IDs, dates, text length and optional entity offsets, and records source URL, license and SHA-256 hash. Invalid rows are reported by line number and reason. The importer writes to an ignored staging directory.

It does **not** automatically mix external data into the benchmark or train on it. A real-data experiment needs permission, a label mapping, reviewed annotations and a separate split design first. The source URL and license arguments document provenance; entering them does not itself prove that use is permitted.

## 4. Big Data processing with Apache Spark

### Why Spark?

A normal Python script often processes a dataset in one process. Spark divides a DataFrame into **partitions** and schedules work on executors. A DataFrame resembles a table with named columns and known types.

A **driver** plans the work. A **master** coordinates worker resources in standalone mode. A **worker** hosts executor processes. Executors run tasks over partitions. An **action**, such as counting or writing data, triggers execution; transformations such as filtering describe the work to do.

This project is a modest corpus demonstrating distributed techniques. Sixty thousand reports do not require a production-scale cluster.

### Cleaning steps

1. Read JSONL using an explicit schema rather than guessing all column types.
2. Reject missing or blank text and dates that cannot be parsed.
3. Keep original text and create a normalized lowercase text column.
4. Calculate a SHA-256 text hash and remove repeated normalized narratives.
5. Validate split assignments and retain one record for each duplicate group.
6. Write partitioned Parquet and aggregate counts.
7. Train and evaluate a distributed MLlib classification baseline.

Parquet is a column-oriented file format. Reading only the columns needed for a chart can reduce I/O compared with loading every long narrative. Partition folders organize reports by their split.

A **shuffle** redistributes records across executors. Deduplication and grouped aggregation may require shuffles; they can be more expensive than filtering a partition independently.

### What was actually executed?

Version 3 was run against a standalone master and **two worker processes on this one Windows computer**. The run processed all 60,000 clean reports. `artifacts/cluster_execution.json` records worker registration and completion; `spark_metrics.json` records the actual master and counts. `spark_execution_plan.txt` exposes the physical plan.

This is separate-worker execution on one host, not proof of multi-machine scalability, fault tolerance under machine failure, or an HDFS deployment. The default simpler reproduction uses `local[2]`, which runs local Spark tasks without a separate master.

### MLlib baseline

The MLlib pipeline is:

`RegexTokenizer -> StopWordsRemover -> HashingTF -> IDF -> LogisticRegression`

It converts text into numerical features, learns a category classifier, and evaluates a held-out partition. It uses more training and test rows than the four-model Python comparison. Therefore its score is informative but not an equal-sample model comparison.

Hadoop is an ecosystem; HDFS is distributed storage; MapReduce is a map/shuffle/reduce processing model. Spark can work with Hadoop storage, but Spark and Hadoop are not the same product. Here storage is local Parquet. HDFS, YARN, MongoDB and Kafka are not running services.

## 5. Preprocessing, grammar and morphology

**Tokenization** divides text into words or subword pieces. **Lemmatization** maps inflections to dictionary forms: “reported” becomes “report.” **POS tagging** labels grammatical roles such as noun, verb and adjective. **Morphology** describes features such as tense, number or verb form.

A **dependency parse** links a word to the word it depends on. In “Alex reported a theft,” “reported” may be the root verb, “Alex” its subject, and “theft” its object. spaCy predicts these relationships. The dashboard draws arrows and also lists tokens, lemmas, POS tags, morphology and heads.

Different tasks need different preprocessing. Document vectorizers tokenize and normalize internally. Topic modeling uses selected noun, verb and adjective lemmas and suppresses names and common reporting words. NER retains word shapes, capitalization and character offsets because they help identify names.

The spaCy model is pretrained. The project does not train a dependency parser or report a syntax accuracy score without a gold syntax test set.

## 6. How words become numbers

Machine-learning models consume numerical representations. This project compares three representation families on the same document split.

### 6.1 Bag of Words

Suppose the vocabulary is [phone, stolen, fire]. “Stolen phone” becomes [1, 1, 0]; “fire fire” becomes [0, 0, 2]. This representation counts words and largely ignores order. Bigrams add adjacent pairs such as “stolen phone,” giving limited local order information.

The implementation uses CountVectorizer and logistic regression. Logistic regression learns a weight for each feature and class, then converts scores into class probabilities.

### 6.2 TF-IDF

Term Frequency–Inverse Document Frequency gives more weight to words that distinguish a document from the rest of the training corpus. A word appearing in nearly every report usually provides less information.

A common smoothed inverse frequency is:

`IDF(word) = log((1 + number_of_documents) / (1 + documents_containing_word)) + 1`

In a corpus of 100 documents, a word appearing in 99 has IDF about 1.01; a word appearing in 4 has IDF about 4.01. Actual feature values also depend on the vectorizer's term-frequency and normalization settings. These numbers are a teaching illustration, not saved experiment features.

The vocabulary and IDF are fitted on training text only. Fitting them on the test set would leak information about the evaluation distribution.

Version 3 combines word unigrams/bigrams with character 3–5-grams in the same TF-IDF representation and logistic regression model. A typo can lose its whole-word match while retaining useful character fragments. Character features receive weight 0.75 relative to the word branch. This does not autocorrect the source and does not introduce a new classification algorithm. Keyword search continues to use the word branch so a misspelled string cannot become an unexplained exact-keyword match.

### 6.3 Neural Word2Vec embeddings

Word2Vec is a small neural model trained to learn contextual relationships between words. This project uses Gensim skip-gram with 100-dimensional vectors, trained on its training partition. A document vector is the average of its known word vectors, followed by logistic regression.

Averaging is efficient but loses word order. “Alex accused Morgan” and “Morgan accused Alex” can have similar averages even though their roles differ. Unknown words contribute no vector; an entirely unknown document gets a zero vector.

### 6.4 Contextual BERT representations

BERT uses transformer attention so a word's representation depends on surrounding words. The project fine-tunes a compact pretrained BERT for document classification. Training updates model parameters; a randomly initialized category head alone is not a useful pretrained crime classifier.

The checkpoint is selected using validation macro-F1. A larger neural model is not automatically better than a simple word-count model, especially on repetitive synthetic data with a small training sample.

The BERT document classifier has an input token limit; long narratives are truncated by its tokenizer. BERT entity extraction uses overlapping word windows instead. These are two separately fine-tuned tasks.

## 7. Sequence labeling: finding entity roles

Document classification assigns one category to a complete narrative. Sequence labeling assigns a label to each token.

Eight entity types are supported: SUSPECT, VICTIM, LOCATION, WEAPON, PROPERTY, DATE, TIME and EVIDENCE.

### BIO notation

| Token | Label |
|---|---|
| Alex | B-VICTIM |
| Lee | I-VICTIM |
| reported | O |
| a | O |
| phone | B-PROPERTY |

B means beginning, I means inside the same entity, and O means outside an entity. Both span boundaries and entity type must be correct for strict entity scoring.

### CRF

A Conditional Random Field learns relationships between nearby labels and features such as capitalization, neighboring words, position and sentence context. It can learn that a name after a reporting phrase often describes a victim, while a name after an identification phrase may be a suspect. These are learned patterns, not guaranteed rules.

Version 3 uses a four-token lexical window on each side plus word shape and distances to action, role and preposition cues within eight tokens. Training includes lowercase and invented names, active/passive constructions and varied name lengths. Cue observations do not directly assign roles: the same linear-chain CRF learns their weights. No inference-time name or country list is used.

### BERT token classifier

A second model fine-tunes BERT to predict BIO labels. BERT may split one word into multiple subwords; only the first subword receives a training label and other pieces use the ignored loss value -100. Predictions are mapped back to original words.

Overlapping windows reduce boundary losses in long cases. Probabilities for repeated words are averaged, and invalid orphan I-tags are repaired to begin a span. Extremely unusual text can still challenge tokenization.

### Which model is shown?

Both models are trained and scored on validation. The higher validation strict span F1 wins; ties prefer the faster CRF. The current selection is saved in the model folder and shown in the dashboard. Users can explicitly compare both in Narrative Lab. Model choice is not changed after inspecting final-test scores.

A short preceding-word heuristic marks mentions such as “No knife” as negated. It is not full logical reasoning and may mishandle complicated negation or quoted allegations. Entity confidence is a model score, not verified factual confidence.

## 8. Topics, sentiment and summaries

### Topic modeling: LDA and NMF

A topic is a group of words that often occur together. Latent Dirichlet Allocation represents a document as a mixture of topics. A report could contain both a property theme and a digital-access theme. Topic numbers are arbitrary identifiers, not crime labels.

Training uses lemmatized content words and reduces proper names and routine reporting vocabulary. LDA's top words, document-topic probabilities, held-out perplexity, UMass coherence and top-term diversity are saved. NMF provides a second unsupervised factorization and top-term comparison.

Lower perplexity indicates better fit under a fixed vocabulary and protocol; it does not guarantee more meaningful topics. Coherence measures word co-occurrence; diversity measures how many top words are distinct. Human review is still needed to decide whether themes are useful. NMF is not presented as a fully evaluated replacement selected by human topic judgments.

### VADER sentiment and threat-language cues

NLTK VADER produces a compound polarity score between -1 and +1. Negative language is not equivalent to high danger. A neutral report can describe a serious event, and frightened wording can appear in a low-severity event.

A separate transparent heuristic counts selected unnegated threat-language cues. It is shown as a demonstration cue level, not a probability of future violence or an operational risk assessment. There is no labeled danger dataset in this project.

### TextRank summarization

TextRank treats sentences as nodes in a graph, connects similar sentences, and ranks them using PageRank. The implementation also includes a small lead-sentence preference. Selected sentences are copied from the source and placed in source order.

This is **extractive** summarization. It does not generate a new account. It can still omit an important denial or overemphasize repeated material. Structured notes list extracted mentions and source sentence references; users should check them against the narrative.

No ROUGE score is claimed because human reference summaries have not been collected.

## 9. Search, evidence retrieval and explanation

Keyword search uses TF-IDF cosine similarity. It works well when query and report share terms. Semantic search uses the pretrained all-MiniLM-L6-v2 encoder, attention-mask mean pooling and normalized 384-dimensional embeddings.

Each case embedding uses at most 128 subword tokens for bounded CPU cost. Later details in a long case can be missed. The index stores float16 vectors on disk and loads them through a memory map. Query scoring reads chunks instead of copying the entire embedding matrix.

Hybrid search combines keyword and semantic **ranks**, using reciprocal-rank fusion. It does not pretend that the two raw similarity scales are identical. Entity filters use predicted mentions stored in SQLite; missed entities can therefore cause missed filtered results.

For a question within a case, the app retrieves up to two matching source sentences, displays the report ID and sentence numbers, and returns “Insufficient evidence” below its threshold. This is evidence retrieval, not a general conversational assistant or verified question-answering model. A retrieved sentence may be relevant without answering the question.

TF-IDF prediction explanations show positively contributing words, phrases and character fragments, with their feature type. They describe correlations in a linear classifier, not causal reasons why a crime occurred.

## 10. Evaluation: how to judge the results honestly

### Four data roles

| Split | Use | V3 sample for document models |
|---|---|---:|
| Training | Fit parameters and feature vocabulary | 12,000 |
| Validation | Select checkpoints and fit temperature | 800 |
| Development | Separate diagnostic reporting | 800 |
| Final test | Score frozen models | 1,200 |

The complete corpus has larger split partitions. These samples keep the CPU experiment practical. NER uses its own documented subsets: 8,000 training reports, 400 validation reports and 600 final-test reports.

Incident variants, participant phrasings and first-name pools are assigned before fitting. Composite groups do not overlap between splits. Three-fold grouped cross-validation of TF-IDF uses incident variants within training, keeping related layouts together.

A SHA-256 receipt records model files before final predictions. The final-test receipt records completion. Once results have been viewed, the test is no longer unseen to the researchers; future tuning should use a new independent evaluation protocol.

The corpus still shares generator vocabulary and grammar across splits. This protocol reduces some leakage but cannot establish real-world generalization.

### Metrics with small examples

If 80 out of 100 report categories are correct, accuracy is 80%.

If a model marks 10 names as victims and 8 are correct, precision is 8/10. If 12 actual victim mentions existed, recall is 8/12. F1 balances precision and recall through their harmonic mean.

Macro-F1 gives each category equal weight. Weighted-F1 weights categories by their number of examples. A confusion matrix places reference labels on rows and predicted labels on columns; off-diagonal counts show mistakes.

Strict entity F1 requires an exact span and type match. Correctly finding “Alex” when the gold span is “Alex Lee” is not a correct full-span prediction.

### Uncertainty and calibration

The report includes a 95% accuracy interval from 400 report-level bootstrap resamples. It captures variation when resampling this synthetic test, not uncertainty from new templates, new sources or new cities.

A score of 0.9 is useful only if similarly scored predictions are correct roughly 90% of the time. Temperature scaling adjusts TF-IDF probabilities using validation labels. It preserves the winning category and changes score sharpness.

Expected Calibration Error (ECE) compares average confidence with accuracy in bins. Lower is generally better under the same evaluation. Calibration can become worse when the data distribution changes, so before/after results are reported honestly.

Predictions below the configured 0.60 score are marked for review. The app also flags fewer than three words, low training-vocabulary coverage, raw confidence below 0.45 or a raw top-two margin below 0.15. These are transparent heuristics, not a separate model. Reported calibration coverage measures only the 0.60 calibrated-score threshold, not the combined UI policy. Accuracy among accepted predictions must be read alongside coverage: accepting very few easy cases can inflate that accuracy.

### Comparing versions

V1 artifacts are retained in `artifacts/baselines/v1/`. V2 changes the corpus, entities, splits and training protocol. A higher V2 NER score therefore does not isolate the benefit of one algorithm. It is not an apples-to-apples causal improvement claim. The measured appendix gives all current model scores, including disappointing results.

### Historical V2 results (archived; not the current benchmark)

Word2Vec document classification reaches 95.17% accuracy on the fixed final sample, compared with 83.67% for TF-IDF and 83.25% for compact BERT. This is evidence about this particular experiment, not a universal model ranking.

TF-IDF's training-only grouped cross-validation macro-F1 is only 0.2502, much lower than its 0.7934 final-test macro-F1. These different held-out variant sets have very different difficulty. The disparity warns that the final split alone gives an incomplete picture of robustness.

CRF's overall strict entity F1 is 0.9846, but PROPERTY recall is only 0.44 and its F1 is 0.6111. More frequent, easier entity types dominate the overall score. This is why a useful evaluation shows per-type results, not just one impressive headline number. BERT entity F1 is 0.8741 on the final sample.

TF-IDF calibration lowers final-test ECE from 0.3830 to 0.0757 in this run. At the 0.60 threshold, calibrated coverage is 92.17% and accepted accuracy is 85.80%. These values describe confidence behavior on the synthetic sample; they do not certify real-world correctness.

Syntax, polarity, threat cues, summaries and retrieval do not yet have independent human-labeled evaluation sets. Do not invent accuracy numbers for them.

## 11. Dashboard: a guided tour

1. **Overview:** filter synthetic counts by category, district and reporting period. Trends describe the generator, not a real city's crime rate.
2. **Case explorer:** choose Keyword, Semantic or Hybrid search; filter predicted entity types or values; move between result pages; open a case.
3. **Narrative lab:** paste English text or upload UTF-8 .txt, choose Automatic/CRF/BERT entity extraction, and analyze. The limit is 30,000 characters.
4. **Model evaluation:** compare classifiers, inspect confusion matrices, entity scores, calibration and training history.
5. **Topics:** inspect top words, temporal counts, example reports and the NMF comparison.
6. **Pipeline & syllabus:** inspect cleaning counts, Spark execution evidence and syllabus mapping.
7. **Learning guide:** read the short explanation directly inside the app.

Within an analysis, the tabs show entity highlights and summaries, grammatical dependencies, source-sentence evidence search, and prediction details. JSON exports contain the full analysis. The separate redacted-text download masks detected people and contact details but can miss information.

The overview loads metadata columns. Full narrative text is fetched from SQLite for selected records. CSV export applies to the visible result page, avoiding an unexpectedly huge download.

The optional shared password is set through `CRIME_DASHBOARD_PASSWORD`. It is a basic local access gate, not SSO, per-user roles or a production authentication system. Keep the demonstration bound to localhost unless deploying it with appropriate hosting controls.

User-submitted text is analyzed in memory. Saving a correction is an explicit action; the local feedback database stores its text hash, predicted class and corrected class. It does not automatically retrain a model. Clear the session to discard its in-memory analysis.

## 12. Run the project step by step

### On the completed development computer

Open the project folder, run `START_DASHBOARD.cmd`, and visit http://localhost:8502. Read the root README, not `.pytest_cache/README.md`: that cache file belongs to the testing tool.

### A fresh Windows installation

Install Python 3.12 and Java 17. Use a trusted compatible Hadoop native distribution on Windows and set HADOOP_HOME if needed. The supplied setup script can prepare the local Java and Python resources; Linux avoids Windows Hadoop native-library issues. Allow several GB for packages, data and downloaded models.

Open PowerShell in the repository folder:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe scripts\download_resources.py
.\.venv312\Scripts\python.exe run_pipeline.py
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

A virtual environment is a project-specific collection of Python packages. Using its full Python path ensures that installation and execution use the same interpreter.

The first resource download needs internet. Ordinary inference uses local models afterwards. The GitHub repository excludes large generated data, weights and runtimes; a fresh clone must rebuild them.

On Linux, create a Python 3.12 virtual environment, activate it, install Java 17 and the requirements, then run the same scripts using `python`. The separate native standalone launcher is intended for the tested Windows host; the provided container configuration is an unverified deployment option.

### Run separate stages

```powershell
.\.venv312\Scripts\python.exe run_pipeline.py --stage generate
.\.venv312\Scripts\python.exe scripts\run_cluster.py
.\.venv312\Scripts\python.exe run_pipeline.py --stage train
.\.venv312\Scripts\python.exe run_pipeline.py --stage enrich
```

The cluster launcher runs the Spark stage against its master and two workers, then stops only processes it started. Do not run another Spark rebuild concurrently.

`--resume` reuses a stage only when recorded source, settings, inputs and output hashes match. Hashing is conservative and can take time. Changes to source can invalidate earlier stages even when a narrower dependency analysis would permit reuse.

Stop the dashboard before rebuilding. Upstream changes require downstream rebuilding. Do not retrain merely to obtain a more flattering final-test score.

### Troubleshooting

| Symptom | What to check |
|---|---|
| Module not found | Use .venv312 Python and install requirements there |
| Java gateway fails | Java 17, JAVA_HOME, available ports and Windows Hadoop libraries |
| Dashboard shows setup | Finish enrichment and check its log; all V2 indexes must exist |
| Address already in use | An existing dashboard may already be running on port 8501 |
| Missing pretrained files | Run download_resources.py with network access |
| No search matches | Broaden filters; Keyword search requires vocabulary overlap |
| Model score seems wrong | Inspect the narrative and model limitations; never silently relabel gold data |
| PDF still looks old in the IDE | Close and reopen the regenerated PROJECT_REPORT.pdf |

## 13. Code and artifact tour

| File or folder | Responsibility |
|---|---|
| settings.toml | Seeds, training sizes, thresholds and Spark defaults |
| run_pipeline.py | Stage ordering, arguments, hashes and run receipts |
| crime_nlp/generate.py, corpus_patterns.py | Fictional corpus and reference spans |
| crime_nlp/spark_pipeline.py | Distributed cleaning, quality reports and MLlib |
| crime_nlp/train.py | Shared splits, classical models, topics and final scoring |
| crime_nlp/bert_model.py, ner_model.py | BERT document and token classification |
| crime_nlp/evaluation.py | Metrics, calibration and BIO repair |
| crime_nlp/inference.py | Assemble one narrative's complete analysis |
| crime_nlp/text.py, topics.py | Text utilities, summary, cue heuristic and topic preprocessing |
| crime_nlp/semantic.py | Sentence embeddings, fusion and evidence retrieval |
| crime_nlp/store.py, privacy.py | SQLite lookup, feedback, mention assertions and redaction |
| crime_nlp/enrich.py | Batch predictions and search indexes |
| crime_nlp/audit.py, import_data.py | Hash receipts and external staging validation |
| app.py | Streamlit pages and interactions |
| scala/CrimeAnalytics.scala | Scala and Spark Dataset exercise |
| tests/ | Unit, artifact and dashboard checks |
| artifacts/ | Measured metrics, split IDs and reproducibility evidence |
| docs/ | Beginner guide, report, measured results, cards and demonstration |

To trace a feature, start with its dashboard control, find the called inference function, then find the saved model it loads and the training function that created it. This is more manageable than reading every file from top to bottom.

Tests check behavior rather than proving real-world model reliability. Unit tests cover offsets, negation, calibration, imports and privacy. Integration tests check 60,000-row output alignment, split separation, model inference and dashboard pages. CI runs lightweight tests without rebuilding the full dataset.

## 14. Syllabus coverage and the Scala example

The detailed table is in SYLLABUS_MAPPING.md. NLP foundations appear in tokenization, morphology and parsing; representations in BoW, TF-IDF and neural embeddings; sequence models in BERT and CRF; applications in NER, sentiment and summarization.

Big Data topics appear in Spark partitions and execution, relational DataFrames and SQL, transformations/actions, Parquet and MLlib pipelines. The Scala file demonstrates immutable values, collections, case classes, traits, pattern matching, closures and map/filter-style operations.

BERT and CRF are chosen from the requested alternatives. GPT, HMM, machine translation, GloVe, streaming ingestion and a generative n-gram language model are not additional implemented systems. Bigrams in a vectorizer are features, not a probabilistic text generator.

Run `python scripts/run_scala.py` inside the environment to execute the Scala example against the processed data. The report does not imply that the main Python dashboard was rewritten in Scala.

## 15. What improved across versions?

Version 3 broadens the data to informal English and international names, preserves spans through real noise edits, uses word and character TF-IDF in the existing classifier, expands the CRF context features and retrains the original model families. It adds direct name/role checks, explicit input-quality review reasons and a hand-written development comparison against retained V2 weights. ROBUSTNESS_V3.md documents the constraints and remaining limits. No external spell checker, translation service or new model family was added.

The earlier Version 2 changes are retained:

The upgrade adds varied composite scenarios, eight entity types, a separately fine-tuned BERT entity tagger, validation-based tagger selection, a final-test freeze receipt, grouped cross-validation, bootstrap intervals, calibration and review flags.

It also adds cleaner topic inputs and NMF comparison; real two-worker single-host Spark execution and quality reports; semantic/hybrid retrieval and cited evidence sentences; a disk-backed case store and pagination; prediction explanations, redacted downloads and opt-in corrections; a validated external import staging path; and this beginner documentation.

These are implemented features, not guarantees that all scores increased. Refer to IMPROVEMENTS_V2.md for the distinction between delivered work and future extensions.

## 16. Limits and useful future work

The highest-value next experiment is an independently annotated, permissioned real narrative dataset with double annotation and disagreement review. It should have a source-separated test set, not a random split of near-duplicate stories.

Other future work includes human judgments for retrieval and summaries, explicit false-negative analysis for redaction, multilingual evaluation, temporal drift monitoring, and calibrated entity scores. Scalability work should measure throughput and memory across physical machines, then choose a distributed store or approximate nearest-neighbor index based on observed need.

Streaming, HDFS, cloud deployment, OAuth/RBAC and centralized audit retention require additional infrastructure and are not claimed complete. A synthetic demonstration should not be used to rank real individuals, predict criminality or automate consequential decisions.

## 17. Glossary and viva questions

| Term | Plain-language meaning |
|---|---|
| Corpus | A collection of texts |
| Feature | A numerical signal used by a model |
| Label | The answer a supervised model learns to predict |
| Embedding | A dense learned vector representing text |
| Epoch | One pass through training examples |
| Hyperparameter | A setting chosen outside the fitted model weights |
| Overfitting | Learning training-specific patterns that fail on new data |
| Leakage | Evaluation information influencing training or selection |
| Inference | Running a trained model on input |
| Calibration | Aligning confidence scores with observed correctness |
| NER | Finding typed mentions inside text |
| Partition | One portion of a distributed dataset |
| Checkpoint | Saved model parameters |
| Provenance | Where data came from and how it was processed |
| Hash | A fingerprint used to detect content changes |

**Why compare BoW with BERT?** To test whether contextual neural complexity helps under the same data protocol, instead of assuming that it must.

**Why preserve the raw narrative?** So highlights and source citations remain anchored to the exact input.

**Why use CRF alongside BERT?** They provide different sequence-modeling approaches; validation determines which works better here.

**Does high synthetic accuracy prove deployment readiness?** No. Shared generation patterns and missing real-source evaluation remain major limitations.

**Why not call sentiment a danger score?** Polarity measures emotional language, while danger needs a different labeled task and validation.

**Why have both Parquet and SQLite?** Parquet serves batch analytics and selected-column reads; SQLite serves small indexed case lookups and local feedback.

**What proves Spark ran?** Saved execution metrics, the physical plan, standalone worker registration and successful output invariants—not just importing PySpark.

## References and further reading

- [Apache Spark 3.5.5 documentation](https://spark.apache.org/docs/3.5.5/) — architecture, SQL and MLlib.
- [BERT paper](https://aclanthology.org/N19-1423/) — contextual transformer pretraining.
- [Hugging Face token classification](https://huggingface.co/docs/transformers/tasks/token_classification) — subword/label alignment.
- [MiniLM sentence encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — local semantic representation.
- [Word2Vec paper](https://arxiv.org/abs/1301.3781) — neural word representations.
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) — interpreting classifier probabilities.
- [spaCy linguistic features](https://spacy.io/usage/linguistic-features) — syntax and morphology.
- [TextRank paper](https://aclanthology.org/W04-3252/) — graph-based ranking for text.
- [NLTK VADER documentation](https://www.nltk.org/api/nltk.sentiment.vader.html) — lexicon-based polarity analysis.
