# Data card — synthetic corpus V2

**Purpose:** university NLP and distributed-processing demonstration. **Origin:** deterministic local generator, seed 42. No real personal records or downloaded crime narratives are included.

**Volume:** 60,000 valid reports; 62,700 raw rows including controlled faults. Eight categories and 384 composite scenario groups. Groups combine category, incident variant and sentence layout; they are not independently authored source documents.

**Annotations:** document categories and eight entity span types generated alongside text. Names and roles are fictional. Unknown suspects, multiple participants, negation and noisy follow-up text are included.

**Splits:** incident variants 0–4 training, 5 validation, 6 development, 7 final test. First-name pools and participant phrasings are split-specific. Shared grammar and vocabulary remain.

**Storage:** raw JSONL, cleaned partitioned Parquet, dashboard Parquet and indexed SQLite. Original text is retained for offset alignment. Raw and processed corpora are excluded from Git.

**Limitations:** generated language lacks the diversity, ambiguity and annotation disagreement of real case files. Charts describe synthetic distributions. Generator labels are not human-reviewed facts.

**External files:** scripts/import_jsonl.py requires source and license metadata and creates an isolated staged dataset. Review permission and annotations before planning a separate real-data experiment.
