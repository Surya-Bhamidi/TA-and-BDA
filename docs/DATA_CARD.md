# Data card — synthetic corpus V3

**Purpose:** university NLP and distributed-processing demonstration. **Origin:** deterministic local generator, seed 42. No real personal records or downloaded crime narratives are included.

**Volume:** 60,000 unique valid reports; 62,700 raw rows including controlled faults. Eight categories and 528 scenario groups: 384 formal composite groups and 144 informal event groups. There are 12,000 formal and 48,000 informal reports. These are not independently authored source documents.

**Annotations:** document categories and eight entity span types generated alongside text. Roles and events are fictional; real city names and varied naming conventions are used as vocabulary examples. Unknown suspects, multiple participants, negation, actual misspellings, short fragments and irregular spacing/casing are included. Gold offsets are transported through every text edit; names are not autocorrected.

**Splits:** formal event variants 0–4 training, 5 validation, 6 development, 7 final test; formal first-name pools and participant phrasings are split-specific. Informal event variants 0–11 training, 12–13 validation, 14–15 development, 16–17 final test, with disjoint explicit full-name and place pools. Randomly invented names add unfamiliar surfaces. All noise styles of one event remain in its split. Normalized duplicate text is rejected globally. Shared grammar and vocabulary remain.

**Storage:** raw JSONL, cleaned partitioned Parquet, dashboard Parquet and indexed SQLite. Original text is retained for offset alignment. Raw and processed corpora are excluded from Git.

**Limitations:** generated language lacks the diversity, ambiguity and annotation disagreement of real case files. Charts describe synthetic distributions. Generator labels are not human-reviewed facts.

**External files:** scripts/import_jsonl.py requires source and license metadata and creates an isolated staged dataset. Review permission and annotations before planning a separate real-data experiment.
