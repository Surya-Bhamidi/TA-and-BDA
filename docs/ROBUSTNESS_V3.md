# Informal English and name generalization — Version 3

Launch **START_DASHBOARD.cmd**, open **http://localhost:8502**, and select **Narrative lab**. Enter your own description; perfect spelling, full sentences and title-cased names are not required. The app preserves the submitted text and highlights spans in that original text.

## What changed within the existing constraints

| Existing component | Revision | Algorithm retained |
|---|---|---|
| Synthetic generator | 12,000 formal and 48,000 informal reports; fragmented sentences, common Indian English phrasing, short reports, actual typos, omitted function words, varied casing and whitespace | Template generation and text preprocessing |
| Names and places | International naming conventions, mononyms, initials, apostrophes, hyphens, diacritics and randomly invented names; explicit pools split before fitting | Contextual sequence labeling, with no name/country whitelist at inference |
| TF-IDF category model | Word unigrams/bigrams plus character 3–5-grams; character fragments retain signal when a word is misspelled | TF-IDF and multinomial logistic regression |
| CRF entities | Wider token context, word shape and distances to action/role/preposition cues; offsets transported through training edits | Linear-chain CRF with L-BFGS and BIO tags |
| Remaining models | Retrained BoW, Word2Vec, BERT classifier, BERT entity tagger, LDA and NMF on the revised splits | The original model families |
| Uncertainty | Low raw confidence, close category scores, unfamiliar vocabulary or too little detail trigger a review message | The existing confidence/review heuristic |
| Corpus processing and retrieval | Rebuilt Spark Parquet, predictions, SQLite, word TF-IDF index and MiniLM index | Existing Spark, keyword/semantic search processes |

No new model family, external correction/translation service, LLM API or package dependency is introduced. Character n-grams are a finer-grained TF-IDF representation, not a new classification algorithm. The system does not autocorrect names or silently rewrite the complaint. spaCy, VADER, threat cues and extractive TextRank remain the existing components.

## Examples

- `sir my moblie stoln from pocket in bus ystrday`
- `two fellows showed knife and took my phone near bus stand`
- `went out came home door lock brokn laptop missing inside`
- `José García hit Wei Zhang with iron rod near Nairobi`
- `paid advance for phone seller disapeared no delivery and blocked my number`
- `my acount hackd pasword changed unable login`

## Evidence and reproducibility

The full dataset has 60,000 unique valid reports and 528 scenario groups. All casing and typo variants of an informal event stay in that event's split. Formal data retains its original split policy. Normalized duplicate narratives are rejected across all splits. Labels are generated from templates; an exact span round-trip test covers every informal event and noise style.

Models are saved under `artifacts/models/v3`. V2 weights remain under `artifacts/models/v2`, with historical metrics in `artifacts/baselines/v2`. Current metrics are in `artifacts/evaluation.json`, `ner_evaluation.json` and `docs/RESULTS.md`. Final model hashes are recorded before final-test predictions. Training uses 12,000 reports for document models and 8,000 for NER.

`python scripts/check_robustness.py` measures the same 32 hand-written classification challenges and eight exact role/location challenges against both releases. See `artifacts/robustness_development.json` for every prediction, including failures. This is explicitly a development/regression comparison: those examples informed the changes. It is separate from the generated final-test benchmark.

The delivered run gets **31/32 category challenges correct**, compared with **23/32** for V2. CRF gets the exact people and location spans in **8/8** challenges, compared with **0/8** for V2. The remaining category error is `house break in happened last nite things taken from cupboard`, classified as Theft instead of Burglary. On the separate 1,200-report final synthetic test, TF-IDF accuracy is **88.25%**; selected CRF strict entity F1 is **91.41%** on 600 reports. These two evaluations have different scopes and should not be combined.

Rebuild with the commands in README. The existing two-worker Spark route is `python scripts/run_cluster.py`; then run the train and enrich stages. Run `python -m pytest -q` for source, model, corpus and dashboard checks.

## What this does not establish

No finite synthetic dataset makes an English classifier flawless for all countries, spellings or descriptions. This release improves the tested informal English cases; it does not claim unrestricted multilingual or Hinglish understanding. The eight original category labels are retained. Multiple incidents, ambiguous descriptions, wholly unfamiliar slang and non-crime input can still produce errors. A review flag helps expose uncertainty but cannot catch every wrong prediction.

Participant roles come from the narrative context and may be allegations. Entity confidence is not calibrated. Spelling errors may still hide a threat cue or change spaCy syntax. TextRank remains extractive, so it preserves broken grammar rather than inventing a corrected statement. The optional BERT document comparison retains its existing 160-subword input limit; default TF-IDF processes the entire accepted narrative. The BERT entity tagger retains its existing long-text windows. Real city names are vocabulary examples in fictional events, not statistics about those cities.
