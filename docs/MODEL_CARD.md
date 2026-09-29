# Model card — Version 3.1

Entity training mixes 4,000 existing-corpus reports with 4,000 role/ownership/administrative-context reports. The CRF received additional destination and property-list refinement; both retained entity candidates are compared on the same current validation set. The NER evaluation includes a previously viewed V3 subset and generated context cases, so it is a diagnostic evaluation, not untouched real-source validation. See ENTITY_CONTEXT_V3_1.md. Document models and their V3 benchmark are unchanged.

**Task:** English synthetic narrative classification and entity extraction. **Intended use:** university demonstration and experimentation.

**Document models:** BoW/logistic regression, word + character TF-IDF/logistic regression, train-only Gensim Word2Vec averaging/logistic regression, and fine-tuned compact BERT. TF-IDF is the dashboard's default document classifier for speed and explanation. Character 3–5-grams help tolerate unknown spellings without rewriting names or introducing a new algorithm.

**Entity models:** CRF and fine-tuned BERT token classifier. Validation strict span F1 selects the automatic model; see `artifacts/models/v3_1/ner_selection.json`. Eight entity types are supported. CRF uses lexical context, shape and cue distances; no name/country whitelist is used at inference.

**Search:** local pretrained all-MiniLM-L6-v2 embeddings, 384 dimensions, masked mean pooling, L2 normalization, first 128 subword tokens per case. Semantic retrieval is not trained on case relevance judgments.

**Evaluation:** document sample sizes 12,000/800/800/1,200 across train/validation/development/test. NER uses 8,000/400/600 train/validation/test. Vocabulary fitting and Word2Vec use training only. Validation selects checkpoints and fits temperature. Frozen-model hashes precede final scoring. Report-level bootstrap intervals do not include domain shift. The separate hand-written robustness suite is a development/regression set, not an independent final benchmark.

**Evidence:** artifacts/evaluation.json, ner_evaluation.json, evaluation_plan.json, frozen_models.json, final_evaluation_receipt.json, robustness_development.json and docs/RESULTS.md. Prior version benchmarks are historical because corpus and protocol changed.

**Known limitations:** shared synthetic patterns, possible role confusion, imperfect negation, uncalibrated entity scores, BERT document truncation at 160 subwords, summaries that omit context, and no real-source evaluation. TF-IDF reads the full accepted input. Broader names/places do not imply unrestricted multilingual understanding. VADER polarity and cue levels are not validated danger measures. Review flags combine the original confidence threshold with raw score/margin and vocabulary coverage; they cannot detect every error.

**Privacy:** inference is local after resource downloads. Opt-in feedback stores text hashes and corrected categories, not submitted narratives. Redaction can miss identifiers. Optional shared-password access is not production identity management.

**Updates:** use a new independent evaluation protocol for changes motivated by the viewed final-test results. Never overwrite benchmark labels to hide mistakes.
