# Model card — Version 2

**Task:** English synthetic narrative classification and entity extraction. **Intended use:** university demonstration and experimentation.

**Document models:** BoW/logistic regression, TF-IDF/logistic regression, train-only Gensim Word2Vec averaging/logistic regression, and fine-tuned compact BERT. TF-IDF is the dashboard's default document classifier; this is an explicit speed/explanation choice, not a claim that it has the best final score.

**Entity models:** CRF and fine-tuned BERT token classifier. Validation strict span F1 selects the automatic model. Eight entity types are supported. CRF is selected in this executed release.

**Search:** local pretrained all-MiniLM-L6-v2 embeddings, 384 dimensions, masked mean pooling, L2 normalization, first 128 subword tokens per case. Semantic retrieval is not trained on case relevance judgments.

**Evaluation:** document sample sizes 6,400/800/800/1,200 across train/validation/development/test. NER uses 3,200/400/600 train/validation/test. Vocabulary fitting and Word2Vec use training only. Validation selects checkpoints and fits temperature. Frozen-model hashes precede final scoring. Report-level bootstrap intervals do not include domain shift.

**Evidence:** artifacts/evaluation.json, ner_evaluation.json, evaluation_plan.json, frozen_models.json, final_evaluation_receipt.json and docs/RESULTS.md. V1 comparisons are historical because corpus and protocol changed.

**Known limitations:** shared synthetic patterns, possible role confusion, imperfect negation, uncalibrated entity scores, truncated document inputs, summaries that omit context, and no real-source evaluation. VADER polarity and cue levels are not validated danger measures.

**Privacy:** inference is local after resource downloads. Opt-in feedback stores text hashes and corrected categories, not submitted narratives. Redaction can miss identifiers. Optional shared-password access is not production identity management.

**Updates:** use a new independent evaluation protocol for changes motivated by the viewed final-test results. Never overwrite benchmark labels to hide mistakes.
