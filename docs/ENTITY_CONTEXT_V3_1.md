# Entity roles and exact text — Version 3.1

This revision addresses the reported failures on short ownership/violence sentences and a long FIR-style complaint. It retains the existing eight entity types, CRF, BERT, TF-IDF and all other model families. No name-specific replacement, external language service or new classifier is used.

## Changes

- Entity training now mixes 4,000 reports from the existing corpus with 4,000 generated role/context examples. These include `A stole the item of B`, active/passive violence, international/invented names, honorifics, relatives, witnesses, officers, dotted dates, addresses, serial numbers, phone numbers, legal boilerplate and escape destinations.
- Relatives, recording officers, section numbers and serial-number labels are negative examples for SUSPECT/VICTIM. The role taggers learn these distinctions from context. Generic spaCy entities remain available separately.
- Character offsets are preserved during every generated text edit. Dates, initials and `No.` no longer trigger false sentence breaks in shared sentence processing.
- The annotated narrative and summary escape literal Markdown/HTML and preserve line breaks, tabs and punctuation. Short highlighted spans, including dates, stay together visually. An “Original text, exactly as entered” view is available.
- Model and case-analysis caches include model-file revisions, preventing old cached predictions from surviving a model replacement. The UI displays the active release.
- The original eight crime categories do not include homicide. Inputs mentioning killing/murder show that coverage limitation explicitly instead of presenting the nearest available class without qualification.

## Verification

`tests/test_entity_context.py` contains the user-provided examples, a passive-voice counterpart, and a name substitution. Those exact names and sentences are not inserted into the training generator. `scripts/check_entity_context.py` saves every predicted span in `artifacts/entity_context_regressions.json`. These are development regressions, not an independent final benchmark.

Both existing entity-model families were trained on the broader contexts. The CRF then received a destination-phrasing refinement; the retained BERT candidate is remeasured on the same current validation examples before selecting the default. See `artifacts/models/v3_1/ner_selection.json` and `artifacts/ner_evaluation.json` for measured results. Document/topic weights and the 60,000-report corpus are retained from V3; the library's entity predictions are refreshed. Earlier weights remain in their original model folders.

The NER evaluation combines retained V3 examples with generated context examples. The retained subset has previously been viewed and the generated cases share grammar with training; its score must not be described as independent real-world accuracy. The original document benchmark is unchanged. No claim of correctness for every possible sentence is made.

For a full reproducible rebuild, use `run_pipeline.py --stage train` followed by `--stage enrich`. The standard training path includes the additional entity contexts. The targeted `scripts/refine_entities.py` records the upgrade from the retained V3 model folder; updating an already active CRF requires explicit `--crf-only --update-active` and replaces the model file atomically. An entity-only update can use `run_pipeline.py --stage enrich --entities-only` to retain unchanged classifiers and retrieval indexes while refreshing the case-library spans.

This workspace uses **http://localhost:8502**. The separately running original project occupied port 8501. The launchers and Streamlit configuration now point to 8502 so the corrected copy is unambiguous.
