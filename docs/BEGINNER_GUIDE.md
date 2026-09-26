# Start here: what does this project do?

It reads fictional crime descriptions and turns them into categories, highlighted mentions, summaries and searchable patterns. The complete explanation is in **docs/PROJECT_REPORT.pdf**, generated from **docs/REPORT.md** and **docs/RESULTS.md**.

## Follow one example

“Alex reported a stolen phone near Cedar Market. No knife was reported.”

1. **Spark cleans the data.** It rejects empty reports and bad dates and removes duplicates. It keeps the original sentence so highlighted positions remain correct.
2. **Words become numbers.** Bag of Words counts words. TF-IDF emphasizes distinctive words. Word2Vec learns word vectors. BERT considers surrounding context.
3. **A classifier suggests a category.** It learned from labeled training reports. Its suggestion can be wrong.
4. **NER finds mentions.** A sequence tagger can label a victim, property and location. “No knife” is a negated mention, not proof of weapon use.
5. **spaCy explains grammar.** It identifies nouns, verbs, base forms and connections between words.
6. **Topics find recurring language.** LDA groups co-occurring words. A topic is not necessarily a crime category.
7. **VADER measures polarity.** Negative wording is not the same as real danger.
8. **TextRank summarizes.** It copies selected source sentences. It can omit context.
9. **Search retrieves evidence.** Keyword matches words; semantic search matches learned representations; hybrid combines ranks.

## How do we know whether it works?

Training examples teach the model. Validation chooses checkpoints and adjusts confidence. Development provides separate diagnostics. The final test measures frozen models.

Accuracy counts correct document labels. Precision asks how many predictions were correct. Recall asks how many true items were found. F1 balances precision and recall. Strict entity F1 requires the correct complete span and label.

All reported scores come from saved experiment artifacts. The data are synthetic and share grammar across splits, so these scores do not establish accuracy on real police narratives.

## Use the seven pages

- **Overview:** explore generated counts and trends.
- **Case explorer:** search, filter by entity and open a report.
- **Narrative lab:** paste text, choose CRF or BERT NER, then analyze.
- **Model evaluation:** compare measured scores and mistakes.
- **Topics:** inspect recurring words and sample cases.
- **Pipeline & syllabus:** inspect Spark evidence and course coverage.
- **Learning guide:** return to this explanation.

Inside a case, open Syntax & morphology for dependency arrows. Use Evidence search for source sentences and citations. Analysis details explains influential TF-IDF words and allows an explicit local correction. The redacted download masks detected people and contacts; review it before sharing.

## Open the right files

Start with the root **README.md**, then read **docs/PROJECT_REPORT.pdf**. The file **.pytest_cache/README.md** is an automatically created testing-cache note, not the project guide.

Code is under **crime_nlp/**. Settings live in **settings.toml**. Results live in **artifacts/**. Tests live in **tests/**. Large generated data and weights are rebuilt after a fresh GitHub clone.

## Try it yourself

Search “stolen phone” with Keyword, then “someone took a mobile device” with Semantic. Compare the returned text rather than assuming the second method is always better.

Paste a short narrative, inspect the entity spans and sentence arrows, and find one model mistake. Explaining a limitation is part of understanding the project.
