# Beginner demonstration guide — Version 2

## Before presenting

Run START_DASHBOARD.cmd in the completed workspace and open http://localhost:8501. A fresh clone requires the setup and full pipeline commands in README.md first. Keep docs/PROJECT_REPORT.pdf and docs/RESULTS.md available.

## A ten-minute walkthrough

1. **Problem (one minute):** “Written descriptions are hard to aggregate. This project turns them into searchable structured information.” State that all 60,000 reports are fictional.
2. **Pipeline (one minute):** Open Pipeline & syllabus. Explain 62,700 raw rows, invalid rows, duplicates, partitions and 60,000 clean rows. Show the actual two-worker single-host execution record.
3. **Trends (one minute):** Open Overview and choose one category. Explain that the chart reflects the generator, not real crime statistics.
4. **Retrieval (two minutes):** Search “stolen phone” in Case explorer. Compare Semantic or Hybrid with a paraphrase. Filter an entity and change result pages. Open a case.
5. **Language (two minutes):** Show entity highlights, structured notes and source sentences. Open Syntax & morphology and explain one noun, verb and dependency arrow. Demonstrate that a negated weapon mention differs from a weapon-use claim.
6. **Models (two minutes):** Show the document-model comparison and one confusion matrix. Explain validation versus final test, macro-F1 and exact-span entity scoring. Show that CRF won validation selection even though BERT is also implemented.
7. **Conclusion (one minute):** Open Learning guide and point to the full report. Name the real-data validation and multi-machine benchmark still needed.

## A pasteable fictional example

On 2025-03-18 at 21:15, Alex Lee reported a stolen phone near Cedar Market. A witness identified Morgan Reed as a suspect. No knife was reported. Officers reviewed camera footage. The account remains an allegation pending review.

Choose Automatic NER, analyze, then compare BERT if time permits. This example is illustrative; do not promise exact outputs in advance. Inspect errors openly.

## Questions you should be able to answer

- Why do word counts provide a useful baseline?
- What does BERT add, and why might it still perform worse?
- What is the difference between a document category and an entity role?
- Why preserve character offsets?
- Why can sentiment not directly measure danger?
- What did Spark actually distribute?
- Why does a high score on synthetic data require careful interpretation?
- What is excluded from GitHub and how is it rebuilt?

Answers and definitions are in Chapters 4–10 and 17 of the report.
