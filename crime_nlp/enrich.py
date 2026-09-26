"""Materialize searchable predictions and full-corpus sentiment/topic analytics."""
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .config import PROCESSED, MODELS, ARTIFACTS, RUNTIME, write_json
from .text import threat_language, tokens_with_offsets, token_features, spans_from_bio


def enrich():
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.data.path.insert(0, str(RUNTIME / "nltk_data"))
    started = time.perf_counter()
    df = pd.read_parquet(PROCESSED / "reports").sort_values("report_id").reset_index(drop=True)
    tfidf = joblib.load(MODELS / "tfidf.joblib")
    topics = joblib.load(MODELS / "topics.joblib")
    sentiment = SentimentIntensityAnalyzer()
    crf = joblib.load(MODELS / "crf.joblib")
    sentiment_values, risk_values, risk_levels, entities, topic_ids, topic_probs, predictions = [], [], [], [], [], [], []
    for start in range(0, len(df), 1000):
        texts = df.narrative.iloc[start:start + 1000].tolist()
        predictions.extend(tfidf.predict(texts).tolist())
        topic_distribution = topics["model"].transform(topics["vectorizer"].transform(texts))
        topic_ids.extend(topic_distribution.argmax(axis=1).tolist())
        topic_probs.extend(topic_distribution.max(axis=1).tolist())
        token_batches = [tokens_with_offsets(text) for text in texts]
        labels = crf.predict([token_features(tokens) for tokens in token_batches])
        for text, tokens, tags in zip(texts, token_batches, labels):
            sentiment_values.append(sentiment.polarity_scores(text)["compound"])
            threat = threat_language(text)
            risk_values.append(threat["score"])
            risk_levels.append(threat["level"])
            entities.append(json.dumps(spans_from_bio(text, tokens, tags)))
        if start % 10000 == 0:
            print(f"Enriched {min(start + 1000, len(df)):,}/{len(df):,} narratives...", flush=True)
    df["predicted_crime"] = predictions
    df["sentiment"] = sentiment_values
    df["threat_score"] = risk_values
    df["threat_level"] = risk_levels
    df["predicted_entities_json"] = entities
    df["topic_id"] = topic_ids
    df["topic_probability"] = topic_probs
    # Ground truth remains in Spark Parquet for research; UI data contains predicted spans only.
    df.drop(columns=["entities_json"]).to_parquet(PROCESSED / "dashboard.parquet", index=False)
    # Search uses training-fitted IDF; row order matches dashboard.parquet exactly.
    search_matrix = tfidf.named_steps["vectorizer"].transform(df.narrative)
    sparse.save_npz(ARTIFACTS / "search_matrix.npz", search_matrix)
    write_json(ARTIFACTS / "enrichment.json", {"rows": len(df), "seconds": round(time.perf_counter() - started, 2),
              "batch_size": 1000, "classification": "TF-IDF logistic regression", "entities": "CRF predictions",
              "sentiment": "NLTK VADER compound polarity; not an independently validated threat score",
              "threat": "Separate transparent cue heuristic, not calibrated risk", "topics": "LDA",
              "syntax_and_summary": "spaCy morphology/POS/dependencies and TextRank are computed on demand in the dashboard"})
    print("Dashboard dataset and search index saved.", flush=True)


if __name__ == "__main__":
    enrich()
