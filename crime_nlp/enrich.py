"""Materialize searchable predictions and full-corpus sentiment/topic analytics."""
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .config import PROCESSED, MODELS, ARTIFACTS, RUNTIME, SETTINGS, write_json
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
    from .topics import topic_input
    from .evaluation import temperature_scale
    from .privacy import annotate_assertions
    from .features import prediction_review
    calibration = json.loads((MODELS / "calibration.json").read_text())
    selected_ner = json.loads((MODELS / "ner_selection.json").read_text())["selected"]
    if selected_ner == "BERT":
        from .ner_model import load_ner, predict_tags
        entity_model, entity_tokenizer = load_ner()
    confidence_values, review_values = [], []
    sentiment_values, risk_values, risk_levels, entities, topic_ids, topic_probs, predictions = [], [], [], [], [], [], []
    for start in range(0, len(df), 1000):
        texts = df.narrative.iloc[start:start + 1000].tolist()
        raw_probabilities = tfidf.predict_proba(texts)
        probabilities = temperature_scale(raw_probabilities, calibration["temperature"])
        predictions.extend(tfidf.classes_[probabilities.argmax(axis=1)].tolist())
        confidence_values.extend(probabilities.max(axis=1).tolist())
        review_values.extend(prediction_review(tfidf, text, prob, calibration["threshold"], raw)["review_required"]
                             for text, prob, raw in zip(texts, probabilities, raw_probabilities))
        topic_distribution = topics["model"].transform(topics["vectorizer"].transform(topic_input(texts)))
        topic_ids.extend(topic_distribution.argmax(axis=1).tolist())
        topic_probs.extend(topic_distribution.max(axis=1).tolist())
        token_batches = [tokens_with_offsets(text) for text in texts]
        labels = (predict_tags(entity_model, entity_tokenizer, token_batches)[0] if selected_ner == "BERT"
                  else crf.predict([token_features(tokens) for tokens in token_batches]))
        for text, tokens, tags in zip(texts, token_batches, labels):
            sentiment_values.append(sentiment.polarity_scores(text)["compound"])
            threat = threat_language(text)
            risk_values.append(threat["score"])
            risk_levels.append(threat["level"])
            entities.append(json.dumps(annotate_assertions(text, spans_from_bio(text, tokens, tags))))
        if start % 10000 == 0:
            print(f"Enriched {min(start + 1000, len(df)):,}/{len(df):,} narratives...", flush=True)
    df["predicted_crime"] = predictions
    df["sentiment"] = sentiment_values
    df["threat_score"] = risk_values
    df["threat_level"] = risk_levels
    df["predicted_entities_json"] = entities
    df["topic_id"] = topic_ids
    df["topic_probability"] = topic_probs
    df["confidence"] = confidence_values
    df["review_required"] = review_values
    # Ground truth remains in Spark Parquet for research; UI data contains predicted spans only.
    dashboard = df.drop(columns=["entities_json"])
    dashboard.to_parquet(PROCESSED / "dashboard.parquet", index=False)
    from .store import build_store
    build_store(dashboard)
    # Search uses training-fitted IDF; row order matches dashboard.parquet exactly.
    from .features import keyword_vectorizer
    search_matrix = keyword_vectorizer(tfidf).transform(df.narrative)
    sparse.save_npz(ARTIFACTS / "search_matrix.npz", search_matrix)
    from .semantic import encode
    print("Building the neural semantic index...", flush=True)
    vectors = encode(df.narrative.tolist(), progress=True)
    np.save(ARTIFACTS / "semantic_embeddings.npy", vectors.astype(np.float16))
    import hashlib
    write_json(ARTIFACTS / "index_manifest.json", {"rows": len(df), "report_order_sha256": hashlib.sha256("\n".join(df.report_id).encode()).hexdigest(),
               "semantic_model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 384, "max_subword_tokens": 128, "storage": "float16, normalized vectors", "ner_model": selected_ner})
    write_json(ARTIFACTS / "enrichment.json", {"rows": len(df), "seconds": round(time.perf_counter() - started, 2),
              "batch_size": 1000, "classification": "TF-IDF logistic regression + validation temperature calibration", "entities": selected_ner + " predictions",
              "sentiment": "NLTK VADER compound polarity; not an independently validated threat score",
              "threat": "Separate transparent cue heuristic, not calibrated risk", "topics": "LDA",
              "syntax_and_summary": "spaCy morphology/POS/dependencies and TextRank are computed on demand in the dashboard"})
    print("Dashboard dataset and search index saved.", flush=True)
    write_json(ARTIFACTS / "version2_ready.json", {"version": SETTINGS["project"]["version"], "rows": len(df), "ner_model": selected_ner})


def refresh_entity_predictions():
    """Rerun only the existing NER stage when corpus/retrieval are unchanged."""
    from .privacy import annotate_assertions
    from .train import file_hash
    started = time.perf_counter()
    path = PROCESSED / "dashboard.parquet"
    frame = pd.read_parquet(path)
    source = pd.read_parquet(PROCESSED / "reports", columns=["report_id", "narrative"]).sort_values("report_id").reset_index(drop=True)
    if not frame[["report_id", "narrative"]].equals(source):
        raise RuntimeError("Corpus changed; run the full enrichment stage instead.")
    selected = json.loads((MODELS / "ner_selection.json").read_text(encoding="utf-8"))["selected"]
    if selected == "CRF":
        crf = joblib.load(MODELS / "crf.joblib")
    else:
        from .ner_model import load_ner, predict_tags
        bert, tokenizer = load_ner()
    predicted = []
    for start in range(0, len(frame), 1000):
        texts = frame.narrative.iloc[start:start + 1000].tolist()
        batches = [tokens_with_offsets(text) for text in texts]
        tags = crf.predict([token_features(tokens) for tokens in batches]) if selected == "CRF" else predict_tags(bert, tokenizer, batches)[0]
        predicted.extend(json.dumps(annotate_assertions(text, spans_from_bio(text, tokens, labels)), ensure_ascii=False)
                         for text, tokens, labels in zip(texts, batches, tags))
        if start % 10000 == 0:
            print(f"Refreshed entities {min(start + 1000, len(frame)):,}/{len(frame):,}", flush=True)
    frame["predicted_entities_json"] = predicted
    pending = path.with_suffix(".building.parquet")
    frame.to_parquet(pending, index=False)
    pending.replace(path)
    from .store import build_store
    build_store(frame)
    report_path = ARTIFACTS / "enrichment.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(entities=selected + " predictions", entity_refresh_seconds=round(time.perf_counter() - started, 2),
                  entity_model_version=SETTINGS["project"]["version"], entity_model_sha256=file_hash(MODELS / "crf.joblib") if selected == "CRF" else None)
    write_json(report_path, report)
    manifest_path = ARTIFACTS / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(ner_model=selected, entity_model_version=SETTINGS["project"]["version"])
    write_json(manifest_path, manifest)
    write_json(ARTIFACTS / "version2_ready.json", {"version": SETTINGS["project"]["version"], "rows": len(frame), "ner_model": selected})


if __name__ == "__main__":
    enrich()
