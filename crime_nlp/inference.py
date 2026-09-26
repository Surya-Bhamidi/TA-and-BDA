"""Offline narrative analysis, model loading and safe entity highlighting."""
import html
import joblib
from functools import lru_cache

from .config import MODELS, RUNTIME
from .text import tokens_with_offsets, token_features, spans_from_bio, summarize, threat_language

COLORS = {"SUSPECT": "#ffd9c9", "VICTIM": "#d9e6ff", "LOCATION": "#c4eee5", "WEAPON": "#ffe8ad"}


@lru_cache(maxsize=1)
def language_model():
    import spacy
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def models():
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.data.path.insert(0, str(RUNTIME / "nltk_data"))
    return {"classifier": joblib.load(MODELS / "tfidf.joblib"), "crf": joblib.load(MODELS / "crf.joblib"),
            "topics": joblib.load(MODELS / "topics.joblib"), "sentiment": SentimentIntensityAnalyzer()}


@lru_cache(maxsize=1)
def bert_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    path = MODELS / "bert_classifier"
    return AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True), AutoTokenizer.from_pretrained(path, local_files_only=True)


def analyze(text, use_bert=False):
    if not text.strip():
        raise ValueError("Enter a narrative to analyze.")
    if len(text) > 30000:
        raise ValueError("Please use a narrative of at most 30,000 characters.")
    resources = models()
    tokens = tokens_with_offsets(text)
    labels = resources["crf"].predict_single(token_features(tokens))
    classifier = resources["classifier"]
    probabilities = classifier.predict_proba([text])[0]
    topic_distribution = resources["topics"]["model"].transform(resources["topics"]["vectorizer"].transform([text]))[0]
    doc = language_model()(text)
    result = {"text": text, "entities": spans_from_bio(text, tokens, labels), "summary": summarize(text),
              "crime_type": str(classifier.classes_[probabilities.argmax()]), "model_score": float(probabilities.max()),
              "sentiment": resources["sentiment"].polarity_scores(text), "threat": threat_language(text),
              "topic_id": int(topic_distribution.argmax()), "topic_distribution": topic_distribution.tolist(),
              "tokens": [{"token": t.text, "lemma": t.lemma_, "POS": t.pos_, "tag": t.tag_, "morphology": str(t.morph), "dependency": t.dep_, "head": t.head.text} for t in doc],
              "generic_entities": [{"text": e.text, "label": e.label_} for e in doc.ents]}
    if use_bert:
        from .bert_model import predict_bert
        model, tokenizer = bert_model()
        result["bert_prediction"] = predict_bert(model, tokenizer, [text])[0]
    return result


def highlight_entities(text, entities):
    """Escape all user text and labels before inserting spans into HTML."""
    chunks, cursor = [], 0
    for ent in sorted(entities, key=lambda e: (e["start"], e["end"])):
        start, end = int(ent["start"]), int(ent["end"])
        if start < cursor or end <= start or end > len(text):
            continue
        color = COLORS.get(ent["label"], "#e7eaf0")
        chunks.append(html.escape(text[cursor:start]))
        chunks.append(f'<mark style="background:{color};padding:4px 6px;border-radius:5px;color:#17283d">{html.escape(text[start:end])}<small style="font-size:9px;font-weight:700;margin-left:6px">{html.escape(ent["label"])}</small></mark>')
        cursor = end
    chunks.append(html.escape(text[cursor:]))
    return '<div style="line-height:2.4;font-size:16px;white-space:pre-wrap">' + "".join(chunks) + "</div>"
