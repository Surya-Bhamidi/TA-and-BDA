"""Offline narrative analysis, model loading and safe entity highlighting."""
import html
import joblib
import json
import numpy as np
from functools import lru_cache

from .config import MODELS, RUNTIME, SETTINGS
from .text import tokens_with_offsets, token_features, spans_from_bio, summarize, threat_language

COLORS = {"SUSPECT": "#ffd9c9", "VICTIM": "#d9e6ff", "LOCATION": "#c4eee5", "WEAPON": "#ffe8ad"}
COLORS.update(PROPERTY="#eee0fb", DATE="#dfeec9", TIME="#dbeaf2", EVIDENCE="#f9d8e9")


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
            "topics": joblib.load(MODELS / "topics.joblib"), "sentiment": SentimentIntensityAnalyzer(),
            "calibration": json.loads((MODELS / "calibration.json").read_text()),
            "ner_selection": json.loads((MODELS / "ner_selection.json").read_text())["selected"]}


@lru_cache(maxsize=1)
def bert_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    path = MODELS / "bert_classifier"
    return AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True), AutoTokenizer.from_pretrained(path, local_files_only=True)


@lru_cache(maxsize=1)
def entity_transformer():
    from .ner_model import load_ner
    return load_ner()


def explain_prediction(classifier, text, top=8):
    vectorizer = classifier.named_steps["vectorizer"]
    estimator = classifier.named_steps["classifier"]
    row = vectorizer.transform([text])
    class_id = estimator.predict_proba(row)[0].argmax()
    weights = row.toarray()[0] * estimator.coef_[class_id]
    terms = vectorizer.get_feature_names_out()
    return [{"term": terms[i], "contribution": float(weights[i])} for i in weights.argsort()[::-1][:top] if weights[i] > 0]


def analyze(text, use_bert=False, ner_choice="Automatic"):
    if not text.strip():
        raise ValueError("Enter a narrative to analyze.")
    if len(text) > SETTINGS["inference"]["max_characters"]:
        raise ValueError("Please use a narrative of at most 30,000 characters.")
    resources = models()
    tokens = tokens_with_offsets(text)
    selected_ner = resources["ner_selection"] if ner_choice == "Automatic" else ner_choice
    if selected_ner not in {"CRF", "BERT"}:
        raise ValueError("Entity model must be Automatic, CRF or BERT")
    if selected_ner == "BERT":
        from .ner_model import extract_bert
        model, tokenizer = entity_transformer()
        entities = extract_bert(text, model, tokenizer)
    else:
        features = token_features(tokens)
        labels = resources["crf"].predict_single(features)
        entities = spans_from_bio(text, tokens, labels)
        marginal = resources["crf"].predict_marginals_single(features)
        for ent in entities:
            probabilities = [marginal[i].get(labels[i], 0) for i, token in enumerate(tokens) if ent["start"] <= token["start"] < ent["end"]]
            ent["confidence"] = float(np.mean(probabilities))
    classifier = resources["classifier"]
    from .evaluation import temperature_scale
    from .topics import topic_input
    from .privacy import annotate_assertions, structured_summary, redact
    probabilities = temperature_scale(classifier.predict_proba([text]), resources["calibration"]["temperature"])[0]
    topic_distribution = resources["topics"]["model"].transform(resources["topics"]["vectorizer"].transform(topic_input([text])))[0]
    doc = language_model()(text)
    entities = annotate_assertions(text, entities)
    result = {"text": text, "entities": entities, "summary": summarize(text), "ner_model": selected_ner,
              "crime_type": str(classifier.classes_[probabilities.argmax()]), "model_score": float(probabilities.max()),
              "sentiment": resources["sentiment"].polarity_scores(text), "threat": threat_language(text),
              "topic_id": int(topic_distribution.argmax()), "topic_distribution": topic_distribution.tolist(),
              "tokens": [{"token": t.text, "lemma": t.lemma_, "POS": t.pos_, "tag": t.tag_, "morphology": str(t.morph), "dependency": t.dep_, "head": t.head.text} for t in doc],
              "generic_entities": [{"text": e.text, "label": e.label_} for e in doc.ents]}
    result["review_required"] = result["model_score"] < resources["calibration"]["threshold"]
    result["explanation"] = explain_prediction(classifier, text)
    result["structured_summary"] = structured_summary(text, entities, result["summary"])
    people = [{"start": e.start_char, "end": e.end_char, "label": "PERSON"} for e in doc.ents if e.label_ == "PERSON"]
    result["redacted_text"] = redact(text, entities + people)
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
