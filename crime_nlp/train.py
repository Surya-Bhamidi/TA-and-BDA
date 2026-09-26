"""Comparable representation models, topic modeling, and CRF NER evaluation."""
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation

from .config import PROCESSED, MODELS, ARTIFACTS, CATEGORIES, SEED, write_json
from .text import word_tokens, tokens_with_offsets, bio_labels, token_features


def sample_split(df, name, limit):
    part = df[df["split"] == name].sort_values("report_id")
    per_class = max(1, limit // len(CATEGORIES))
    return pd.concat([group.sample(min(len(group), per_class), random_state=SEED) for _, group in part.groupby("crime_type", observed=True)]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def evaluate(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, labels=CATEGORIES, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(y, pred, labels=CATEGORIES, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=CATEGORIES).tolist(), "labels": CATEGORIES}


def document_vectors(model, texts):
    vectors = []
    for text in texts:
        words = [w for w in word_tokens(text) if w in model.wv]
        vectors.append(np.mean(model.wv[words], axis=0) if words else np.zeros(model.vector_size))
    return np.asarray(vectors, dtype=np.float32)


def stable_hash(value):
    return int(hashlib.md5(value.encode()).hexdigest()[:8], 16)


def train_models(train_limit=4800, test_limit=1200, validation_limit=800, bert_epochs=3):
    from gensim.models import Word2Vec
    import sklearn_crfsuite
    from seqeval.metrics import classification_report as sequence_report, f1_score as sequence_f1
    from seqeval.scheme import IOB2
    from .bert_model import fit_bert
    df = pd.read_parquet(PROCESSED / "reports").sort_values("report_id").reset_index(drop=True)
    splits = {name: set(df.loc[df["split"] == name, "template_group"]) for name in ["train", "validation", "test"]}
    assert not (splits["train"] & splits["test"] or splits["train"] & splits["validation"] or splits["validation"] & splits["test"])
    train, test, validation = sample_split(df, "train", train_limit), sample_split(df, "test", test_limit), sample_split(df, "validation", validation_limit)
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        part[["report_id", "template_group", "crime_type"]].to_csv(ARTIFACTS / f"{name}_ids.csv", index=False)
    x_train, x_test, y_train, y_test = train.narrative.tolist(), test.narrative.tolist(), train.crime_type.tolist(), test.crime_type.tolist()
    results = {"protocol": {"seed": SEED, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "split": "Held-out scenario templates, fixed before model fitting", "groups": {k: sorted(v) for k, v in splits.items()}, "limitations": "Synthetic template benchmark. Shared vocabulary and generator style remain; no real-world generalization claim."}, "models": {}}
    for name, vectorizer in [
        ("Bag of Words", CountVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)),
        ("TF-IDF", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2, sublinear_tf=True))]:
        started = time.perf_counter()
        model = Pipeline([("vectorizer", vectorizer), ("classifier", LogisticRegression(max_iter=400, random_state=SEED))])
        model.fit(x_train, y_train)
        result = evaluate(y_test, model.predict(x_test))
        result.update({"seconds": round(time.perf_counter() - started, 2), "dimensions": len(model.named_steps["vectorizer"].vocabulary_)})
        results["models"][name] = result
        joblib.dump(model, MODELS / ("bow.joblib" if name == "Bag of Words" else "tfidf.joblib"))
        print(f"{name}: macro-F1 {result['macro_f1']:.4f}", flush=True)
    started = time.perf_counter()
    w2v = Word2Vec([word_tokens(t) for t in x_train], vector_size=100, window=5, min_count=2, workers=1, sg=1, seed=SEED, epochs=12, hashfxn=stable_hash)
    w2v.save(str(MODELS / "word2vec.model"))
    w2v_classifier = LogisticRegression(max_iter=500, random_state=SEED)
    w2v_classifier.fit(document_vectors(w2v, x_train), y_train)
    result = evaluate(y_test, w2v_classifier.predict(document_vectors(w2v, x_test)))
    result.update({"seconds": round(time.perf_counter() - started, 2), "dimensions": 100, "architecture": "Gensim neural skip-gram with negative sampling; mean document pooling"})
    results["models"]["Word2Vec"] = result
    joblib.dump(w2v_classifier, MODELS / "word2vec_classifier.joblib")
    print(f"Word2Vec: macro-F1 {result['macro_f1']:.4f}", flush=True)
    prediction, details = fit_bert(x_train, y_train, validation.narrative.tolist(), validation.crime_type.tolist(), x_test, epochs=bert_epochs)
    results["models"]["BERT"] = {**evaluate(y_test, prediction), **details, "dimensions": 128}
    write_json(ARTIFACTS / "evaluation.json", results)

    # Supervised linear-chain CRF with exact BIO spans; train on training split only.
    def sequence_data(part):
        features, labels = [], []
        for row in part.itertuples():
            tokens = tokens_with_offsets(row.narrative)
            features.append(token_features(tokens))
            labels.append(bio_labels(tokens, json.loads(row.entities_json)))
        return features, labels
    print("Training CRF entity sequence tagger...", flush=True)
    ner_train = sample_split(df, "train", min(2400, train_limit))
    ner_test = sample_split(df, "test", min(600, test_limit))
    ner_x, ner_y = sequence_data(ner_train)
    ner_test_x, ner_test_y = sequence_data(ner_test)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.1, c2=.1, max_iterations=80, all_possible_transitions=True)
    crf.fit(ner_x, ner_y)
    ner_prediction = crf.predict(ner_test_x)
    ner_metrics = {"model": "Linear-chain CRF (BIO sequence labeling)", "train_rows": len(ner_train), "test_rows": len(ner_test),
                   "strict_entity_f1": float(sequence_f1(ner_test_y, ner_prediction, mode="strict", scheme=IOB2, zero_division=0)),
                   "report": sequence_report(ner_test_y, ner_prediction, mode="strict", scheme=IOB2, output_dict=True, zero_division=0),
                   "labels": ["SUSPECT", "VICTIM", "LOCATION", "WEAPON"], "offset_source": "generator ground truth on unchanged narrative",
                   "split": "held-out templates; no rule-based corrections included in score",
                   "development_disclosure": "Sentence-context observations were added after initial test error analysis. This is a development benchmark, not an untouched external evaluation."}
    joblib.dump(crf, MODELS / "crf.joblib")
    write_json(ARTIFACTS / "ner_evaluation.json", ner_metrics)
    print(f"CRF strict entity F1: {ner_metrics['strict_entity_f1']:.4f}", flush=True)

    print("Learning recurring themes with LDA...", flush=True)
    stop = sorted(set(ENGLISH_STOP_WORDS) | {"victim", "suspect", "reported", "incident", "witness", "witnesses", "report", "person", "named", "officers", "recorded", "statement", "alleged", "case", "evidence"})
    topic_vectorizer = CountVectorizer(max_features=1800, min_df=5, max_df=.45, stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    topic_counts = topic_vectorizer.fit_transform(x_train)
    lda = LatentDirichletAllocation(n_components=8, max_iter=8, learning_method="batch", random_state=SEED, max_doc_update_iter=20, n_jobs=1)
    lda.fit(topic_counts)
    vocabulary = np.asarray(topic_vectorizer.get_feature_names_out())
    topics = [{"id": i, "terms": vocabulary[weights.argsort()[-10:][::-1]].tolist(), "weights": np.sort(weights)[-10:][::-1].tolist()} for i, weights in enumerate(lda.components_)]
    test_counts = topic_vectorizer.transform(x_test)
    topic_metrics = {"method": "Unsupervised LDA, fit on training narratives only", "topics": topics,
                     "held_out_perplexity": float(lda.perplexity(test_counts)), "topic_diversity": len(set(t for x in topics for t in x["terms"])) / 80,
                     "interpretation": "Topics are word co-occurrence patterns; they are not verified modus operandi or crime labels."}
    joblib.dump({"vectorizer": topic_vectorizer, "model": lda}, MODELS / "topics.joblib")
    write_json(ARTIFACTS / "topics.json", topic_metrics)
    print("Model artifacts and measured evaluation metrics saved.", flush=True)
    return results
