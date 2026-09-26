"""Version 2: choose on validation, freeze model hashes, then evaluate final test."""
import hashlib
import json
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import GroupKFold, cross_val_score
from .config import PROCESSED, MODELS, ARTIFACTS, CATEGORIES, ENTITY_LABELS, SEED, SETTINGS, write_json
from .text import word_tokens, tokens_with_offsets, bio_labels, token_features
from .evaluation import classification_metrics, entity_metrics, fit_temperature, temperature_scale, calibration_metrics


def sample_split(df, name, limit):
    part = df[df["split"] == name].sort_values("report_id")
    if part.empty:
        raise ValueError(f"Missing {name} split. Regenerate the V2 dataset first.")
    per_class = max(1, limit // len(CATEGORIES))
    return pd.concat([g.sample(min(len(g), per_class), random_state=SEED) for _, g in part.groupby("crime_type", observed=True)]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def document_vectors(model, texts):
    vectors = []
    for text in texts:
        words = [w for w in word_tokens(text) if w in model.wv]
        vectors.append(np.mean(model.wv[words], axis=0) if words else np.zeros(model.vector_size))
    return np.asarray(vectors, dtype=np.float32)


def stable_hash(value):
    return int(hashlib.md5(value.encode()).hexdigest()[:8], 16)


def sequence_data(part):
    features, labels = [], []
    for row in part.itertuples():
        tokens = tokens_with_offsets(row.narrative)
        features.append(token_features(tokens))
        labels.append(bio_labels(tokens, json.loads(row.entities_json)))
    return features, labels


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_models(plan):
    hashes = {p.relative_to(MODELS).as_posix(): file_hash(p) for p in sorted(MODELS.rglob("*")) if p.is_file()}
    receipt = {"frozen_at": datetime.now(timezone.utc).isoformat(), "model_sha256": hashes, "plan": plan,
               "policy": "No hyperparameter or feature changes based on final-test outputs in this run."}
    write_json(ARTIFACTS / "frozen_models.json", receipt)
    return receipt


def train_topics(x_train):
    from .topics import topic_input
    stop = sorted(set(ENGLISH_STOP_WORDS) | {"victim", "suspect", "report", "incident", "witness", "officer", "statement", "allege", "record", "identify", "name", "person", "case"})
    vectorizer = CountVectorizer(max_features=1800, min_df=4, max_df=.65, stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    counts = vectorizer.fit_transform(topic_input(x_train))
    lda = LatentDirichletAllocation(n_components=8, max_iter=8, random_state=SEED, max_doc_update_iter=20, n_jobs=1).fit(counts)
    nmf = NMF(n_components=8, init="nndsvda", random_state=SEED, max_iter=300).fit(counts)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    topics = [{"id": i, "terms": vocabulary[w.argsort()[-10:][::-1]].tolist(), "weights": np.sort(w)[-10:][::-1].tolist()} for i, w in enumerate(lda.components_)]
    joblib.dump({"vectorizer": vectorizer, "model": lda}, MODELS / "topics.joblib")
    joblib.dump(nmf, MODELS / "nmf.joblib")
    binary = (counts > 0).astype(np.int64)
    co = (binary.T @ binary).toarray()
    coherence = []
    for component in lda.components_:
        ids = component.argsort()[-10:][::-1]
        coherence.append(float(np.mean([np.log((co[ids[i], ids[j]] + 1.) / max(co[ids[j], ids[j]], 1)) for i in range(1, 10) for j in range(i)])))
    return {"method": "Training-only LDA over spaCy noun/verb/adjective lemmas", "topics": topics, "umass_coherence": coherence,
            "topic_diversity": len(set(t for topic in topics for t in topic["terms"])) / 80,
            "nmf_top_terms": [vocabulary[w.argsort()[-10:][::-1]].tolist() for w in nmf.components_],
            "interpretation": "Co-occurrence themes; names and routine words are reduced. Topics are not verified modus operandi."}


def train_models(train_limit=6400, test_limit=1200, validation_limit=800, bert_epochs=4):
    from gensim.models import Word2Vec
    import sklearn_crfsuite
    from .bert_model import fit_bert, predict_bert
    from .ner_model import train_ner, load_ner, predict_tags, sequences
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    df = pd.read_parquet(PROCESSED / "reports").sort_values("report_id").reset_index(drop=True)
    split_names = ["train", "validation", "development", "test"]
    groups = {name: set(df.loc[df["split"] == name, "template_group"]) for name in split_names}
    assert all(not groups[a] & groups[b] for a in split_names for b in split_names if a != b)
    train, validation, development = (sample_split(df, name, limit) for name, limit in [("train", train_limit), ("validation", validation_limit), ("development", validation_limit)])
    plan = {"seed": SEED, "train_rows": len(train), "validation_rows": len(validation), "development_rows": len(development),
            "test_rows": min(test_limit, len(df[df.split == "test"])), "split": "V2 incident groups, participant phrasings and first-name pools fixed before fitting",
            "groups": {k: sorted(v) for k, v in groups.items()}, "selection": "Validation only; final predictions after model hashes are frozen",
            "limitations": "Synthetic composite templates share grammar and vocabulary; this is not external real-world validation."}
    write_json(ARTIFACTS / "evaluation_plan.json", plan)
    for name, part in [("train", train), ("validation", validation), ("development", development)]:
        part[["report_id", "template_group", "crime_type"]].to_csv(ARTIFACTS / f"{name}_ids.csv", index=False)
    x, y = train.narrative.tolist(), train.crime_type.tolist()
    fitted, timing = {}, {}
    for name, vectorizer in [("Bag of Words", CountVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)),
                            ("TF-IDF", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2, sublinear_tf=True))]:
        started = time.perf_counter()
        model = Pipeline([("vectorizer", vectorizer), ("classifier", LogisticRegression(max_iter=500, random_state=SEED))]).fit(x, y)
        timing[name] = time.perf_counter() - started
        fitted[name] = model
        joblib.dump(model, MODELS / ("bow.joblib" if name == "Bag of Words" else "tfidf.joblib"))
        print(f"{name} validation macro-F1: {classification_metrics(validation.crime_type, model.predict(validation.narrative))['macro_f1']:.4f}", flush=True)
    cv_groups = train.template_group.str.replace(r"-layout\d+$", "", regex=True)
    cv = cross_val_score(fitted["TF-IDF"], x, y, groups=cv_groups, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1)
    temperature = fit_temperature(fitted["TF-IDF"].predict_proba(validation.narrative), validation.crime_type, fitted["TF-IDF"].classes_)
    write_json(MODELS / "calibration.json", {"temperature": temperature, "threshold": SETTINGS["inference"]["confidence_threshold"], "fit_split": "validation"})
    started = time.perf_counter()
    w2v = Word2Vec([word_tokens(t) for t in x], vector_size=100, window=5, min_count=2, workers=1, sg=1, seed=SEED, epochs=12, hashfxn=stable_hash)
    w2v.save(str(MODELS / "word2vec.model"))
    word_classifier = LogisticRegression(max_iter=500, random_state=SEED).fit(document_vectors(w2v, x), y)
    joblib.dump(word_classifier, MODELS / "word2vec_classifier.joblib")
    timing["Word2Vec"] = time.perf_counter() - started
    _, bert_details = fit_bert(x, y, validation.narrative.tolist(), validation.crime_type.tolist(), development.narrative.tolist(), epochs=bert_epochs)
    print("Training CRF and BERT entity taggers, selecting on validation...", flush=True)
    ner_train = sample_split(df, "train", SETTINGS["training"]["ner_train_limit"])
    ner_val = sample_split(df, "validation", min(400, validation_limit))
    train_x, train_y = sequence_data(ner_train)
    val_x, val_y = sequence_data(ner_val)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.1, c2=.1, max_iterations=100, all_possible_transitions=True).fit(train_x, train_y)
    joblib.dump(crf, MODELS / "crf.joblib")
    crf_validation = entity_metrics(val_y, crf.predict(val_x))
    ner_details = train_ner(ner_train, ner_val, epochs=SETTINGS["training"]["ner_epochs"])
    selected_ner = "BERT" if ner_details["validation_f1"] > crf_validation["strict_entity_f1"] else "CRF"
    write_json(MODELS / "ner_selection.json", {"selected": selected_ner, "crf_validation_f1": crf_validation["strict_entity_f1"], "bert_validation_f1": ner_details["validation_f1"], "criterion": "strict validation span F1; ties choose faster CRF"})
    topic_metrics = train_topics(x)
    receipt = freeze_models(plan)
    print("Models frozen. Opening final synthetic test split...", flush=True)
    test = sample_split(df, "test", test_limit)
    plan["test_rows"] = len(test)
    test[["report_id", "template_group", "crime_type"]].to_csv(ARTIFACTS / "test_ids.csv", index=False)
    bert = AutoModelForSequenceClassification.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
    guesses = {name: model.predict(test.narrative) for name, model in fitted.items()}
    guesses["Word2Vec"] = word_classifier.predict(document_vectors(w2v, test.narrative))
    guesses["BERT"] = predict_bert(bert, tokenizer, test.narrative.tolist())
    results = {"protocol": plan, "models": {}, "grouped_cv": {"model": "TF-IDF", "fold_macro_f1": cv.tolist(), "mean": float(cv.mean()), "groups": "incident variants within training only"}}
    for name, predicted in guesses.items():
        results["models"][name] = classification_metrics(test.crime_type, predicted)
        results["models"][name]["train_seconds"] = timing.get(name, bert_details["train_seconds"])
    results["models"]["BERT"].update(bert_details)
    for split, part in [("development", development), ("test", test)]:
        probabilities = fitted["TF-IDF"].predict_proba(part.narrative)
        results[f"calibration_{split}"] = {"before": calibration_metrics(part.crime_type, probabilities, fitted["TF-IDF"].classes_),
            "after": calibration_metrics(part.crime_type, temperature_scale(probabilities, temperature), fitted["TF-IDF"].classes_), "temperature": temperature}
    errors = test[["report_id", "crime_type", "template_group"]].copy()
    errors["prediction"], errors["characters"] = guesses["TF-IDF"], test.narrative.str.len()
    errors[errors.crime_type != errors.prediction].to_csv(ARTIFACTS / "classification_errors.csv", index=False)
    write_json(ARTIFACTS / "evaluation.json", results)
    ner_test = sample_split(df, "test", min(600, test_limit))
    final_x, final_y = sequence_data(ner_test)
    crf_metrics = entity_metrics(final_y, crf.predict(final_x))
    ner_bert, ner_tokenizer = load_ner()
    final_tokens, gold = sequences(ner_test)
    ner_predictions, _ = predict_tags(ner_bert, ner_tokenizer, final_tokens)
    bert_metrics = entity_metrics(gold, ner_predictions)
    selected = bert_metrics if selected_ner == "BERT" else crf_metrics
    write_json(ARTIFACTS / "ner_evaluation.json", {**selected, "model": selected_ner, "labels": ENTITY_LABELS, "train_rows": len(ner_train), "validation_rows": len(ner_val), "test_rows": len(ner_test),
               "models": {"CRF": crf_metrics, "BERT": bert_metrics}, "validation": {"CRF": crf_validation["strict_entity_f1"], "BERT": ner_details["validation_f1"]},
               "bert_history": ner_details["history"], "split": "V2 final held-out incident and participant patterns; models fixed before scoring"})
    from .topics import topic_input
    bundle = joblib.load(MODELS / "topics.joblib")
    topic_metrics["held_out_perplexity"] = float(bundle["model"].perplexity(bundle["vectorizer"].transform(topic_input(test.narrative.tolist()))))
    write_json(ARTIFACTS / "topics.json", topic_metrics)
    write_json(ARTIFACTS / "final_evaluation_receipt.json", {"evaluated_at": datetime.now(timezone.utc).isoformat(), "frozen_at": receipt["frozen_at"], "selected_ner": selected_ner,
               "note": "Do not retune this release from final-test outputs; create a new version and independent holdout for future development."})
    print(f"Final selected NER ({selected_ner}) strict entity F1={selected['strict_entity_f1']:.4f}", flush=True)
    return results
