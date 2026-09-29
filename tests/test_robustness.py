"""Robust input, source fidelity and country-independent role regressions."""
import json
import random

import numpy as np
import pytest

from crime_nlp.robust_corpus import make_informal_report, NAMES, PLACES, perturb
from crime_nlp.text import tokens_with_offsets, bio_labels, spans_from_bio, word_tokens
from crime_nlp.features import robust_tfidf, keyword_vectorizer, prediction_review


def test_every_informal_template_and_style_preserves_exact_spans():
    groups = {}
    for i in range(864):
        row = make_informal_report(i)
        entities = json.loads(row["entities_json"])
        tokens = tokens_with_offsets(row["narrative"])
        assert spans_from_bio(row["narrative"], tokens, bio_labels(tokens, entities)) == entities
        groups.setdefault(row["template_group"], set()).add(row["split"])
    assert len(groups) == 144
    assert all(len(split) == 1 for split in groups.values())


def test_international_name_and_place_pools_do_not_leak_between_splits():
    for pools in (NAMES, PLACES):
        sets = [set(v) for v in pools.values()]
        assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert "josé" in word_tokens("José García reported a theft.")


def test_typo_edits_preserve_unicode_name_spans():
    text = "Siobhán O'Neill received money from Özlem Demir"
    spans = [{"start": 0, "end": len("Siobhán O'Neill"), "text": "Siobhán O'Neill", "label": "VICTIM"},
             {"start": text.index("Özlem"), "end": len(text), "text": "Özlem Demir", "label": "SUSPECT"}]
    changed, entities = perturb(text, spans, random.Random(1), 2)
    assert entities[0]["text"] == "siobhán o'neill"
    assert entities[1]["text"] == "özlem demir"
    assert changed != text.lower()
    assert all(changed[e["start"]:e["end"]] == e["text"] for e in entities)


def test_character_tfidf_represents_unseen_typo_without_changing_input():
    texts = ["someone stole a mobile phone", "a mobile phone stolen from pocket"] * 3
    vectorizer = robust_tfidf().fit(texts)
    characters = dict(vectorizer.transformer_list)["char"]
    assert characters.transform(["moblie"]).nnz > 0
    assert vectorizer.get_feature_names_out().size == vectorizer.transform(texts).shape[1]


def test_review_flag_for_unknown_or_insufficient_text():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    texts = ["my mobile stolen from pocket", "he punched me on face"] * 3
    model = Pipeline([("vectorizer", robust_tfidf()), ("classifier", LogisticRegression())]).fit(texts, ["Theft", "Assault"] * 3)
    assert keyword_vectorizer(model).transform(["mobile"]).nnz
    for text in ("hello", "qzxv vzzqq zzqxx"):
        result = prediction_review(model, text, np.array([.99, .01]))
        assert result["review_required"]
        assert result["review_reasons"]


@pytest.mark.parametrize("text", ["12345", "... !!!", "hello\x00world", "  "])
def test_invalid_input_rejected_before_loading_models(text):
    from crime_nlp.inference import analyze
    with pytest.raises(ValueError):
        analyze(text)


def test_rebuilt_models_handle_informal_text_and_unknown_names():
    from crime_nlp.config import MODELS
    if not (MODELS / "ner_selection.json").exists():
        pytest.skip("Run the V3 training and enrichment stages first.")
    from crime_nlp.inference import analyze
    for text, category in [("sir my moblie stoln from pocket in bus ystrday", "Theft"),
                           ("my acount hackd pasword changed unable login", "Cybercrime"),
                           ("rajiv hit sneha with a stick near pune", "Assault")]:
        result = analyze(text, ner_choice="CRF")
        assert result["crime_type"] == category
        assert result["text"] == text
        assert all(text[e["start"]:e["end"]] == e["text"] for e in result["entities"])
    found = {(e["label"], e["text"]) for e in result["entities"]}
    assert {("SUSPECT", "rajiv"), ("VICTIM", "sneha"), ("LOCATION", "pune")} <= found
    assert analyze("please help")["review_required"]


def test_benchmark_and_corpus_are_not_mixed_across_versions():
    from crime_nlp.config import MODELS, ARTIFACTS
    if not (MODELS / "ner_selection.json").exists():
        pytest.skip("Run the V3 pipeline first.")
    frozen = json.loads((ARTIFACTS / "frozen_models.json").read_text(encoding="utf-8"))
    from crime_nlp.train import file_hash
    for name, expected in frozen["model_sha256"].items():
        assert file_hash(MODELS / name) == expected
