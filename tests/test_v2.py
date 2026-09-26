"""Regression tests for confidence, role handling, imports and private feedback."""
import json
import sqlite3
import numpy as np
import pytest

from crime_nlp.evaluation import temperature_scale, repair_bio, calibration_metrics
from crime_nlp.privacy import annotate_assertions, redact, structured_summary
from crime_nlp.import_data import validate_record, stage_jsonl
from crime_nlp.semantic import rank_scores


def test_calibration_preserves_category_and_normalizes():
    probabilities = np.array([[.8, .15, .05], [.1, .2, .7]])
    scaled = temperature_scale(probabilities, 2)
    assert np.allclose(scaled.sum(axis=1), 1)
    assert np.array_equal(scaled.argmax(1), probabilities.argmax(1))
    assert np.all(scaled.max(axis=1) < probabilities.max(axis=1))


def test_calibration_can_abstain_on_every_report():
    result = calibration_metrics(["a", "b"], [[.5, .5], [.5, .5]], ["a", "b"])
    assert result["coverage"] == 0
    assert result["accuracy_when_accepted"] is None
    assert sum(b["count"] for b in result["bins"]) == 2


def test_bio_repairs_orphan_and_changed_entity_type():
    assert repair_bio(["I-VICTIM", "I-VICTIM", "I-SUSPECT", "O", "I-DATE"]) == ["B-VICTIM", "I-VICTIM", "B-SUSPECT", "O", "B-DATE"]


def test_negation_excluded_from_structured_weapon_notes():
    text = "No knife was reported. Alex Lee called."
    entities = annotate_assertions(text, [{"text": "knife", "start": 3, "end": 8, "label": "WEAPON"}])
    assert entities[0]["assertion"] == "negated"
    assert structured_summary(text, entities, text)["mentioned_facts"]["WEAPON"] == []


def test_redaction_preserves_surrounding_text_and_masks_contacts():
    text = "Alex Lee called alex@example.org at +91 9876543210."
    redacted = redact(text, [{"text": "Alex Lee", "start": 0, "end": 8, "label": "VICTIM"}])
    assert redacted == "[VICTIM] called [EMAIL] at [PHONE]."


def test_hybrid_search_retains_semantic_only_candidate():
    result = rank_scores(np.array([.8, 0., 0.]), np.array([.1, .9, .1]))
    assert result[1] > result[0] > result[2] == 0


def sample_record():
    return {"report_id": "A-1", "narrative": "Alex reported a theft.", "reported_at": "2026-01-01T12:00:00", "district": "Fictional"}


def test_import_rejects_invalid_span_and_date():
    row = sample_record()
    row["entities"] = [{"start": 0, "end": 4, "text": "Wrong", "label": "VICTIM"}]
    with pytest.raises(ValueError):
        validate_record(row)
    row.pop("entities")
    row["reported_at"] = "not a date"
    with pytest.raises(ValueError):
        validate_record(row)


def test_import_quarantines_errors_and_duplicates_without_raw_error_text(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(sample_record()) + "\n" + json.dumps(sample_record()) + "\nnot JSON", encoding="utf-8")
    target, receipt = stage_jsonl(source, "https://example.org/synthetic", "test fixture", tmp_path / "staged")
    assert receipt["accepted"] == 1 and receipt["rejected"] == 2
    assert "Alex" not in (target / "rejections.jsonl").read_text()


def test_feedback_does_not_persist_submitted_narrative(tmp_path, monkeypatch):
    from crime_nlp import store
    monkeypatch.setattr(store, "PROCESSED", tmp_path)
    store.save_feedback("A private narrative", "Theft", "Fraud")
    with sqlite3.connect(tmp_path / "feedback.sqlite") as connection:
        row = connection.execute("SELECT * FROM feedback").fetchone()
    assert "A private narrative" not in str(row)
    assert row[3] == "Fraud"


def test_subword_heavy_windows_preserve_words_and_offsets():
    from crime_nlp.ner_model import bounded_windows
    def tokenizer(words, **kwargs):
        return {"input_ids": [0] * (len(words) * 30 + 2)}
    words = ["word" + str(i) for i in range(48)]
    windows = bounded_windows(words, tokenizer, 512, 7)
    assert [word for _, chunk in windows for word in chunk] == words
    assert all(len(tokenizer(chunk)["input_ids"]) <= 512 for _, chunk in windows)
    assert windows[0][0] == 7
    assert windows[-1][0] + len(windows[-1][1]) == 55


def test_resume_rejects_changed_inputs_or_outputs(tmp_path, monkeypatch):
    from crime_nlp import audit
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "ARTIFACTS", tmp_path / "artifacts")
    output = tmp_path / "example.txt"
    output.write_text("original")
    audit.record_stage("example", "input-v1", [output])
    assert audit.reusable_stage("example", "input-v1")
    assert not audit.reusable_stage("example", "input-v2")
    output.write_text("modified")
    assert not audit.reusable_stage("example", "input-v1")
