import json
import random

import pytest

from crime_nlp.generate import make_report
from crime_nlp.text import bio_labels, tokens_with_offsets, spans_from_bio, summarize, sentences, threat_language, evidence_answer
from crime_nlp.inference import highlight_entities


def test_generated_offsets_and_bio_roundtrip_cover_all_templates():
    rng = random.Random(42)
    groups = {}
    for i in range(512):
        row = make_report(i, rng)
        entities = json.loads(row["entities_json"])
        tokens = tokens_with_offsets(row["narrative"])
        restored = spans_from_bio(row["narrative"], tokens, bio_labels(tokens, entities))
        assert restored == entities
        for ent in entities:
            assert row["narrative"][ent["start"]:ent["end"]] == ent["text"]
        groups.setdefault(row["template_group"], set()).add(row["split"])
    assert len(groups) == 384
    assert all(len(splits) == 1 for splits in groups.values())


@pytest.mark.parametrize("text", ["No weapon was found.", "There were no threats reported.", "The witness denied a knife was present.", "Nobody was injured; no weapon was reported."])
def test_negated_threat_cues(text):
    assert threat_language(text)["score"] == 0


def test_threat_clause_boundaries():
    result = threat_language("No weapon was found, but the assailant threatened a witness with a handgun.")
    assert "weapon" not in result["cues"]
    assert result["level"] == "High"
    assert "handgun" in result["cues"]


def test_summary_is_extractive_and_shorter():
    text = "A laptop was taken from an office. A witness saw someone leave the office. Police interviewed the witness. The office had a broken window."
    summary = summarize(text, 2)
    assert len(sentences(summary)) == 2
    assert all(s in sentences(text) for s in sentences(summary))
    assert len(summary) < len(text)


def test_evidence_search_abstains():
    assert evidence_answer("satellite orbit", "The witness reported a stolen laptop.")["answer"] == "No matching evidence found."
    assert "laptop" in evidence_answer("laptop", "The witness reported a stolen laptop.")["answer"]


def test_html_highlighting_escapes_untrusted_text():
    text = '<script>alert("test")</script> Mira Vale'
    rendered = highlight_entities(text, [{"start": 30, "end": len(text), "label": '<img src=x onerror=alert(1)>'}])
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;" in rendered


def test_illegal_iob_continuation_starts_new_span():
    text = "Ari Vale"
    entities = spans_from_bio(text, tokens_with_offsets(text), ["I-SUSPECT", "I-SUSPECT"])
    assert entities == [{"start": 0, "end": 8, "label": "SUSPECT", "text": text}]
