"""Regressions for the user's actual failures, not training examples."""
import json
from html.parser import HTMLParser

import pytest
from crime_nlp.entity_corpus import make_context_report
from crime_nlp.text import tokens_with_offsets, bio_labels, spans_from_bio, sentences
from crime_nlp.inference import highlight_entities

FIR = ("Information is received today on 27.09.2026 at 22:15 hrs via a written complaint submitted by Shri Rajesh Kumar "
       "Sharma S/o Late Shri Ramesh Kumar Sharma, aged 41 years, R/o House No. 45, Pocket-B, Mayur Vihar "
       "Phase-2, New Delhi (Mobile: +91-9876543210), stating that today at around 20:30 hrs, while he was walking near Gate "
       "No. 3, Rajiv Chowk Metro Station, New Delhi, one unknown male youth aged about 22-25 years riding a black "
       "Splendor motorcycle without a number plate came from behind and forcibly snatched his office bag containing one black "
       "Lenovo LOQ gaming laptop (Serial No. R90XYZ12), important identity cards, and cash amounting to ₹3,500/-, causing "
       "him to fall and sustain minor injuries, after which the accused fled towards the Outer Circle. The statement has been "
       "read over and explained to the complainant, and upon disclosure of cognizable offences under Section 303(2) and 324(4) "
       "BNS, the present FIR is registered electronically at the police station. The original complaint is attached to the case file, and "
       "the investigation of the case is hereby entrusted to Sub-Inspector Amit Kumar.")


def test_all_context_spans_survive_punctuation_and_unicode():
    for split in ("train", "validation", "development", "test"):
        for i in range(160):
            row = make_context_report(i, split)
            tokens = tokens_with_offsets(row["narrative"])
            entities = json.loads(row["entities_json"])
            assert spans_from_bio(row["narrative"], tokens, bio_labels(tokens, entities)) == entities


def test_date_address_and_initials_are_not_sentence_stops():
    text = "Mr. A. K. Rao lives at House No. 45. He reported it on 27.09.2026 at 22:15."
    assert sentences(text) == ["Mr. A. K. Rao lives at House No. 45.", "He reported it on 27.09.2026 at 22:15."]


def test_highlight_and_plain_summary_preserve_every_source_character():
    class SourceParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts, self.label = [], False
        def handle_starttag(self, tag, attrs):
            if tag == "small":
                self.label = True
        def handle_endtag(self, tag):
            if tag == "small":
                self.label = False
        def handle_data(self, data):
            if not self.label:
                self.parts.append(data)
    text = "**Mayank**\tstole ₹3,500/- & a <phone>.\n\n1. Date: 29-06-2026\n2. Name: José O’Neill"
    span = {"start": text.index("29-"), "end": text.index("29-") + 10, "label": "DATE"}
    for entities in ([], [span]):
        parser = SourceParser()
        parser.feed(highlight_entities(text, entities))
        assert "".join(parser.parts) == text


def active_analyzer():
    from crime_nlp.config import MODELS
    if MODELS.name != "v3_1" or not (MODELS / "ner_selection.json").exists():
        pytest.skip("Run the role-context entity revision first")
    from crime_nlp.inference import analyze
    return analyze


@pytest.mark.parametrize("text, expected", [
    ("Mayank stole the phone of Surya while he was sleeping on 29-06-2026",
     {("SUSPECT", "Mayank"), ("VICTIM", "Surya"), ("PROPERTY", "phone"), ("DATE", "29-06-2026")}),
    ("Surya Killed Mayank on 29-06-2026",
     {("SUSPECT", "Surya"), ("VICTIM", "Mayank"), ("DATE", "29-06-2026")}),
    ("mayank was killed by surya on 29.06.2026",
     {("SUSPECT", "surya"), ("VICTIM", "mayank"), ("DATE", "29.06.2026")}),
    ("Chiamaka stole the wallet of Tenzin on 02/03/2026",
     {("SUSPECT", "Chiamaka"), ("VICTIM", "Tenzin"), ("PROPERTY", "wallet"), ("DATE", "02/03/2026")}),
    ("Fatima Al-Zahra hit Jean-Paul O’Connor with a stick near Montréal",
     {("SUSPECT", "Fatima Al-Zahra"), ("VICTIM", "Jean-Paul O’Connor"), ("LOCATION", "Montréal")}),
    ("Jean-Paul O’Connor was beaten by Fatima Al-Zahra near Montréal",
     {("SUSPECT", "Fatima Al-Zahra"), ("VICTIM", "Jean-Paul O’Connor"), ("LOCATION", "Montréal")}),
])
def test_people_and_ownership_in_short_sentences(text, expected):
    result = active_analyzer()(text, ner_choice="CRF")
    assert expected <= {(e["label"], e["text"]) for e in result["entities"]}
    assert result["text"] == text
    if "killed" in text.lower():
        assert result["review_required"]


def test_administrative_fields_are_not_suspects_or_victims():
    result = active_analyzer()(FIR, ner_choice="CRF")
    for ent in result["entities"]:
        assert FIR[ent["start"]:ent["end"]] == ent["text"]
        if ent["label"] in {"SUSPECT", "VICTIM"}:
            assert not any(value in ent["text"] for value in ["House", "Gate", "Serial", "Outer Circle", "Ramesh", "Amit", "Section"])
    found = {(e["label"], e["text"]) for e in result["entities"]}
    assert ("VICTIM", "Rajesh Kumar Sharma") in found
    assert ("DATE", "27.09.2026") in found
    assert ("TIME", "22:15") in found
    assert ("TIME", "20:30") in found
    assert ("LOCATION", "Outer Circle") in found
    assert ("PROPERTY", "identity cards") in found
    assert ("PROPERTY", "office bag") in found
