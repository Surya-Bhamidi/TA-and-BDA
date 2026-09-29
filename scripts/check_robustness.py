"""Report a fixed, hand-written development challenge against V2 and V3.

This set is a regression/development suite, not an untouched final benchmark.
Run after rebuilding: python scripts/check_robustness.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ARTIFACTS, MODELS, write_json
from crime_nlp.text import tokens_with_offsets, token_features, spans_from_bio

CASES = [
    ("Theft", "sir my moblie stoln from pocket in bus ystrday"),
    ("Theft", "someone took my bag when i was buying ticket"),
    ("Theft", "my bicycle gone from outside cafe no one saw"),
    ("Theft", "someone quietly removed my wallet in the crowd"),
    ("Robbery", "two fellows showed knife and took my phone near bus stand"),
    ("Robbery", "he said give cash or i stab you so i gave cash"),
    ("Robbery", "they pushed me down snatched gold chain and ran"),
    ("Robbery", "gun point they took all money from me"),
    ("Burglary", "went out came home door lock brokn laptop missing inside"),
    ("Burglary", "someone entered our locked shop by breaking shutter and stole cash"),
    ("Burglary", "thieves climbed through window into flat and took jewellery"),
    ("Burglary", "house break in happened last nite things taken from cupboard"),
    ("Assault", "he beating me with rod i got hurt on head"),
    ("Assault", "José García hit Wei Zhang with iron rod near Nairobi"),
    ("Assault", "fatima was punched by rajesh outside office"),
    ("Assault", "one fellow slaped me then kickd me nothing stolen"),
    ("Fraud", "paid advance for phone seller disapeared no delivery and blocked my number"),
    ("Fraud", "sir job promise they collected 5000 rupees now fake company"),
    ("Fraud", "someone pretending bank staff made me transfer money"),
    ("Fraud", "fake investment double money promise all savings gone"),
    ("Cybercrime", "my acount hackd pasword changed unable login"),
    ("Cybercrime", "clicked email link now all files locked asking bitcoin"),
    ("Cybercrime", "someone using my email after stealing password"),
    ("Cybercrime", "fake login site took otp and accessed my account"),
    ("Vandalism", "someone scrached car and broke mirrors purposely"),
    ("Vandalism", "they painted bad words on wall and smashed public bench"),
    ("Vandalism", "parked bike tyres punctured by person intentionally"),
    ("Vandalism", "bus stop glass broken deliberately nothing stolen"),
    ("Arson", "he put petrol on my shop and lit match full shop burning"),
    ("Arson", "someone purposely set our scooter on fire"),
    ("Arson", "poured kerosene then burned the shed intentionally"),
    ("Arson", "our warehouse torched by someone last night"),
]


def evaluate(classifier, crf):
    predictions = classifier.predict([text for _, text in CASES])
    rows = [{"text": text, "expected": expected, "prediction": str(guess), "correct": expected == guess}
            for (expected, text), guess in zip(CASES, predictions)]
    roles = []
    for suspect, victim, place in [("José García", "Wei Zhang", "Nairobi"),
                                   ("Kavya Subramaniam", "Oluwaseun Adebisi", "Timbuktu"),
                                   ("Fatima Al-Zahra", "Jean-Paul O’Connor", "Montréal"),
                                   ("rajiv", "sneha", "pune")]:
        for pattern in ("{suspect} hit {victim} with a stick near {place}",
                        "{victim} was beaten by {suspect} near {place}"):
            text = pattern.format(suspect=suspect, victim=victim, place=place)
            tokens = tokens_with_offsets(text)
            spans = spans_from_bio(text, tokens, crf.predict_single(token_features(tokens)))
            found = {(e["label"], e["text"]) for e in spans}
            expected = {("SUSPECT", suspect), ("VICTIM", victim), ("LOCATION", place)}
            roles.append({"text": text, "exact_roles_and_location": expected <= found, "entities": spans})
    return {"category_model": "TF-IDF", "entity_model": "CRF", "classification_accuracy": sum(r["correct"] for r in rows) / len(rows), "cases": rows,
            "exact_role_cases": sum(r["exact_roles_and_location"] for r in roles), "role_cases": roles}


def main():
    import joblib
    results = {"scope": "Fixed hand-written development/regression examples; not an independent real-world benchmark."}
    for version, directory in [("v2", ARTIFACTS / "models" / "v2"), ("v3", ARTIFACTS / "models" / "v3"), ("v3.1", MODELS)]:
        if (directory / "tfidf.joblib").exists():
            results[version] = evaluate(joblib.load(directory / "tfidf.joblib"), joblib.load(directory / "crf.joblib"))
            print(version, results[version]["classification_accuracy"], "exact role cases", results[version]["exact_role_cases"], flush=True)
    write_json(ARTIFACTS / "robustness_development.json", results)
    print(json.dumps(results.get("v3.1", {}), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
