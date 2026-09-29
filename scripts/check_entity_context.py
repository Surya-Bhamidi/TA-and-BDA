"""Inspect the screenshot regressions and retain actual model predictions."""
import argparse
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from crime_nlp.config import ROOT, MODELS, ARTIFACTS, write_json
from crime_nlp.text import tokens_with_offsets, token_features, spans_from_bio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    directory = ARTIFACTS / "models" / "v3_1" if args.candidate else MODELS
    crf = joblib.load(directory / "crf.joblib")
    fir = runpy.run_path(str(ROOT / "tests/test_entity_context.py"))["FIR"]
    texts = ["Mayank stole the phone of Surya while he was sleeping on 29-06-2026",
             "Surya Killed Mayank on 29-06-2026", "mayank was killed by surya on 29.06.2026",
             "Chiamaka stole the wallet of Tenzin on 02/03/2026", fir]
    results = []
    for text in texts:
        tokens = tokens_with_offsets(text)
        spans = spans_from_bio(text, tokens, crf.predict_single(token_features(tokens)))
        results.append({"text": text, "entities": spans})
        print(json.dumps(results[-1], ensure_ascii=True), flush=True)
    write_json(ARTIFACTS / "entity_context_regressions.json", {"model": str(directory), "scope": "User-provided development regressions and name substitutions; not an independent benchmark", "cases": results})


if __name__ == "__main__":
    main()
