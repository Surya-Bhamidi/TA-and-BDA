"""Record a transparent CRF context ablation and validation-selected revision.

The original test outcome motivated error analysis. Keep it and explicitly mark
the revised test as development evaluation, not a pristine external benchmark.
"""
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import sklearn_crfsuite
from seqeval.metrics import f1_score, classification_report
from seqeval.scheme import IOB2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import PROCESSED, ARTIFACTS, MODELS, write_json
from crime_nlp.text import tokens_with_offsets, token_features, bio_labels
from crime_nlp.train import sample_split


def prepare(part):
    x, y = [], []
    for row in part.itertuples():
        tokens = tokens_with_offsets(row.narrative)
        x.append(token_features(tokens))
        y.append(bio_labels(tokens, json.loads(row.entities_json)))
    return x, y


def main():
    ablation_path = ARTIFACTS / "ner_context_ablation.json"
    if ablation_path.exists():
        print("Ablation already recorded; retaining the original baseline and selected model.")
        print(ablation_path.read_text(encoding="utf-8"))
        return
    df = pd.read_parquet(PROCESSED / "reports")
    train = sample_split(df, "train", 2400)
    validation = sample_split(df, "validation", 600)
    test = sample_split(df, "test", 600)
    x_train, y_train = prepare(train)
    x_val, y_val = prepare(validation)
    x_test, y_test = prepare(test)
    baseline = joblib.load(MODELS / "crf.joblib")
    baseline_val = f1_score(y_val, baseline.predict(x_val), mode="strict", scheme=IOB2, zero_division=0)
    baseline_metrics = json.loads((ARTIFACTS / "ner_evaluation.json").read_text())
    write_json(ARTIFACTS / "ner_initial_baseline.json", baseline_metrics)
    candidate = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.1, c2=.1, max_iterations=80, all_possible_transitions=True)
    candidate.fit(x_train, y_train)
    candidate_val = f1_score(y_val, candidate.predict(x_val), mode="strict", scheme=IOB2, zero_division=0)
    accepted = bool(candidate_val >= baseline_val)
    if accepted:
        joblib.dump(candidate, MODELS / "crf.joblib")
    chosen = candidate if accepted else baseline
    predicted = chosen.predict(x_test)
    result = {"model": "Linear-chain CRF with sentence-context observations (BIO)", "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
              "strict_entity_f1": float(f1_score(y_test, predicted, mode="strict", scheme=IOB2, zero_division=0)),
              "report": classification_report(y_test, predicted, mode="strict", scheme=IOB2, output_dict=True, zero_division=0),
              "labels": ["SUSPECT", "VICTIM", "LOCATION", "WEAPON"], "offset_source": "generator offsets on original narrative",
              "split": "held-out templates; revised after initial error analysis; development evaluation",
              "ablation": {"initial_validation_f1": float(baseline_val), "context_validation_f1": float(candidate_val), "selected_by_validation": accepted, "initial_test_f1": baseline_metrics["strict_entity_f1"],
                           "disclosure": "Initial test role errors informed feature development; the revised test score is not an untouched final benchmark."}}
    write_json(ARTIFACTS / "ner_evaluation.json", result)
    write_json(ARTIFACTS / "ner_context_ablation.json", result["ablation"])
    print(json.dumps({"validation_before": float(baseline_val), "validation_after": float(candidate_val), "revised_test_f1": result["strict_entity_f1"], "accepted": accepted}))


if __name__ == "__main__":
    main()
