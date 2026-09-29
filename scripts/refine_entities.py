"""Train the existing CRF/BERT entity models on broader role contexts.

Document/topic models and the 60,000-row corpus are retained. The candidate is
written separately; changing config.MODELS activates it after verification.
"""
import json
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import sklearn_crfsuite
from crime_nlp.config import ARTIFACTS, PROCESSED, MODELS, SETTINGS, ENTITY_LABELS, write_json
from crime_nlp.train import entity_split, sequence_data
from crime_nlp.evaluation import entity_metrics
from crime_nlp import ner_model, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf-only", action="store_true", help="Retain trained candidate BERT and remeasure it on the current validation cases")
    parser.add_argument("--update-active", action="store_true", help="Atomically replace the active CRF after fitting; requires --crf-only")
    args = parser.parse_args()
    target = ARTIFACTS / "models" / "v3_1"
    if args.update_active and not args.crf_only:
        raise RuntimeError("--update-active requires --crf-only")
    if target == MODELS and not args.update_active:
        raise RuntimeError("Candidate already active. Use run_pipeline.py --stage train for a complete rebuild.")
    target.mkdir(parents=True, exist_ok=True)
    if not args.crf_only:
        shutil.copytree(MODELS, target, dirs_exist_ok=True)
    archive = ARTIFACTS / "baselines" / "v3"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["ner_evaluation.json", "evaluation_plan.json", "frozen_models.json", "final_evaluation_receipt.json", "evaluation.json"]:
        if not (archive / name).exists():
            shutil.copy2(ARTIFACTS / name, archive / name)
    df = pd.read_parquet(PROCESSED / "reports")
    fitting = entity_split(df, "train", SETTINGS["training"]["ner_train_limit"])
    validation = entity_split(df, "validation", 400)
    for split, frame in [("train", fitting), ("validation", validation)]:
        frame[["report_id", "narrative", "entities_json", "split", "template_group"]].to_json(
            PROCESSED / f"entity_{split}.jsonl", orient="records", lines=True, force_ascii=False)
    x, y = sequence_data(fitting)
    vx, vy = sequence_data(validation)
    print(f"Training CRF on {len(fitting)} role/context reports...", flush=True)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.1, c2=.1, max_iterations=100, all_possible_transitions=True).fit(x, y)
    crf_val = entity_metrics(vy, crf.predict(vx))["strict_entity_f1"]
    if crf_val < .85:
        raise RuntimeError("Candidate validation score is too low; the active model was retained.")
    temporary = target / "crf.pending.joblib"
    joblib.dump(crf, temporary)
    temporary.replace(target / "crf.joblib")
    print(f"CRF validation F1: {crf_val:.4f}", flush=True)
    del x, y, vx, vy
    ner_model.MODELS = target
    if args.crf_only:
        bert, tokenizer = ner_model.load_ner()
        vt, vg = ner_model.sequences(validation)
        details = json.loads((target / "bert_ner" / "training.json").read_text(encoding="utf-8"))
        details["validation_f1"] = entity_metrics(vg, ner_model.predict_tags(bert, tokenizer, vt)[0])["strict_entity_f1"]
    else:
        details = ner_model.train_ner(fitting, validation, epochs=SETTINGS["training"]["ner_epochs"])
    selected = "BERT" if details["validation_f1"] > crf_val else "CRF"
    write_json(target / "ner_selection.json", {"selected": selected, "crf_validation_f1": crf_val,
               "bert_validation_f1": details["validation_f1"], "criterion": "strict validation span F1; ties choose CRF", "revision": "3.1"})
    plan = json.loads((ARTIFACTS / "evaluation_plan.json").read_text(encoding="utf-8"))
    plan["entity_revision"] = {"version": "3.1", "train_rows": len(fitting), "validation_rows": len(validation),
        "recipe": "50% existing corpus, 50% generated role/ownership/administrative contexts; CRF additionally refined for destination phrasing; unchanged eight labels and model families",
        "evaluation_note": "Retained V3 test subset is previously viewed; new synthetic context cases share grammar. Screenshot cases are development regressions, not a final benchmark."}
    train.MODELS = target
    receipt = train.freeze_models(plan)
    final = entity_split(df, "test", 600)
    tx, ty = sequence_data(final)
    crf_result = entity_metrics(ty, crf.predict(tx))
    bert, tokenizer = ner_model.load_ner()
    tokens, labels = ner_model.sequences(final)
    bert_result = entity_metrics(labels, ner_model.predict_tags(bert, tokenizer, tokens)[0])
    metrics = bert_result if selected == "BERT" else crf_result
    write_json(ARTIFACTS / "ner_evaluation.json", {**metrics, "model": selected, "labels": ENTITY_LABELS,
        "train_rows": len(fitting), "validation_rows": len(validation), "test_rows": len(final),
        "models": {"CRF": crf_result, "BERT": bert_result}, "validation": {"CRF": crf_val, "BERT": details["validation_f1"]},
        "bert_history": details["history"], "split": plan["entity_revision"]["evaluation_note"]})
    write_json(ARTIFACTS / "evaluation_plan.json", plan)
    write_json(ARTIFACTS / "final_evaluation_receipt.json", {"evaluated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_at": receipt["frozen_at"], "selected_ner": selected, "note": plan["entity_revision"]["evaluation_note"]})
    print(f"Candidate ready: {target}. Selected {selected}, entity F1 {metrics['strict_entity_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
