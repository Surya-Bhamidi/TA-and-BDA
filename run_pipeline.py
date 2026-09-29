"""Reproducible command-line entry point. Errors fail loudly and retain stage status."""
import argparse
import importlib.metadata
import platform
import time
import traceback
from datetime import datetime, timezone

from crime_nlp.config import ARTIFACTS, ROOT, SETTINGS, write_json


def main():
    parser = argparse.ArgumentParser(description="Decoding Crime Narratives: Spark + NLP pipeline")
    parser.add_argument("--stage", choices=["all", "generate", "spark", "train", "enrich"], default="all")
    parser.add_argument("--records", type=int, default=60000)
    parser.add_argument("--master", default="local[2]", help="Spark master URL; default two local Spark execution cores")
    parser.add_argument("--train-limit", type=int, default=SETTINGS["training"]["train_limit"])
    parser.add_argument("--test-limit", type=int, default=1200)
    parser.add_argument("--validation-limit", type=int, default=800)
    parser.add_argument("--bert-epochs", type=int, default=SETTINGS["training"]["bert_epochs"])
    parser.add_argument("--resume", action="store_true", help="Skip stages only when recorded input and output hashes still match")
    parser.add_argument("--entities-only", action="store_true", help="Enrich stage only: refresh NER after an entity-only model update; retain unchanged corpus/search/classifier outputs")
    args = parser.parse_args()
    if args.entities_only and args.stage != "enrich":
        parser.error("--entities-only requires --stage enrich")
    if args.records < 50000 and args.stage in {"all", "generate"}:
        parser.error("The full university project requires at least 50,000 valid reports.")
    if min(args.train_limit, args.test_limit, args.validation_limit) < 8 or args.bert_epochs < 1:
        parser.error("Model splits must have at least 8 records and BERT needs at least one epoch.")
    from crime_nlp.generate import generate
    stages = {"generate": lambda: generate(args.records)}
    def spark():
        from crime_nlp.spark_pipeline import run_spark
        run_spark(args.master)
    def train():
        from crime_nlp.train import train_models
        train_models(args.train_limit, args.test_limit, args.validation_limit, args.bert_epochs)
    def enrich():
        from crime_nlp.enrich import enrich as process, refresh_entity_predictions
        if args.entities_only:
            return refresh_entity_predictions()
        process()
    stages.update(spark=spark, train=train, enrich=enrich)
    selected = list(stages) if args.stage == "all" else [args.stage]
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args), "python": platform.python_version(), "platform": platform.platform(), "stages": {}, "status": "running"}
    status_path = ARTIFACTS / f"run_{args.stage}.json"
    started = time.perf_counter()
    from crime_nlp.audit import stage_signature, stage_outputs, record_stage, reusable_stage
    try:
        for name in selected:
            print(f"\n>>> Stage: {name}", flush=True)
            stage_start = time.perf_counter()
            signature = stage_signature(name, vars(args))
            if args.resume and reusable_stage(name, signature):
                status["stages"][name] = {"status": "reused", "reason": "input and output hashes match"}
                print(f"Reused unchanged stage {name}", flush=True)
                continue
            status["stages"][name] = {"status": "running"}
            write_json(status_path, status)
            stages[name]()
            record_stage(name, signature, stage_outputs(name))
            status["stages"][name] = {"status": "complete", "seconds": round(time.perf_counter() - stage_start, 2)}
            write_json(status_path, status)
        status["status"] = "complete"
        packages = ["pyspark", "spacy", "nltk", "gensim", "scikit-learn", "sklearn-crfsuite", "torch", "transformers", "streamlit", "pandas", "numpy"]
        status["versions"] = {p: importlib.metadata.version(p) for p in packages if _installed(p)}
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        traceback.print_exc()
        raise
    finally:
        status["seconds"] = round(time.perf_counter() - started, 2)
        write_json(status_path, status)
        history_path = ARTIFACTS / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + f"-{args.stage}.json")
        write_json(history_path, status)
    print(f"\nCompleted in {status['seconds']:.1f}s. Launch: python -m streamlit run app.py", flush=True)


def _installed(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
