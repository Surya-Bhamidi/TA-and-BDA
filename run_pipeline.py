"""Reproducible command-line entry point. Errors fail loudly and retain stage status."""
import argparse
import importlib.metadata
import platform
import time
import traceback
from datetime import datetime, timezone

from crime_nlp.config import ARTIFACTS, ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description="Decoding Crime Narratives: Spark + NLP pipeline")
    parser.add_argument("--stage", choices=["all", "generate", "spark", "train", "enrich"], default="all")
    parser.add_argument("--records", type=int, default=60000)
    parser.add_argument("--master", default="local[2]", help="Spark master URL; default two local Spark execution cores")
    parser.add_argument("--train-limit", type=int, default=4800)
    parser.add_argument("--test-limit", type=int, default=1200)
    parser.add_argument("--validation-limit", type=int, default=800)
    parser.add_argument("--bert-epochs", type=int, default=3)
    args = parser.parse_args()
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
        from crime_nlp.enrich import enrich as process
        process()
    stages.update(spark=spark, train=train, enrich=enrich)
    selected = list(stages) if args.stage == "all" else [args.stage]
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args), "python": platform.python_version(), "platform": platform.platform(), "stages": {}, "status": "running"}
    status_path = ARTIFACTS / f"run_{args.stage}.json"
    started = time.perf_counter()
    try:
        for name in selected:
            print(f"\n>>> Stage: {name}", flush=True)
            stage_start = time.perf_counter()
            status["stages"][name] = {"status": "running"}
            write_json(status_path, status)
            stages[name]()
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
    print(f"\nCompleted in {status['seconds']:.1f}s. Launch: python -m streamlit run app.py", flush=True)


def _installed(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
