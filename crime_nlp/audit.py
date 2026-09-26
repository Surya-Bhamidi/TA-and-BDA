"""Content hashes, stage reuse checks, and append-only run manifests."""
import hashlib
import json
from pathlib import Path
from .config import ROOT, RAW, PROCESSED, ARTIFACTS, MODELS, write_json


def digest_path(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in paths:
        digest.update((file.relative_to(path).as_posix() if path.is_dir() else file.name).encode())
        with file.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def stage_outputs(stage):
    return {"generate": [RAW, ARTIFACTS / "dataset_manifest.json"],
            "spark": [PROCESSED / "reports", ARTIFACTS / "spark_metrics.json", ARTIFACTS / "data_quality.json"],
            "train": [MODELS, ARTIFACTS / "evaluation.json", ARTIFACTS / "ner_evaluation.json", ARTIFACTS / "topics.json"],
            "enrich": [PROCESSED / "dashboard.parquet", PROCESSED / "cases.sqlite", ARTIFACTS / "semantic_embeddings.npy", ARTIFACTS / "search_matrix.npz"]}[stage]


def stage_signature(stage, arguments):
    source = hashlib.sha256()
    for file in sorted((ROOT / "crime_nlp").glob("*.py")) + [ROOT / "settings.toml", ROOT / "run_pipeline.py"]:
        source.update(file.read_bytes())
    prior = {"generate": [], "spark": [RAW], "train": [PROCESSED / "reports"], "enrich": [PROCESSED / "reports", MODELS]}[stage]
    parameters = {k: v for k, v in arguments.items() if k not in {"stage", "resume"}}
    source.update(json.dumps(parameters, sort_keys=True).encode())
    for path in prior:
        source.update(str(digest_path(path)).encode())
    return source.hexdigest()


def record_stage(stage, signature, outputs):
    hashes = {str(p.relative_to(ROOT)): digest_path(p) for p in outputs}
    if any(value is None for value in hashes.values()):
        raise RuntimeError(f"Stage {stage} did not produce all required outputs")
    write_json(ARTIFACTS / "stage_state" / f"{stage}.json", {"input_sha256": signature, "outputs": hashes})


def reusable_stage(stage, signature):
    path = ARTIFACTS / "stage_state" / f"{stage}.json"
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    return state["input_sha256"] == signature and all(digest_path(ROOT / p) == value for p, value in state["outputs"].items())
