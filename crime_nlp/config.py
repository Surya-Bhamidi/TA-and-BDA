"""Project-local paths and runtime configuration; no machine-wide changes."""
import json
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "reports.jsonl"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models" / "v3_1"
RUNTIME = ROOT / ".runtime"
SEED = 42
with (ROOT / "settings.toml").open("rb") as settings_file:
    SETTINGS = tomllib.load(settings_file)
SEED = int(SETTINGS["project"]["seed"])
ENTITY_LABELS = ["SUSPECT", "VICTIM", "LOCATION", "WEAPON", "PROPERTY", "DATE", "TIME", "EVIDENCE"]
BERT_ID = "google/bert_uncased_L-2_H-128_A-2"
CATEGORIES = ["Arson", "Assault", "Burglary", "Cybercrime", "Fraud", "Robbery", "Theft", "Vandalism"]


def configure():
    for path in (RAW.parent, PROCESSED, ARTIFACTS, MODELS, RUNTIME, ROOT / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(RUNTIME / "huggingface"))
    os.environ.setdefault("NLTK_DATA", str(RUNTIME / "nltk_data"))
    if os.name == "nt":
        # Spark 3.5's batch helper does not quote Python paths in `where`/`if`.
        # Search the active venv first and give that helper a space-free command.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    python_command = "python.exe" if os.name == "nt" else sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_command)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_command)
    # Spark's Windows helper invokes Python without quoting a path containing spaces.
    # Supply SPARK_HOME explicitly so that helper does not need to rediscover it.
    import importlib.util
    spark_spec = importlib.util.find_spec("pyspark")
    if spark_spec and spark_spec.submodule_search_locations:
        os.environ.setdefault("SPARK_HOME", next(iter(spark_spec.submodule_search_locations)))
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    java = list((RUNTIME / "java17").glob("*/bin/java.exe"))
    if java and os.name == "nt":
        os.environ["JAVA_HOME"] = str(java[0].parents[1])
    if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
        for candidate in (RUNTIME / "hadoop", Path("C:/hadoop")):
            if (candidate / "bin" / "winutils.exe").exists():
                os.environ["HADOOP_HOME"] = str(candidate)
                os.environ["PATH"] = str(candidate / "bin") + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    # Keep the handle alive for the life of the process.
                    globals()["_hadoop_dll"] = os.add_dll_directory(str(candidate / "bin"))
                break


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    tmp.replace(path)


configure()
