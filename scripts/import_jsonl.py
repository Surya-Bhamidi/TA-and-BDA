"""Usage: python scripts/import_jsonl.py file.jsonl --source-url URL --license NAME"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.import_data import stage_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    arguments = parser.parse_args()
    path, receipt = stage_jsonl(arguments.file, arguments.source_url, arguments.license)
    print(path)
    print(receipt)
