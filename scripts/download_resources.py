"""Download actual pretrained BERT and VADER into the project for offline inference."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import RUNTIME, BERT_ID, write_json


def main():
    import nltk
    from transformers import AutoTokenizer, AutoModel
    import spacy
    dest = RUNTIME / "bert"
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    model = AutoModel.from_pretrained(BERT_ID)
    tokenizer.save_pretrained(dest)
    model.save_pretrained(dest, safe_serialization=True)
    write_json(RUNTIME / "bert_source.json", {"model_id": BERT_ID, "commit": getattr(model.config, "_commit_hash", None), "architecture": model.config.architectures, "hidden_size": model.config.hidden_size, "layers": model.config.num_hidden_layers})
    nltk.download("vader_lexicon", download_dir=str(RUNTIME / "nltk_data"), raise_on_error=True)
    nlp = spacy.load("en_core_web_sm")
    print("Ready: pretrained BERT, VADER, spaCy", nlp.pipe_names, flush=True)


if __name__ == "__main__":
    main()
