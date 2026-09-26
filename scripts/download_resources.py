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
    semantic_id = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_dest = RUNTIME / "semantic_encoder"
    if not (semantic_dest / "model.safetensors").exists():
        semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_id)
        semantic_model = AutoModel.from_pretrained(semantic_id)
        semantic_tokenizer.save_pretrained(semantic_dest)
        semantic_model.save_pretrained(semantic_dest, safe_serialization=True)
        write_json(RUNTIME / "semantic_source.json", {"model_id": semantic_id, "commit": getattr(semantic_model.config, "_commit_hash", None), "pooling": "attention-mask mean; L2 normalized", "dimensions": 384})
    nltk.download("vader_lexicon", download_dir=str(RUNTIME / "nltk_data"), raise_on_error=True)
    nlp = spacy.load("en_core_web_sm")
    print("Ready: pretrained BERT, semantic encoder, VADER, spaCy", nlp.pipe_names, flush=True)


if __name__ == "__main__":
    main()
