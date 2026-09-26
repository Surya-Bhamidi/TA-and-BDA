"""Local sentence embeddings, hybrid retrieval, and evidence-grounded answers."""
from functools import lru_cache
import numpy as np
from .config import RUNTIME, ARTIFACTS, SETTINGS
from .text import sentences


@lru_cache(maxsize=1)
def encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    path = RUNTIME / "semantic_encoder"
    return AutoModel.from_pretrained(path, local_files_only=True).eval(), AutoTokenizer.from_pretrained(path, local_files_only=True)


def encode(texts, batch_size=64, progress=False):
    import torch
    model, tokenizer = encoder()
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            output.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
            if progress and start % 6400 == 0:
                print(f"Semantic index: {min(start + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    return np.concatenate(output) if output else np.empty((0, 384), dtype=np.float32)


@lru_cache(maxsize=1)
def embedding_index():
    return np.load(ARTIFACTS / "semantic_embeddings.npy", mmap_mode="r")


def semantic_scores(query):
    vector = encode([query])[0]
    matrix = embedding_index()
    return np.concatenate([np.asarray(matrix[start:start + 4096], dtype=np.float32) @ vector for start in range(0, len(matrix), 4096)])


def rank_scores(lexical, semantic, mode="Hybrid"):
    if mode == "Keyword":
        return lexical
    if mode == "Semantic":
        return np.maximum(semantic, 0)
    # Reciprocal-rank fusion avoids pretending lexical and semantic cosine scales match.
    combined = np.zeros(len(lexical), dtype=np.float64)
    weight = SETTINGS["inference"]["semantic_weight"]
    for scores, contribution in [(lexical, 1 - weight), (semantic, weight)]:
        order = np.argsort(-scores, kind="stable")
        valid = scores[order] > (.02 if scores is lexical else .25)
        ranks = np.arange(1, len(order) + 1)
        combined[order[valid]] += contribution / (60 + ranks[valid])
    return combined


def semantic_evidence(question, text, report_id="custom narrative"):
    """Return verbatim source sentences and a citation; never invent an answer."""
    source = sentences(text)
    if not source or not question.strip():
        return {"answer": "Insufficient evidence in this narrative.", "citations": [], "score": 0.}
    # Avoid unbounded attention over an arbitrary uploaded file.
    scores = encode(source) @ encode([question])[0]
    order = np.argsort(-scores)[:2]
    chosen = [int(i) for i in order if scores[i] >= .30]
    if not chosen:
        return {"answer": "Insufficient evidence in this narrative.", "citations": [], "score": float(scores.max())}
    return {"answer": " ".join(source[i] for i in sorted(chosen)), "score": float(scores[order[0]]),
            "citations": [{"report_id": report_id, "sentence": i + 1, "text": source[i]} for i in chosen],
            "method": "Semantic evidence retrieval; threshold heuristic, not an answer-confidence probability"}
