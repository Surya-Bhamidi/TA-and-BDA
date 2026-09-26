"""Real BERT token classification with word/subword alignment and long-text windows."""
import json
import numpy as np
from .config import ENTITY_LABELS, RUNTIME, MODELS, SEED, write_json
from .text import tokens_with_offsets, bio_labels, spans_from_bio
from .evaluation import entity_metrics, repair_bio

BIO_TAGS = ["O"] + [prefix + name for name in ENTITY_LABELS for prefix in ("B-", "I-")]


def align_labels(encoding, word_labels):
    """Only each word's first subtoken contributes to loss; others use -100."""
    aligned = []
    for i, labels in enumerate(word_labels):
        previous, row = None, []
        for word_id in encoding.word_ids(batch_index=i):
            row.append(BIO_TAGS.index(labels[word_id]) if word_id is not None and word_id != previous else -100)
            previous = word_id
        aligned.append(row)
    return aligned


def sequences(part):
    token_lists, gold = [], []
    for row in part.itertuples():
        tokens = tokens_with_offsets(row.narrative)
        token_lists.append(tokens)
        gold.append(bio_labels(tokens, json.loads(row.entities_json)))
    return token_lists, gold


def bounded_windows(words, tokenizer, max_tokens, offset=0):
    """Split exceptional subword-heavy inputs while preserving every word offset."""
    size = len(tokenizer(words, is_split_into_words=True, add_special_tokens=True)["input_ids"])
    if size <= max_tokens:
        return [(offset, words)]
    if len(words) <= 1:
        raise ValueError("A token exceeds the entity model's supported input length")
    middle = len(words) // 2
    return bounded_windows(words[:middle], tokenizer, max_tokens, offset) + bounded_windows(words[middle:], tokenizer, max_tokens, offset + middle)


def predict_tags(model, tokenizer, token_lists, batch_size=32):
    """Use overlapping 48-word windows; no part of a long case is silently dropped."""
    import torch
    windows, mapping = [], []
    for doc_id, tokens in enumerate(token_lists):
        for start in range(0, len(tokens), 32):
            chunk = tokens[start:start + 48]
            words = [x["text"] for x in chunk]
            for offset, bounded in bounded_windows(words, tokenizer, model.config.max_position_embeddings, start):
                windows.append(bounded)
                mapping.append((doc_id, offset, len(bounded)))
    sums = [np.zeros((len(tokens), len(BIO_TAGS)), dtype=np.float32) for tokens in token_lists]
    counts = [np.zeros(len(tokens), dtype=np.float32) for tokens in token_lists]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = tokenizer(windows[start:start + batch_size], is_split_into_words=True, truncation=False, padding=True, return_tensors="pt")
            logits = model(**batch).logits.softmax(-1).cpu().numpy()
            for row in range(len(logits)):
                doc_id, offset, _ = mapping[start + row]
                previous = None
                for position, word_id in enumerate(batch.word_ids(batch_index=row)):
                    if word_id is not None and word_id != previous:
                        sums[doc_id][offset + word_id] += logits[row, position]
                        counts[doc_id][offset + word_id] += 1
                    previous = word_id
    predictions, confidence = [], []
    for scores, count in zip(sums, counts):
        averages = scores / np.maximum(count[:, None], 1)
        predictions.append(repair_bio([BIO_TAGS[i] for i in averages.argmax(axis=1)]))
        confidence.append(averages.max(axis=1).tolist())
    return predictions, confidence


def train_ner(train, validation, epochs=5):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, set_seed
    set_seed(SEED)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(RUNTIME / "bert", local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(RUNTIME / "bert", num_labels=len(BIO_TAGS), id2label=dict(enumerate(BIO_TAGS)), label2id={v: k for k, v in enumerate(BIO_TAGS)}, local_files_only=True, attn_implementation="eager")
    train_tokens, train_gold = sequences(train)
    words, tags = [], []
    for tokens, labels in zip(train_tokens, train_gold):
        for start in range(0, len(tokens), 40):
            words.append([t["text"] for t in tokens[start:start + 64]])
            tags.append(repair_bio(labels[start:start + 64]))
    encoded = tokenizer(words, is_split_into_words=True, padding=True, truncation=False, return_tensors="pt")
    labels = torch.tensor(align_labels(encoded, tags))
    val_tokens, val_gold = sequences(validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    random = torch.Generator().manual_seed(SEED)
    history, best = [], -1.
    target = MODELS / "bert_ner"
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=random)
        losses = []
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            output = model(**{key: value[indices] for key, value in encoded.items()}, labels=labels[indices])
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(output.loss.detach()))
        guesses, _ = predict_tags(model, tokenizer, val_tokens)
        score = entity_metrics(val_gold, guesses)["strict_entity_f1"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_entity_f1": score})
        print(f"BERT NER epoch {epoch + 1}/{epochs}: validation span F1={score:.4f}", flush=True)
        if score > best:
            best = score
            model.save_pretrained(target, safe_serialization=True)
            tokenizer.save_pretrained(target)
    write_json(target / "training.json", {"history": history, "validation_f1": best, "labels": BIO_TAGS, "train_rows": len(train), "validation_rows": len(validation), "window_words": 48, "stride_words": 32})
    return {"history": history, "validation_f1": best}


def load_ner():
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    torch.set_num_threads(4)
    path = MODELS / "bert_ner"
    return AutoModelForTokenClassification.from_pretrained(path, local_files_only=True), AutoTokenizer.from_pretrained(path, local_files_only=True)


def extract_bert(text, model, tokenizer):
    tokens = tokens_with_offsets(text)
    tags, confidence = predict_tags(model, tokenizer, [tokens])
    entities = spans_from_bio(text, tokens, tags[0])
    for entity in entities:
        values = [p for t, p in zip(tokens, confidence[0]) if entity["start"] <= t["start"] < entity["end"]]
        entity["confidence"] = float(np.mean(values))
    return entities
