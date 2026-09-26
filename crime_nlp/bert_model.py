"""Fine-tune real, pretrained two-layer BERT for narrative sequence classification."""
import time
import numpy as np

from .config import RUNTIME, MODELS, SEED, CATEGORIES


def fit_bert(train_text, train_y, val_text, val_y, test_text, epochs=3, batch_size=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
    from sklearn.metrics import f1_score
    set_seed(SEED)
    torch.set_num_threads(4)
    source = RUNTIME / "bert"
    if not (source / "config.json").exists():
        raise RuntimeError("Pretrained BERT is missing. Run python scripts/download_resources.py first.")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(CATEGORIES), id2label=dict(enumerate(CATEGORIES)),
        label2id={v: k for k, v in enumerate(CATEGORIES)}, local_files_only=True,
        attn_implementation="eager")
    train_encoding = tokenizer(list(train_text), padding="max_length", truncation=True, max_length=160, return_tensors="pt")
    labels = torch.tensor([CATEGORIES.index(y) for y in train_y])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    generator = torch.Generator().manual_seed(SEED)
    history, best = [], -1.
    started = time.perf_counter()
    dest = MODELS / "bert_classifier"
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = {key: value[indices] for key, value in train_encoding.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch, labels=labels[indices])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(out.loss.detach()))
        predicted = predict_bert(model, tokenizer, val_text)
        f1 = f1_score(val_y, predicted, labels=CATEGORIES, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_f1": float(f1)})
        print(f"BERT epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}, validation macro-F1={f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(dest, safe_serialization=True)
            tokenizer.save_pretrained(dest)
    model = AutoModelForSequenceClassification.from_pretrained(dest, local_files_only=True)
    prediction = predict_bert(model, tokenizer, test_text)
    return prediction, {"history": history, "best_validation_macro_f1": best, "train_seconds": round(time.perf_counter() - started, 2), "architecture": "Pretrained BERT: 2 layers, 128 hidden units, 2 attention heads; all layers fine-tuned", "max_tokens": 160}


def predict_bert(model, tokenizer, texts, batch_size=64):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True, max_length=160, return_tensors="pt")
            ids = model(**inputs).logits.argmax(dim=-1).tolist()
            result.extend(CATEGORIES[i] for i in ids)
    return result
