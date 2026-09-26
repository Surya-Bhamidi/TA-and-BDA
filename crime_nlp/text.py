"""Shared tokenization, BIO alignment, extractive summary and transparent threat cues."""
import re
import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def word_tokens(text):
    return WORD_RE.findall(text.lower())


def tokens_with_offsets(text):
    return [{"text": m.group(), "start": m.start(), "end": m.end()} for m in TOKEN_RE.finditer(text)]


def bio_labels(tokens, entities):
    labels = ["O"] * len(tokens)
    for ent in entities:
        indices = [i for i, t in enumerate(tokens) if t["start"] >= ent["start"] and t["end"] <= ent["end"]]
        if not indices or tokens[indices[0]]["start"] != ent["start"] or tokens[indices[-1]]["end"] != ent["end"]:
            raise ValueError(f"Entity does not align with tokenizer: {ent}")
        for j, i in enumerate(indices):
            labels[i] = ("B-" if j == 0 else "I-") + ent["label"]
    return labels


def token_features(tokens):
    """Linear-chain CRF observations. Never read ground truth at inference."""
    # Sentence-level role cues supplement the narrow token window. These are
    # observations learned by the CRF, not post-hoc assignments of entity labels.
    # They allow role words to be separated from names by modifiers or clauses.
    contexts = [{} for _ in tokens]
    start = 0
    for end in range(len(tokens)):
        if tokens[end]["text"] in {".", "!", "?"} or end == len(tokens) - 1:
            words = {t["text"].lower() for t in tokens[start:end + 1]}
            cues = {"sentence:victim_cue": bool(words & {"victim", "complainant"}),
                    "sentence:suspect_cue": bool(words & {"suspect", "offender", "assailant", "responsible"})}
            for j in range(start, end + 1):
                contexts[j] = cues
            start = end + 1
    result = []
    for i, token in enumerate(tokens):
        word = token["text"]
        f = {"bias": 1., "lower": word.lower(), "suffix3": word[-3:].lower(), "suffix2": word[-2:].lower(),
             "title": word.istitle(), "upper": word.isupper(), "digit": word.isdigit(), "punct": not word.isalnum()}
        f.update(contexts[i])
        for offset in [-2, -1, 1, 2]:
            j = i + offset
            if 0 <= j < len(tokens):
                other = tokens[j]["text"]
                f[f"{offset}:lower"] = other.lower()
                f[f"{offset}:title"] = other.istitle()
            else:
                f[f"{offset}:boundary"] = True
        result.append(f)
    return result


def spans_from_bio(text, tokens, labels):
    spans, active = [], None
    for tok, tag in zip(tokens, labels):
        prefix, _, label = tag.partition("-")
        if tag == "O":
            active = None
            continue
        if prefix == "B" or active is None or active["label"] != label:
            active = {"start": tok["start"], "end": tok["end"], "label": label}
            spans.append(active)
        else:
            active["end"] = tok["end"]
    for span in spans:
        span["text"] = text[span["start"]:span["end"]]
    return spans


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def summarize(text, count=2):
    """TextRank over TF-IDF sentence cosine similarities, preserving source order."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import networkx as nx
    sents = sentences(text)
    if len(sents) <= count:
        return " ".join(sents)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:
        return " ".join(sents[:count])
    sim = (matrix @ matrix.T).toarray()
    np.fill_diagonal(sim, 0)
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, max_iter=200)
    # Mild lead prior anchors the summary in the described event; still extractive.
    selected = sorted(sorted(scores, key=lambda i: scores[i] + (.12 if i == 0 else 0), reverse=True)[:count])
    return " ".join(sents[i] for i in selected)


THREAT_CUES = {"knife": 3, "handgun": 4, "weapon": 2, "threatened": 3, "threats": 3, "injured": 3,
               "injuries": 3, "assailant": 2, "violent": 2, "attack": 2, "attacked": 2, "fire": 2,
               "blaze": 2, "ransomware": 1, "malware": 1, "punched": 2, "kicked": 2}


def threat_language(text):
    """Demonstration heuristic about language, not a validated safety assessment."""
    found = {}
    for clause in re.split(r"[.!?;,]|\bbut\b", text.lower()):
        words = word_tokens(clause)
        for i, word in enumerate(words):
            if word in THREAT_CUES:
                context = words[max(0, i - 4):i]
                if not set(context) & {"no", "not", "without", "never", "denied", "nobody"}:
                    found[word] = THREAT_CUES[word]
    score = min(sum(found.values()), 12)
    return {"score": score, "level": "High" if score >= 6 else "Moderate" if score >= 2 else "Low", "cues": sorted(found), "method": "negation-aware cue heuristic"}


def evidence_answer(question, text):
    """Return matching source sentences, with no invented answer or facts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = sentences(text)
    if not question.strip() or not sents:
        return {"answer": "No matching evidence found.", "score": 0., "method": "lexical evidence retrieval"}
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(sents)
        scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
    except ValueError:
        scores = np.zeros(len(sents))
    index = int(scores.argmax())
    return {"answer": sents[index] if scores[index] > .04 else "No matching evidence found.",
            "score": float(scores[index]), "method": "lexical evidence retrieval; score is similarity, not confidence"}
