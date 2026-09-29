"""Shared tokenization, BIO alignment, extractive summary and transparent threat cues."""
import re
import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# Shared CRF context observations, not rules that assign entity labels. The
# same cues are used for every name, spelling, casing and country.
ROLE_ACTIONS = set("hit hits hitting beat beats beaten beating punch punched slap slapped kick kicked attack attacked stab stabbed strike struck steal stole stolen rob robbed cheat cheated trick tricked hack hacked threaten threatened snatch snatched grab grabbed damage damaged burn burned burnt torch torched deface defaced force forced".split())
ROLE_ACTIONS.update("kill killed killing murder murdered shoot shot shooting assault assaulted taken took".split())

ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "shri", "smt", "no", "nos", "sr", "jr", "st", "hrs", "approx", "etc"}


def sentence_break(tokens, index):
    """A dot in dates, initials and administrative abbreviations isn't a stop."""
    current = tokens[index]
    if current["text"] in {"!", "?"}:
        return True
    if current["text"] != ".":
        return False
    before = tokens[index - 1]["text"] if index else ""
    after = tokens[index + 1]["text"] if index + 1 < len(tokens) else ""
    if before.isdigit() and after.isdigit() and tokens[index - 1]["end"] == current["start"] and current["end"] == tokens[index + 1]["start"]:
        return False
    if before.lower() in ABBREVIATIONS or (len(before) == 1 and before.isalpha()):
        return False
    return True


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
    lowered = [t["text"].lower() for t in tokens]
    start = 0
    for end in range(len(tokens)):
        if sentence_break(tokens, end) or end == len(tokens) - 1:
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
             "title": word.istitle(), "upper": word.isupper(), "digit": word.isdigit(), "punct": not word.isalnum(),
             "prefix2": word[:2].lower(), "prefix3": word[:3].lower(), "alpha": word.isalpha(),
             "length": min(len(word), 15), "shape": re.sub(r"a+", "a", re.sub(r"[^\W\d_]", "a", word.lower())),
             "action_word": word.lower() in ROLE_ACTIONS,
             "BOS": i == 0, "EOS": i == len(tokens) - 1}
        f.update(contexts[i])
        # Wider lexical context distinguishes "A hit B" / "A was hit by B"
        # while staying a linear-chain CRF with hand-designed observations.
        for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
            j = i + offset
            if 0 <= j < len(tokens):
                other = tokens[j]["text"]
                f[f"{offset}:lower"] = other.lower()
                f[f"{offset}:title"] = other.istitle()
                f[f"{offset}:digit"] = other.isdigit()
            else:
                f[f"{offset}:boundary"] = True
        f["previous_pair"] = "|".join(lowered[max(0, i - 2):i])
        f["next_pair"] = "|".join(lowered[i + 1:i + 3])
        for side, step in [("left", -1), ("right", 1)]:
            # A compact cue-distance observation transfers the same relationship
            # from a multiword name to an unseen lowercase mononym.
            found = set()
            for distance in range(1, 9):
                j = i + step * distance
                if not 0 <= j < len(tokens) or lowered[j] == ";" or sentence_break(tokens, j):
                    break
                cue = lowered[j]
                kinds = {"action": cue in ROLE_ACTIONS, "place": cue in {"near", "at", "in", "outside"},
                         "by": cue == "by", "victim": cue in {"victim", "complainant"},
                         "suspect": cue in {"suspect", "accused", "offender"}}
                for kind, present in kinds.items():
                    if present and kind not in found:
                        f[f"{side}:{kind}:distance"] = str(distance)
                        found.add(kind)
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
    tokens = tokens_with_offsets(text)
    result, start = [], 0
    for i, token in enumerate(tokens):
        end = token["end"]
        if sentence_break(tokens, i) and (end == len(text) or text[end].isspace()):
            if text[start:end].strip():
                result.append(text[start:end].strip())
            start = end
    if text[start:].strip():
        result.append(text[start:].strip())
    return result


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
