"""Existing TF-IDF representation at word and character resolution.

Character n-grams tolerate unknown spellings without rewriting the source.
Both branches still feed the project's multinomial logistic regression.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def robust_tfidf():
    return FeatureUnion([
        ("word", TfidfVectorizer(max_features=18000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=26000, min_df=3, sublinear_tf=True)),
    ], transformer_weights={"word": 1., "char": .75})


def keyword_vectorizer(classifier):
    vectorizer = classifier.named_steps["vectorizer"]
    return dict(vectorizer.transformer_list)["word"] if isinstance(vectorizer, FeatureUnion) else vectorizer


def prediction_review(classifier, text, probabilities, threshold=.6, raw_probabilities=None):
    """Extend the existing confidence review flag with observed input coverage.

    Temperature can sharpen weak evidence; retain the uncalibrated margin and
    word coverage as explicit reasons for review. This is not a new classifier.
    """
    import numpy as np
    vectorizer = keyword_vectorizer(classifier)
    words = vectorizer.build_tokenizer()(text.lower())
    known = sum(word in vectorizer.vocabulary_ for word in words)
    coverage = known / max(len(words), 1)
    raw = classifier.predict_proba([text])[0] if raw_probabilities is None else np.asarray(raw_probabilities)
    ranked = np.sort(raw)
    reasons = []
    if len(words) < 3:
        reasons.append("Too little detail to identify an incident reliably.")
    if coverage < .35:
        reasons.append("Much of this wording is unfamiliar to the training data.")
    if max(probabilities) < threshold or max(raw) < .45:
        reasons.append("The category prediction has limited supporting evidence.")
    if ranked[-1] - ranked[-2] < .15:
        reasons.append("More than one category is plausible.")
    return {"review_required": bool(reasons), "review_reasons": reasons,
            "word_coverage": coverage, "raw_model_score": float(max(raw))}
