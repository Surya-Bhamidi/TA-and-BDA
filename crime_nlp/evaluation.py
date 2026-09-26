"""Readable, reproducible statistics for document and entity models."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from .config import CATEGORIES, SEED


def classification_metrics(y, predicted):
    """Scores and a report-level bootstrap interval; rows share synthetic templates."""
    truth, guesses = np.asarray(y), np.asarray(predicted)
    random = np.random.default_rng(SEED)
    bootstrap = [np.mean(truth[index] == guesses[index]) for index in random.integers(0, len(truth), size=(400, len(truth)))]
    return {"accuracy": float(accuracy_score(truth, guesses)),
            "macro_f1": float(f1_score(truth, guesses, labels=CATEGORIES, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, guesses, average="weighted", zero_division=0)),
            "classification_report": classification_report(truth, guesses, labels=CATEGORIES, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, guesses, labels=CATEGORIES).tolist(), "labels": CATEGORIES,
            "accuracy_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "ci_note": "400 report-level bootstrap resamples; excludes uncertainty from new templates or real-world sources."}


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(np.asarray(probabilities), 1e-10, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, labels, classes):
    optimum = minimize_scalar(lambda t: log_loss(labels, temperature_scale(probabilities, t), labels=classes), bounds=(.25, 5.), method="bounded")
    return float(optimum.x)


def calibration_metrics(labels, probabilities, classes, threshold=.60):
    probability = np.asarray(probabilities)
    confidence = probability.max(axis=1)
    predicted = np.asarray(classes)[probability.argmax(axis=1)]
    correct = predicted == np.asarray(labels)
    bins, error = [], 0.
    for left in np.arange(0., 1., .1):
        chosen = (confidence >= left) & (confidence < left + .1 if left < .9 else confidence <= 1.)
        if chosen.any():
            accuracy, mean = float(correct[chosen].mean()), float(confidence[chosen].mean())
            error += abs(accuracy - mean) * chosen.mean()
            bins.append({"bin_start": round(float(left), 1), "mean_confidence": mean, "accuracy": accuracy, "count": int(chosen.sum())})
    accepted = confidence >= threshold
    return {"ece": float(error), "log_loss": float(log_loss(labels, probability, labels=classes)),
            "threshold": threshold, "coverage": float(accepted.mean()),
            "accuracy_when_accepted": float(correct[accepted].mean()) if accepted.any() else None, "bins": bins}


def entity_metrics(gold, predicted):
    from seqeval.metrics import classification_report as report, f1_score as f1
    from seqeval.scheme import IOB2
    return {"strict_entity_f1": float(f1(gold, predicted, mode="strict", scheme=IOB2, zero_division=0)),
            "report": report(gold, predicted, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)}


def repair_bio(tags):
    """An I-tag without a matching predecessor starts a new entity."""
    repaired, previous = [], "O"
    for tag in tags:
        if tag.startswith("I-") and previous not in {"B-" + tag[2:], tag}:
            tag = "B-" + tag[2:]
        repaired.append(tag)
        previous = tag
    return repaired
