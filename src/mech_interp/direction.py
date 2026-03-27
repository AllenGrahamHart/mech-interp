from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mech_interp.config import DirectionConfig
from mech_interp.utils import normalize_vector


@dataclass
class DirectionFit:
    direction: np.ndarray
    threshold: float
    positive_mean: np.ndarray
    negative_mean: np.ndarray
    train_metrics: dict[str, float]


def _scores_to_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    predictions = (scores >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def fit_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    config: DirectionConfig,
) -> DirectionFit:
    positive_mask = labels == config.positive_label
    negative_mask = ~positive_mask

    if not positive_mask.any() or not negative_mask.any():
        raise ValueError("Need both positive and negative examples to fit a direction.")

    if config.method != "mean_diff":
        raise NotImplementedError(
            f"Direction method '{config.method}' is not implemented in the baseline."
        )

    positive_mean = activations[positive_mask].mean(axis=0)
    negative_mean = activations[negative_mask].mean(axis=0)
    direction = positive_mean - negative_mean
    if config.normalize:
        direction = normalize_vector(direction)

    scores = project_onto_direction(activations, direction)
    threshold = 0.5 * (
        scores[positive_mask].mean() + scores[negative_mask].mean()
    )
    metrics = _scores_to_metrics(scores=scores, labels=labels, threshold=threshold)

    return DirectionFit(
        direction=direction.astype(np.float32),
        threshold=float(threshold),
        positive_mean=positive_mean.astype(np.float32),
        negative_mean=negative_mean.astype(np.float32),
        train_metrics=metrics,
    )


def project_onto_direction(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return activations @ direction


def evaluate_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    scores = project_onto_direction(activations, direction)
    return _scores_to_metrics(scores=scores, labels=labels, threshold=threshold)

