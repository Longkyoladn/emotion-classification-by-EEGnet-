"""Aggregate window predictions and report trial-first binary metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PREDICTION_REQUIRED_COLUMNS = {
    "sample_id",
    "subject",
    "trial",
    "split_group",
    "label",
    "probability_high",
}


def aggregate_trial_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean window probabilities within trial; labels must be trial-consistent."""
    missing = PREDICTION_REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if not predictions.groupby("split_group")["label"].nunique().eq(1).all():
        raise ValueError("All windows from one trial must share the same label")
    if not predictions["probability_high"].between(0.0, 1.0).all():
        raise ValueError("probability_high must be in [0, 1]")

    trials = (
        predictions.groupby(["subject", "trial", "split_group"], as_index=False)
        .agg(
            label=("label", "first"),
            probability_high=("probability_high", "mean"),
            windows=("sample_id", "size"),
        )
        .sort_values(["subject", "trial"], ignore_index=True)
    )
    trials["prediction"] = trials["probability_high"].map(
        lambda probability: "high" if probability >= 0.5 else "low"
    )
    return trials


def _binary_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    mapping = {"low": 0, "high": 1}
    unknown = (set(frame["label"]) | set(frame[prediction_column])) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown binary labels: {sorted(unknown)}")
    truth = frame["label"].map(mapping)
    predicted = frame[prediction_column].map(mapping)
    labels = [0, 1]
    result: dict[str, Any] = {
        "samples": len(frame),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_low_high": confusion_matrix(
            truth, predicted, labels=labels
        ).tolist(),
    }
    if truth.nunique() == 2 and "probability_high" in frame:
        result["roc_auc"] = float(roc_auc_score(truth, frame["probability_high"]))
    else:
        result["roc_auc"] = None
    return result


def evaluate_binary_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return trial metrics as primary and window metrics as secondary only."""
    trials = aggregate_trial_probabilities(predictions)
    windows = predictions.copy()
    windows["prediction"] = windows["probability_high"].map(
        lambda probability: "high" if probability >= 0.5 else "low"
    )
    return {
        "primary_level": "trial",
        "probability_aggregation": "mean_across_windows",
        "trial_level_primary": _binary_metrics(trials, "prediction"),
        "window_level_secondary": _binary_metrics(windows, "prediction"),
    }
