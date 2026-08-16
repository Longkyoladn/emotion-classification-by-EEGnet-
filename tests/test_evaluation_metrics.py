from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_emotion.evaluation import (
    aggregate_trial_probabilities,
    evaluate_binary_predictions,
)


def test_trial_probability_is_mean_of_windows() -> None:
    predictions = pd.DataFrame(
        [
            {
                "sample_id": "a1",
                "subject": "s1",
                "trial": 1,
                "split_group": "s1-trial-1",
                "label": "high",
                "probability_high": 0.9,
            },
            {
                "sample_id": "a2",
                "subject": "s1",
                "trial": 1,
                "split_group": "s1-trial-1",
                "label": "high",
                "probability_high": 0.3,
            },
            {
                "sample_id": "b1",
                "subject": "s1",
                "trial": 2,
                "split_group": "s1-trial-2",
                "label": "low",
                "probability_high": 0.2,
            },
        ]
    )
    trials = aggregate_trial_probabilities(predictions)

    assert len(trials) == 2
    assert trials.loc[trials["trial"].eq(1), "probability_high"].item() == 0.6
    assert trials.loc[trials["trial"].eq(1), "prediction"].item() == "high"


def test_primary_metrics_are_trial_level() -> None:
    predictions = pd.DataFrame(
        [
            {
                "sample_id": "a1",
                "subject": "s1",
                "trial": 1,
                "split_group": "s1-trial-1",
                "label": "high",
                "probability_high": 0.8,
            },
            {
                "sample_id": "a2",
                "subject": "s1",
                "trial": 1,
                "split_group": "s1-trial-1",
                "label": "high",
                "probability_high": 0.7,
            },
            {
                "sample_id": "b1",
                "subject": "s1",
                "trial": 2,
                "split_group": "s1-trial-2",
                "label": "low",
                "probability_high": 0.1,
            },
        ]
    )
    result = evaluate_binary_predictions(predictions)

    assert result["primary_level"] == "trial"
    assert result["trial_level_primary"]["samples"] == 2
    assert result["window_level_secondary"]["samples"] == 3
    assert result["trial_level_primary"]["balanced_accuracy"] == 1.0
