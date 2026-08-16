"""Evaluation utilities with trial-level primary metrics."""

from .metrics import aggregate_trial_probabilities, evaluate_binary_predictions

__all__ = ["aggregate_trial_probabilities", "evaluate_binary_predictions"]

