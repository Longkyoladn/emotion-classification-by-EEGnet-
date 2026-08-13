from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_emotion.data.splits import (
    build_nested_loso_splits,
    build_personalization_splits,
    build_sensitivity_split,
    build_subject_dependent_splits,
)

TASK = "valence_binary_fixed5"


def example_index(include_sensitivity: bool = False) -> pd.DataFrame:
    rows = []
    subjects = ["s1", "s2", "s3", "s4"]
    if include_sensitivity:
        subjects.append("s5")
    for subject in subjects:
        trials = (
            ((1, "low", 2.0),)
            if subject == "s5"
            else (
                (1, "low", 2.0),
                (2, "low", 4.0),
                (3, "high", 7.0),
                (4, "high", 9.0),
            )
        )
        for trial, label, valence in trials:
            for window in (1, 2):
                rows.append(
                    {
                        "sample_id": f"{subject}-{trial}-{window}",
                        "subject": subject,
                        "trial": trial,
                        "split_group": f"{subject}-trial-{trial}",
                        "valence": valence,
                        f"eligible_{TASK}": True,
                        f"label_{TASK}": label,
                    }
                )
    return pd.DataFrame(rows)


def test_subject_dependent_is_trial_stratified_and_repeated() -> None:
    splits, summary = build_subject_dependent_splits(
        example_index(), ["s1", "s2", "s3", "s4"], TASK, repeats=3, seed=7
    )

    assert summary["all_folds_leakage_free"] is True
    assert summary["all_folds_class_complete"] is True
    assert (
        splits.groupby(["subject", "repeat", "fold", "split_group"])["partition"]
        .nunique()
        .eq(1)
        .all()
    )
    test = splits[splits["partition"].eq("test")]
    assert test.groupby(["subject", "repeat", "sample_id"]).size().eq(1).all()
    for _, fold in splits.groupby(["subject", "repeat", "fold"]):
        training_trials = fold[fold["partition"].eq("train")].drop_duplicates(
            "split_group"
        )
        assert fold["subject_relative_threshold"].nunique() == 1
        assert (
            fold["subject_relative_threshold"].iloc[0]
            == training_trials["valence"].median()
        )


def test_nested_loso_excludes_outer_test_and_has_refit_stage() -> None:
    subjects = ["s1", "s2", "s3", "s4"]
    splits, summary = build_nested_loso_splits(example_index(), subjects, TASK)

    assert summary["all_inner_folds_exclude_outer_test"] is True
    assert summary["all_folds_leakage_free"] is True
    assert summary["all_folds_class_complete"] is True
    assert "label_subject_relative" not in splits.columns
    for outer_fold, test_subject in enumerate(subjects, start=1):
        inner = splits[
            splits["outer_fold"].eq(outer_fold) & splits["stage"].eq("inner_selection")
        ]
        outer = splits[
            splits["outer_fold"].eq(outer_fold) & splits["stage"].eq("outer_evaluation")
        ]
        assert test_subject not in set(inner["subject"])
        assert set(outer.loc[outer["partition"].eq("test"), "subject"]) == {
            test_subject
        }
        assert set(outer.loc[outer["partition"].eq("refit"), "subject"]) == (
            set(subjects) - {test_subject}
        )


def test_personalization_enumerates_calibration_pairs_without_trial_leakage() -> None:
    subjects = ["s1", "s2", "s3", "s4"]
    splits, summary = build_personalization_splits(example_index(), subjects, TASK)

    assert summary["total_calibration_scenarios"] == 16
    assert summary["all_scenarios_leakage_free"] is True
    assert summary["all_calibrations_have_one_trial_per_class"] is True
    assert summary["all_tests_have_two_classes"] is True
    assert (
        splits.groupby(["target_subject", "scenario", "split_group"])["partition"]
        .nunique()
        .eq(1)
        .all()
    )
    for (target, _), scenario in splits.groupby(["target_subject", "scenario"]):
        target_rows = scenario[scenario["subject"].eq(target)]
        calibration_trials = target_rows[
            target_rows["partition"].eq("calibration")
        ].drop_duplicates("split_group")
        assert target_rows["subject_relative_threshold"].nunique() == 1
        assert target_rows["subject_relative_threshold"].iloc[0] == (
            calibration_trials["valence"].median()
        )
        for source, source_rows in scenario[
            scenario["partition"].eq("pretrain")
        ].groupby("subject"):
            source_trials = source_rows.drop_duplicates("split_group")
            assert source != target
            assert source_rows["subject_relative_threshold"].nunique() == 1
            assert (
                source_rows["subject_relative_threshold"].iloc[0]
                == source_trials["valence"].median()
            )


def test_sensitivity_subject_is_test_only_and_descriptive() -> None:
    splits, summary = build_sensitivity_split(
        example_index(include_sensitivity=True),
        ["s1", "s2", "s3", "s4"],
        ["s5"],
        TASK,
    )

    assert summary["descriptive_only"] is True
    assert "label_subject_relative" not in splits.columns
    assert set(splits.loc[splits["subject"].eq("s5"), "partition"]) == {
        "sensitivity_test"
    }
