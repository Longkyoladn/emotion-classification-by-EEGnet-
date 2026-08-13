"""Create deterministic, leakage-safe M2 split manifests."""

from __future__ import annotations

import hashlib
from itertools import product
from typing import Any

import numpy as np
import pandas as pd


def _task_frame(index: pd.DataFrame, subjects: list[str], task: str) -> pd.DataFrame:
    eligible_column = f"eligible_{task}"
    label_column = f"label_{task}"
    required = {
        "sample_id",
        "subject",
        "trial",
        "split_group",
        "valence",
        eligible_column,
        label_column,
    }
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Modeling index is missing columns: {sorted(missing)}")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject lists must not contain duplicates")

    frame = index[
        index["subject"].isin(subjects) & index[eligible_column].astype(bool)
    ].copy()
    frame = frame.rename(columns={label_column: "label"})
    absent = sorted(set(subjects) - set(frame["subject"].unique()))
    if absent:
        raise ValueError(f"Subjects without eligible samples for {task}: {absent}")
    return frame


def _trial_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["subject", "trial", "split_group", "label", "valence"]
    for column in ("label", "valence"):
        if not frame.groupby("split_group")[column].nunique().eq(1).all():
            raise ValueError(f"A split_group maps to multiple {column} values")
    return frame[columns].drop_duplicates("split_group").reset_index(drop=True)


def _counts(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "windows": len(frame),
        "trials": int(frame["split_group"].nunique()),
        "subjects": int(frame["subject"].nunique()),
        "window_labels": {
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "trial_labels": {
            str(label): int(count)
            for label, count in (
                _trial_table(frame)["label"].value_counts().sort_index().items()
            )
        },
    }


def _relative_labels(values: pd.Series, thresholds: pd.Series) -> np.ndarray:
    return np.where(
        values > thresholds,
        "high",
        np.where(values < thresholds, "low", "ambiguous"),
    )


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _partitions_are_group_atomic(frame: pd.DataFrame) -> bool:
    return bool(frame.groupby("split_group")["partition"].nunique().eq(1).all())


def _partitions_have_two_classes(frame: pd.DataFrame) -> bool:
    return all(
        partition["label"].nunique() >= 2 for _, partition in frame.groupby("partition")
    )


def build_subject_dependent_splits(
    index: pd.DataFrame,
    subjects: list[str],
    task: str,
    repeats: int = 10,
    seed: int = 20260814,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repeated stratified 2-fold splits at trial level for each subject."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    task_index = _task_frame(index, subjects, task)
    trial_table = _trial_table(task_index)
    rows: list[pd.DataFrame] = []
    subject_summaries: list[dict[str, Any]] = []

    for subject in subjects:
        subject_windows = task_index[task_index["subject"].eq(subject)].copy()
        subject_trials = trial_table[trial_table["subject"].eq(subject)].copy()
        trial_label_counts = subject_trials["label"].value_counts()
        if len(trial_label_counts) < 2 or int(trial_label_counts.min()) < 2:
            raise ValueError(
                f"{subject} needs at least two trials in each class for stratified 2-fold"
            )

        fold_audits: list[dict[str, Any]] = []
        for repeat in range(1, repeats + 1):
            test_groups: list[set[str]] = [set(), set()]
            for label in sorted(subject_trials["label"].unique()):
                groups = subject_trials.loc[
                    subject_trials["label"].eq(label), "split_group"
                ].to_numpy(copy=True)
                rng = np.random.default_rng(_stable_seed(seed, subject, repeat, label))
                rng.shuffle(groups)
                test_groups[0].update(str(group) for group in groups[0::2])
                test_groups[1].update(str(group) for group in groups[1::2])

            for fold_number, test_group_set in enumerate(test_groups, start=1):
                fold = subject_windows.copy()
                fold["partition"] = np.where(
                    fold["split_group"].isin(test_group_set), "test", "train"
                )
                train_trials = _trial_table(fold[fold["partition"].eq("train")])
                threshold = float(train_trials["valence"].median())
                fold["subject_relative_threshold"] = threshold
                fold["label_subject_relative"] = _relative_labels(
                    fold["valence"], fold["subject_relative_threshold"]
                )
                fold["eligible_subject_relative"] = fold["label_subject_relative"].ne(
                    "ambiguous"
                )
                fold.insert(
                    0, "protocol", "subject_dependent_repeated_stratified_2fold"
                )
                fold.insert(1, "repeat", repeat)
                fold.insert(2, "fold", fold_number)
                fold.insert(3, "task", task)
                rows.append(
                    fold[
                        [
                            "protocol",
                            "repeat",
                            "fold",
                            "task",
                            "partition",
                            "sample_id",
                            "subject",
                            "trial",
                            "split_group",
                            "label",
                            "valence",
                            "subject_relative_threshold",
                            "label_subject_relative",
                            "eligible_subject_relative",
                        ]
                    ]
                )

                atomic = _partitions_are_group_atomic(fold)
                class_complete = _partitions_have_two_classes(fold)
                fold_audits.append(
                    {
                        "repeat": repeat,
                        "fold": fold_number,
                        "leakage_free": atomic,
                        "both_partitions_have_two_classes": class_complete,
                        "subject_relative_threshold": threshold,
                        "partitions": {
                            name: _counts(partition)
                            for name, partition in fold.groupby("partition")
                        },
                    }
                )

        subject_summaries.append(
            {
                "subject": subject,
                "eligible_trials": len(subject_trials),
                "eligible_windows": len(subject_windows),
                "trial_labels": {
                    str(label): int(count)
                    for label, count in trial_label_counts.sort_index().items()
                },
                "folds": fold_audits,
            }
        )

    splits = pd.concat(rows, ignore_index=True).sort_values(
        ["subject", "repeat", "fold", "partition", "trial", "sample_id"],
        ignore_index=True,
    )
    per_repeat_test = (
        splits[splits["partition"].eq("test")]
        .groupby(["subject", "repeat", "sample_id"])
        .size()
    )
    if not per_repeat_test.eq(1).all():
        raise ValueError("Each sample must be test exactly once per repeat")
    if (
        not splits.groupby(["subject", "repeat", "fold", "split_group"])["partition"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise ValueError("Trial leakage detected in subject-dependent splits")

    all_audits = [fold for subject in subject_summaries for fold in subject["folds"]]
    summary = {
        "protocol": "subject_dependent_repeated_stratified_2fold",
        "task": task,
        "subjects": subjects,
        "repeats": repeats,
        "seed": seed,
        "stratification_unit": "trial",
        "subject_relative_threshold_source": "training_trials_only",
        "all_folds_leakage_free": all(fold["leakage_free"] for fold in all_audits),
        "all_folds_class_complete": all(
            fold["both_partitions_have_two_classes"] for fold in all_audits
        ),
        "subject_summaries": subject_summaries,
    }
    return splits, summary


def build_nested_loso_splits(
    index: pd.DataFrame,
    subjects: list[str],
    task: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nested LOSO with inner subject validation and an explicit outer refit stage."""
    if len(subjects) < 3:
        raise ValueError("Nested LOSO requires at least three subjects")
    task_index = _task_frame(index, subjects, task)
    rows: list[pd.DataFrame] = []
    outer_summaries: list[dict[str, Any]] = []

    for outer_fold, test_subject in enumerate(subjects, start=1):
        outer_train_subjects = [
            subject for subject in subjects if subject != test_subject
        ]
        inner_summaries: list[dict[str, Any]] = []
        for inner_fold, validation_subject in enumerate(outer_train_subjects, start=1):
            inner = task_index[task_index["subject"].isin(outer_train_subjects)].copy()
            inner["partition"] = np.where(
                inner["subject"].eq(validation_subject), "validation", "train"
            )
            inner.insert(0, "protocol", "cross_subject_nested_loso")
            inner.insert(1, "stage", "inner_selection")
            inner.insert(2, "outer_fold", outer_fold)
            inner.insert(3, "inner_fold", inner_fold)
            inner.insert(4, "task", task)
            rows.append(
                inner[
                    [
                        "protocol",
                        "stage",
                        "outer_fold",
                        "inner_fold",
                        "task",
                        "partition",
                        "sample_id",
                        "subject",
                        "trial",
                        "split_group",
                        "label",
                    ]
                ]
            )
            inner_summaries.append(
                {
                    "inner_fold": inner_fold,
                    "validation_subject": validation_subject,
                    "test_subject_absent": test_subject not in set(inner["subject"]),
                    "leakage_free": _partitions_are_group_atomic(inner),
                    "both_partitions_have_two_classes": _partitions_have_two_classes(
                        inner
                    ),
                    "partitions": {
                        name: _counts(partition)
                        for name, partition in inner.groupby("partition")
                    },
                }
            )

        outer = task_index.copy()
        outer["partition"] = np.where(
            outer["subject"].eq(test_subject), "test", "refit"
        )
        outer.insert(0, "protocol", "cross_subject_nested_loso")
        outer.insert(1, "stage", "outer_evaluation")
        outer.insert(2, "outer_fold", outer_fold)
        outer.insert(3, "inner_fold", pd.NA)
        outer.insert(4, "task", task)
        rows.append(
            outer[
                [
                    "protocol",
                    "stage",
                    "outer_fold",
                    "inner_fold",
                    "task",
                    "partition",
                    "sample_id",
                    "subject",
                    "trial",
                    "split_group",
                    "label",
                ]
            ]
        )
        outer_summaries.append(
            {
                "outer_fold": outer_fold,
                "test_subject": test_subject,
                "refit_subjects": outer_train_subjects,
                "inner_folds": inner_summaries,
                "outer_leakage_free": _partitions_are_group_atomic(outer),
                "outer_partitions_have_two_classes": _partitions_have_two_classes(
                    outer
                ),
                "outer_partitions": {
                    name: _counts(partition)
                    for name, partition in outer.groupby("partition")
                },
            }
        )

    splits = pd.concat(rows, ignore_index=True).sort_values(
        ["outer_fold", "stage", "inner_fold", "partition", "subject", "sample_id"],
        ignore_index=True,
    )
    inner_rows = splits[splits["stage"].eq("inner_selection")]
    for outer_fold, test_subject in enumerate(subjects, start=1):
        if test_subject in set(
            inner_rows.loc[inner_rows["outer_fold"].eq(outer_fold), "subject"]
        ):
            raise ValueError("Outer test subject leaked into inner selection")
    if (
        not splits.groupby(
            ["outer_fold", "stage", "inner_fold", "split_group"], dropna=False
        )["partition"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise ValueError("Split-group leakage detected in nested LOSO")

    inner_audits = [
        inner for outer in outer_summaries for inner in outer["inner_folds"]
    ]
    summary = {
        "protocol": "cross_subject_nested_loso",
        "task": task,
        "subjects": subjects,
        "outer_folds": len(subjects),
        "inner_subject_folds_per_outer_fold": len(subjects) - 1,
        "hyperparameter_selection_stage": "inner_selection",
        "final_training_stage": "outer_evaluation/refit",
        "subject_relative_labels_allowed": False,
        "all_inner_folds_exclude_outer_test": all(
            inner["test_subject_absent"] for inner in inner_audits
        ),
        "all_folds_leakage_free": all(
            outer["outer_leakage_free"] for outer in outer_summaries
        )
        and all(inner["leakage_free"] for inner in inner_audits),
        "all_folds_class_complete": all(
            outer["outer_partitions_have_two_classes"] for outer in outer_summaries
        )
        and all(inner["both_partitions_have_two_classes"] for inner in inner_audits),
        "folds": outer_summaries,
    }
    return splits, summary


def build_personalization_splits(
    index: pd.DataFrame,
    subjects: list[str],
    task: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enumerate every one-High/one-Low target calibration pair."""
    if len(subjects) < 2:
        raise ValueError("Personalization requires at least two subjects")
    task_index = _task_frame(index, subjects, task)
    trial_table = _trial_table(task_index)
    rows: list[pd.DataFrame] = []
    target_summaries: list[dict[str, Any]] = []

    for target_subject in subjects:
        target_trials = trial_table[trial_table["subject"].eq(target_subject)]
        high_trials = target_trials[target_trials["label"].eq("high")]
        low_trials = target_trials[target_trials["label"].eq("low")]
        if len(high_trials) < 2 or len(low_trials) < 2:
            raise ValueError(
                f"{target_subject} needs at least two High and two Low trials "
                "so calibration leaves both classes for test"
            )

        source_subjects = [subject for subject in subjects if subject != target_subject]
        source_trials = trial_table[trial_table["subject"].isin(source_subjects)]
        source_thresholds = (
            source_trials.groupby("subject")["valence"].median().to_dict()
        )
        scenario_audits: list[dict[str, Any]] = []

        for scenario, (high_row, low_row) in enumerate(
            product(
                high_trials.itertuples(index=False), low_trials.itertuples(index=False)
            ),
            start=1,
        ):
            calibration_groups = {high_row.split_group, low_row.split_group}
            target_threshold = float(np.median([high_row.valence, low_row.valence]))
            split = task_index.copy()
            split["partition"] = "pretrain"
            is_target = split["subject"].eq(target_subject)
            split.loc[is_target, "partition"] = "test"
            split.loc[
                is_target & split["split_group"].isin(calibration_groups), "partition"
            ] = "calibration"

            threshold_by_subject = {
                **source_thresholds,
                target_subject: target_threshold,
            }
            split["subject_relative_threshold"] = split["subject"].map(
                threshold_by_subject
            )
            split["label_subject_relative"] = _relative_labels(
                split["valence"], split["subject_relative_threshold"]
            )
            split["eligible_subject_relative"] = split["label_subject_relative"].ne(
                "ambiguous"
            )
            split.insert(0, "protocol", "personalization_high_low_calibration")
            split.insert(1, "target_subject", target_subject)
            split.insert(2, "scenario", scenario)
            split.insert(3, "task", task)
            split.insert(4, "calibration_high_trial", int(high_row.trial))
            split.insert(5, "calibration_low_trial", int(low_row.trial))
            rows.append(
                split[
                    [
                        "protocol",
                        "target_subject",
                        "scenario",
                        "task",
                        "calibration_high_trial",
                        "calibration_low_trial",
                        "partition",
                        "sample_id",
                        "subject",
                        "trial",
                        "split_group",
                        "label",
                        "valence",
                        "subject_relative_threshold",
                        "label_subject_relative",
                        "eligible_subject_relative",
                    ]
                ]
            )

            calibration = split[split["partition"].eq("calibration")]
            test = split[split["partition"].eq("test")]
            calibration_trial_labels = _trial_table(calibration)["label"].value_counts()
            test_trial_labels = _trial_table(test)["label"].value_counts()
            audit = {
                "scenario": scenario,
                "calibration_high_trial": int(high_row.trial),
                "calibration_low_trial": int(low_row.trial),
                "target_subject_relative_threshold": target_threshold,
                "leakage_free": _partitions_are_group_atomic(split),
                "calibration_has_one_trial_per_class": (
                    calibration_trial_labels.to_dict() == {"high": 1, "low": 1}
                ),
                "test_has_two_classes": len(test_trial_labels) >= 2,
                "partitions": {
                    name: _counts(partition)
                    for name, partition in split.groupby("partition")
                },
            }
            scenario_audits.append(audit)

        target_summaries.append(
            {
                "target_subject": target_subject,
                "pretrain_subjects": source_subjects,
                "high_trials": len(high_trials),
                "low_trials": len(low_trials),
                "calibration_scenarios": int(len(high_trials) * len(low_trials)),
                "scenarios": scenario_audits,
            }
        )

    splits = pd.concat(rows, ignore_index=True).sort_values(
        ["target_subject", "scenario", "partition", "subject", "trial", "sample_id"],
        ignore_index=True,
    )
    if (
        not splits.groupby(["target_subject", "scenario", "split_group"])["partition"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise ValueError("Trial leakage detected in personalization splits")

    scenario_audits = [
        scenario for target in target_summaries for scenario in target["scenarios"]
    ]
    summary = {
        "protocol": "personalization_high_low_calibration",
        "task": task,
        "subjects": subjects,
        "calibration_strategy": "all_high_trial_x_low_trial_pairs",
        "source_subject_relative_threshold_source": "all_pretraining_trials_per_subject",
        "target_subject_relative_threshold_source": "two_calibration_trials_only",
        "total_calibration_scenarios": len(scenario_audits),
        "all_scenarios_leakage_free": all(
            scenario["leakage_free"] for scenario in scenario_audits
        ),
        "all_calibrations_have_one_trial_per_class": all(
            scenario["calibration_has_one_trial_per_class"]
            for scenario in scenario_audits
        ),
        "all_tests_have_two_classes": all(
            scenario["test_has_two_classes"] for scenario in scenario_audits
        ),
        "target_summaries": target_summaries,
    }
    return splits, summary


def build_sensitivity_split(
    index: pd.DataFrame,
    training_subjects: list[str],
    sensitivity_subjects: list[str],
    task: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit on the primary cohort and reserve sensitivity-only subjects for test."""
    overlap = set(training_subjects) & set(sensitivity_subjects)
    if overlap:
        raise ValueError(
            f"Training and sensitivity subjects overlap: {sorted(overlap)}"
        )
    subjects = [*training_subjects, *sensitivity_subjects]
    task_index = _task_frame(index, subjects, task)
    split = task_index.copy()
    split["partition"] = np.where(
        split["subject"].isin(sensitivity_subjects), "sensitivity_test", "refit"
    )
    split.insert(0, "protocol", "single_class_sensitivity")
    split.insert(1, "task", task)
    output = split[
        [
            "protocol",
            "task",
            "partition",
            "sample_id",
            "subject",
            "trial",
            "split_group",
            "label",
            "valence",
        ]
    ].sort_values(["partition", "subject", "trial", "sample_id"], ignore_index=True)
    if not _partitions_are_group_atomic(output):
        raise ValueError("Trial leakage detected in sensitivity split")

    sensitivity = output[output["partition"].eq("sensitivity_test")]
    summary = {
        "protocol": "single_class_sensitivity",
        "task": task,
        "training_subjects": training_subjects,
        "sensitivity_subjects": sensitivity_subjects,
        "descriptive_only": sensitivity["label"].nunique() < 2,
        "subject_relative_labels_allowed": False,
        "leakage_free": True,
        "partitions": {
            name: _counts(partition) for name, partition in output.groupby("partition")
        },
    }
    return output, summary
