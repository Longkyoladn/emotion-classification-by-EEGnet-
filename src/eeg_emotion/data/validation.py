"""Validate modeling index and split manifests before model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

INDEX_REQUIRED_COLUMNS = {
    "sample_id",
    "subject",
    "trial",
    "window_index",
    "start_sample",
    "stop_sample",
    "fif_path",
    "qc_status",
    "valence_label",
    "arousal_label",
    "quadrant_label",
    "split_group",
    "sample_weight",
}


def _all_true(checks: dict[str, bool]) -> bool:
    return all(bool(value) for value in checks.values())


def validate_modeling_index(
    index: pd.DataFrame,
    project_root: Path,
    verify_fif_headers: bool = True,
) -> dict[str, Any]:
    """Check schema, trial weights, sample ranges and FIF compatibility."""
    missing = INDEX_REQUIRED_COLUMNS - set(index.columns)
    if missing:
        raise ValueError(f"Modeling index is missing columns: {sorted(missing)}")
    trial_weight_sums = index.groupby("split_group")["sample_weight"].sum()
    checks = {
        "sample_id_unique": not index["sample_id"].duplicated().any(),
        "split_group_subject_trial_atomic": bool(
            index.groupby("split_group")[["subject", "trial"]]
            .nunique()
            .eq(1)
            .all()
            .all()
        ),
        "qc_status_primary_pass_only": set(index["qc_status"]) == {"pass"},
        "sample_ranges_positive": bool(
            (index["start_sample"] >= 0).all()
            and (index["stop_sample"] > index["start_sample"]).all()
        ),
        "sample_weight_positive": bool((index["sample_weight"] > 0).all()),
        "sample_weight_sums_to_one_per_trial": bool(
            np.allclose(trial_weight_sums.to_numpy(), 1.0)
        ),
        "labels_constant_within_trial": bool(
            index.groupby("split_group")
            [["valence_label", "arousal_label", "quadrant_label"]]
            .nunique()
            .eq(1)
            .all()
            .all()
        ),
    }

    fif_summary: dict[str, Any] = {"verified": False}
    if verify_fif_headers:
        paths = sorted(index["fif_path"].unique())
        missing_paths: list[str] = []
        out_of_range: list[str] = []
        sampling_rates: set[float] = set()
        channel_orders: set[tuple[str, ...]] = set()
        channel_counts: set[int] = set()
        for value in paths:
            path = Path(str(value))
            path = path if path.is_absolute() else project_root / path
            if not path.exists():
                missing_paths.append(str(path))
                continue
            raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
            sampling_rates.add(float(raw.info["sfreq"]))
            channel_orders.add(tuple(raw.ch_names))
            channel_counts.add(len(raw.ch_names))
            rows = index[index["fif_path"].eq(value)]
            if (rows["stop_sample"] > raw.n_times).any():
                out_of_range.append(str(path))
        checks.update(
            {
                "all_fif_paths_exist": not missing_paths,
                "all_windows_within_fif": not out_of_range,
                "sampling_rate_consistent": len(sampling_rates) == 1,
                "channel_count_consistent": len(channel_counts) == 1,
                "channel_order_consistent": len(channel_orders) == 1,
            }
        )
        fif_summary = {
            "verified": True,
            "files": len(paths),
            "missing_paths": missing_paths,
            "out_of_range_paths": out_of_range,
            "sampling_rates_hz": sorted(sampling_rates),
            "channel_counts": sorted(channel_counts),
        }

    return {
        "rows": len(index),
        "subjects": int(index["subject"].nunique()),
        "trials": int(index["split_group"].nunique()),
        "sample_range_convention": "zero_based_stop_exclusive_relative_to_trial_fif",
        "sample_weight_policy": "inverse_primary_pass_windows_per_trial",
        "checks": checks,
        "all_checks_pass": _all_true(checks),
        "fif": fif_summary,
    }


def _atomic(frame: pd.DataFrame, keys: list[str]) -> bool:
    return bool(
        frame.groupby([*keys, "split_group"], dropna=False)["partition"]
        .nunique()
        .eq(1)
        .all()
    )


def validate_split_artifacts(
    index: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame],
    primary_subjects: list[str],
    sensitivity_subjects: list[str],
) -> dict[str, Any]:
    """Validate protocol-specific isolation and class-completeness invariants."""
    required_names = {"subject_dependent", "cross_subject", "personalization", "sensitivity"}
    missing_names = required_names - set(artifacts)
    if missing_names:
        raise ValueError(f"Missing split artifacts: {sorted(missing_names)}")
    valid_ids = set(index["sample_id"])
    common_checks = {
        f"{name}_sample_ids_exist": set(frame["sample_id"]).issubset(valid_ids)
        for name, frame in artifacts.items()
    }

    subject_dependent = artifacts["subject_dependent"]
    sd_groups = ["subject", "repeat", "fold"]
    sd_partition_classes = subject_dependent.groupby(
        [*sd_groups, "partition"]
    )["label"].nunique()
    sd_checks = {
        "trial_atomic": _atomic(subject_dependent, sd_groups),
        "all_partitions_have_two_classes": bool(sd_partition_classes.ge(2).all()),
        "each_sample_test_once_per_repeat": bool(
            subject_dependent[subject_dependent["partition"].eq("test")]
            .groupby(["subject", "repeat", "sample_id"])
            .size()
            .eq(1)
            .all()
        ),
        "subject_relative_available": {
            "subject_relative_threshold",
            "label_subject_relative",
        }.issubset(subject_dependent.columns),
    }

    cross_subject = artifacts["cross_subject"]
    inner = cross_subject[cross_subject["stage"].eq("inner_selection")]
    outer = cross_subject[cross_subject["stage"].eq("outer_evaluation")]
    outer_tests = (
        outer[outer["partition"].eq("test")]
        .groupby("outer_fold")["subject"]
        .first()
        .to_dict()
    )
    test_absent = all(
        outer_tests[fold] not in set(frame["subject"])
        for fold, frame in inner.groupby("outer_fold")
    )
    cross_checks = {
        "trial_atomic": _atomic(
            cross_subject, ["outer_fold", "stage", "inner_fold"]
        ),
        "six_outer_folds": cross_subject["outer_fold"].nunique()
        == len(primary_subjects),
        "five_inner_subject_folds_per_outer": bool(
            inner.groupby("outer_fold")["inner_fold"]
            .nunique()
            .eq(len(primary_subjects) - 1)
            .all()
        ),
        "outer_test_absent_from_inner": test_absent,
        "outer_refit_uses_all_five_training_subjects": bool(
            outer[outer["partition"].eq("refit")]
            .groupby("outer_fold")["subject"]
            .nunique()
            .eq(len(primary_subjects) - 1)
            .all()
        ),
        "subject_relative_forbidden": "label_subject_relative"
        not in cross_subject.columns,
    }

    personalization = artifacts["personalization"]
    scenario_keys = ["target_subject", "scenario"]
    calibration = personalization[personalization["partition"].eq("calibration")]
    calibration_trials = calibration.drop_duplicates([*scenario_keys, "split_group"])
    test = personalization[personalization["partition"].eq("test")]
    personalization_checks = {
        "trial_atomic": _atomic(personalization, scenario_keys),
        "calibration_has_two_trials": bool(
            calibration_trials.groupby(scenario_keys).size().eq(2).all()
        ),
        "calibration_has_high_and_low": bool(
            calibration_trials.groupby(scenario_keys)["label"].nunique().eq(2).all()
        ),
        "test_has_high_and_low": bool(
            test.groupby(scenario_keys)["label"].nunique().eq(2).all()
        ),
        "subject_relative_available": {
            "subject_relative_threshold",
            "label_subject_relative",
        }.issubset(personalization.columns),
    }

    sensitivity = artifacts["sensitivity"]
    sensitivity_test = sensitivity[
        sensitivity["partition"].eq("sensitivity_test")
    ]
    sensitivity_checks = {
        "trial_atomic": _atomic(sensitivity, []),
        "sensitivity_subjects_test_only": set(sensitivity_test["subject"])
        == set(sensitivity_subjects),
        "primary_subjects_refit_only": set(
            sensitivity.loc[sensitivity["partition"].eq("refit"), "subject"]
        )
        == set(primary_subjects),
        "single_class_descriptive_test": sensitivity_test["label"].nunique() == 1,
        "subject_relative_forbidden": "label_subject_relative"
        not in sensitivity.columns,
    }

    protocol_reports = {
        "subject_dependent": {
            "rows": len(subject_dependent),
            "checks": sd_checks,
            "all_checks_pass": _all_true(sd_checks),
        },
        "cross_subject": {
            "rows": len(cross_subject),
            "checks": cross_checks,
            "all_checks_pass": _all_true(cross_checks),
        },
        "personalization": {
            "rows": len(personalization),
            "scenarios": int(
                personalization[scenario_keys].drop_duplicates().shape[0]
            ),
            "checks": personalization_checks,
            "all_checks_pass": _all_true(personalization_checks),
        },
        "sensitivity": {
            "rows": len(sensitivity),
            "checks": sensitivity_checks,
            "all_checks_pass": _all_true(sensitivity_checks),
        },
    }
    return {
        "common_checks": common_checks,
        "protocols": protocol_reports,
        "all_checks_pass": _all_true(common_checks)
        and all(report["all_checks_pass"] for report in protocol_reports.values()),
    }

