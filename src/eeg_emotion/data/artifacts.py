"""Orchestrate reproducible modeling index, split and validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .modeling_index import build_modeling_index
from .splits import (
    build_nested_loso_splits,
    build_personalization_splits,
    build_sensitivity_split,
    build_subject_dependent_splits,
)
from .validation import validate_modeling_index, validate_split_artifacts


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_modeling_configuration(
    project_root: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = config_path if config_path.is_absolute() else project_root / config_path
    modeling_config = _read_json(path)
    cohort_config = _read_json(project_root / modeling_config["cohort_config"])
    if modeling_config["dataset_snapshot"] != cohort_config["dataset"]["snapshot"]:
        raise ValueError("Modeling and cohort snapshots do not match")
    return modeling_config, cohort_config


def build_index_artifact(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    modeling_config, cohort_config = load_modeling_configuration(
        project_root, config_path
    )
    index = build_modeling_index(project_root, cohort_config)
    output = project_root / modeling_config["index_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(output, index=False)
    return {
        "output": output.as_posix(),
        "windows": len(index),
        "subjects": int(index["subject"].nunique()),
        "trials": int(index["split_group"].nunique()),
        "fixed5_valence_eligible_windows": int(
            index["eligible_valence_binary_fixed5"].sum()
        ),
        "sample_weight_sums_to_one_per_trial": bool(
            index.groupby("split_group")["sample_weight"]
            .sum()
            .sub(1.0)
            .abs()
            .lt(1e-12)
            .all()
        ),
    }


def _write_split(
    project_root: Path,
    frame: pd.DataFrame,
    summary: dict[str, Any],
    protocol_config: dict[str, Any],
) -> None:
    output = project_root / protocol_config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    _write_json(project_root / protocol_config["summary_output"], summary)


def create_split_artifacts(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    modeling_config, cohort_config = load_modeling_configuration(
        project_root, config_path
    )
    index_path = project_root / modeling_config["index_output"]
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing modeling index {index_path}; run build_modeling_index.py first"
        )
    index = pd.read_csv(index_path)
    task = modeling_config["primary_task"]
    subjects = cohort_config["subjects"][modeling_config["subject_role"]]
    protocols = modeling_config["protocols"]

    subject_config = protocols["subject_dependent"]
    subject_splits, subject_summary = build_subject_dependent_splits(
        index=index,
        subjects=subjects,
        task=task,
        repeats=subject_config["repeats"],
        seed=subject_config["seed"],
    )
    _write_split(project_root, subject_splits, subject_summary, subject_config)

    nested_splits, nested_summary = build_nested_loso_splits(index, subjects, task)
    _write_split(
        project_root, nested_splits, nested_summary, protocols["cross_subject"]
    )

    personalization_splits, personalization_summary = build_personalization_splits(
        index, subjects, task
    )
    _write_split(
        project_root,
        personalization_splits,
        personalization_summary,
        protocols["personalization"],
    )

    sensitivity_subjects = cohort_config["subjects"][
        protocols["sensitivity"]["subject_role"]
    ]
    sensitivity_splits, sensitivity_summary = build_sensitivity_split(
        index, subjects, sensitivity_subjects, task
    )
    _write_split(
        project_root,
        sensitivity_splits,
        sensitivity_summary,
        protocols["sensitivity"],
    )

    return {
        "subject_dependent": {
            "rows": len(subject_splits),
            "repeats": subject_summary["repeats"],
            "all_checks_pass": subject_summary["all_folds_leakage_free"]
            and subject_summary["all_folds_class_complete"],
        },
        "cross_subject": {
            "rows": len(nested_splits),
            "outer_folds": nested_summary["outer_folds"],
            "inner_folds_per_outer": nested_summary[
                "inner_subject_folds_per_outer_fold"
            ],
            "all_checks_pass": nested_summary["all_folds_leakage_free"]
            and nested_summary["all_folds_class_complete"],
        },
        "personalization": {
            "rows": len(personalization_splits),
            "scenarios": personalization_summary["total_calibration_scenarios"],
            "all_checks_pass": personalization_summary[
                "all_scenarios_leakage_free"
            ]
            and personalization_summary["all_tests_have_two_classes"],
        },
        "sensitivity": {
            "rows": len(sensitivity_splits),
            "descriptive_only": sensitivity_summary["descriptive_only"],
            "all_checks_pass": sensitivity_summary["leakage_free"],
        },
    }


def validate_artifacts(
    project_root: Path,
    config_path: Path,
    verify_fif_headers: bool = True,
) -> dict[str, Any]:
    modeling_config, cohort_config = load_modeling_configuration(
        project_root, config_path
    )
    index = pd.read_csv(project_root / modeling_config["index_output"])
    protocols = modeling_config["protocols"]
    artifacts = {
        name: pd.read_csv(project_root / protocols[name]["output"])
        for name in (
            "subject_dependent",
            "cross_subject",
            "personalization",
            "sensitivity",
        )
    }
    primary_subjects = cohort_config["subjects"][modeling_config["subject_role"]]
    sensitivity_subjects = cohort_config["subjects"][
        protocols["sensitivity"]["subject_role"]
    ]
    index_report = validate_modeling_index(
        index, project_root, verify_fif_headers=verify_fif_headers
    )
    split_report = validate_split_artifacts(
        index, artifacts, primary_subjects, sensitivity_subjects
    )
    report = {
        "dataset_snapshot": modeling_config["dataset_snapshot"],
        "modeling_index": index_report,
        "splits": split_report,
        "normalization_policy": modeling_config["normalization"],
        "evaluation_policy": modeling_config["evaluation"],
        "all_checks_pass": index_report["all_checks_pass"]
        and split_report["all_checks_pass"],
    }
    _write_json(project_root / modeling_config["validation_output"], report)
    if not report["all_checks_pass"]:
        raise ValueError("Modeling artifact validation failed")
    return report

