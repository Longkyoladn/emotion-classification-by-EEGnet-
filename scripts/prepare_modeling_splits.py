"""CLI entrypoint for all M2 modeling-index and split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_emotion.data import (
    build_modeling_index,
    build_nested_loso_splits,
    build_personalization_splits,
    build_sensitivity_split,
    build_subject_dependent_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modeling_m2_v1.0.2.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_artifacts(frame, summary: dict, protocol_config: dict) -> None:
    frame.to_csv(output_path(protocol_config["output"]), index=False)
    output_path(protocol_config["summary_output"]).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    )
    modeling_config = load_json(config_path)
    cohort_config = load_json(PROJECT_ROOT / modeling_config["cohort_config"])
    if modeling_config["dataset_snapshot"] != cohort_config["dataset"]["snapshot"]:
        raise ValueError("Modeling and cohort snapshots do not match")

    index = build_modeling_index(PROJECT_ROOT, cohort_config)
    index.to_csv(output_path(modeling_config["index_output"]), index=False)
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
    write_artifacts(subject_splits, subject_summary, subject_config)

    nested_splits, nested_summary = build_nested_loso_splits(
        index=index, subjects=subjects, task=task
    )
    write_artifacts(nested_splits, nested_summary, protocols["cross_subject"])

    personalization_splits, personalization_summary = build_personalization_splits(
        index=index, subjects=subjects, task=task
    )
    write_artifacts(
        personalization_splits,
        personalization_summary,
        protocols["personalization"],
    )

    sensitivity_subjects = cohort_config["subjects"][
        protocols["sensitivity"]["subject_role"]
    ]
    sensitivity_splits, sensitivity_summary = build_sensitivity_split(
        index=index,
        training_subjects=subjects,
        sensitivity_subjects=sensitivity_subjects,
        task=task,
    )
    write_artifacts(sensitivity_splits, sensitivity_summary, protocols["sensitivity"])

    print(
        json.dumps(
            {
                "indexed_primary_pass_windows": len(index),
                "fixed5_valence_eligible_windows": int(
                    index["eligible_valence_binary_fixed5"].sum()
                ),
                "subject_dependent": {
                    "subjects": len(subjects),
                    "repeats": subject_summary["repeats"],
                    "folds_per_repeat": 2,
                    "leakage_free": subject_summary["all_folds_leakage_free"],
                    "class_complete": subject_summary["all_folds_class_complete"],
                },
                "cross_subject": {
                    "outer_folds": nested_summary["outer_folds"],
                    "inner_folds_per_outer": nested_summary[
                        "inner_subject_folds_per_outer_fold"
                    ],
                    "leakage_free": nested_summary["all_folds_leakage_free"],
                    "class_complete": nested_summary["all_folds_class_complete"],
                },
                "personalization": {
                    "calibration_scenarios": personalization_summary[
                        "total_calibration_scenarios"
                    ],
                    "leakage_free": personalization_summary[
                        "all_scenarios_leakage_free"
                    ],
                    "test_class_complete": personalization_summary[
                        "all_tests_have_two_classes"
                    ],
                },
                "sensitivity": {
                    "subjects": sensitivity_subjects,
                    "descriptive_only": sensitivity_summary["descriptive_only"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
