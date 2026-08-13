"""Audit whether curated DENS v1.0.2 outputs are ready for modeling.

This audit never edits EEG or P9/P10 curation outputs. It verifies the locked
eight-subject cohort and writes a subject-level CSV plus a machine-readable
summary for the next split-design checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LOCKED_V102_SUBJECTS = (
    "sub-mit003",
    "sub-mit004",
    "sub-mit061",
    "sub-mit076",
    "sub-mit117",
    "sub-mit121",
    "sub-mit123",
    "sub-mitb2017007",
)

TASKS = {
    "valence": (
        "valence_label_fixed5",
        "valence_binary_eligible_fixed5",
        "include_primary_valence_binary",
    ),
    "arousal": (
        "arousal_label_fixed5",
        "arousal_binary_eligible_fixed5",
        "include_primary_arousal_binary",
    ),
    "quadrant": (
        "quadrant_label_fixed5",
        "quadrant_label_eligible_fixed5",
        "include_primary_quadrant",
    ),
}


def label_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in series.value_counts().sort_index().items()
    }


def encode_counts(counts: dict[str, int]) -> str:
    return ";".join(f"{label}:{count}" for label, count in counts.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-root", type=Path, default=Path("reports/qc"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/modeling_readiness_v1.0.2.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/modeling_readiness_v1.0.2.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    failed_checks: list[dict[str, str]] = []
    overall_task_trials = {task: {} for task in TASKS}
    overall_task_windows = {task: {} for task in TASKS}

    for subject in LOCKED_V102_SUBJECTS:
        qc_dir = args.qc_root / subject
        trials = pd.read_csv(qc_dir / "p10_curated_trials.csv")
        windows = pd.read_csv(qc_dir / "p10_curated_windows.csv")
        primary_trials = trials.loc[~trials["exclude_primary"].astype(bool)]
        primary_windows = windows.loc[windows["include_primary"].astype(bool)]

        checks = {
            "primary_is_pass": bool((primary_windows["status"] == "pass").all()),
            "primary_not_excluded": bool(
                (~primary_windows["exclude_primary"].astype(bool)).all()
            ),
            "unique_window_key": not windows.duplicated(
                ["subject", "trial", "window_index"]
            ).any(),
            "trial_atomic_split_group": bool(
                (
                    windows.groupby(["subject", "trial"])["split_group"].nunique() == 1
                ).all()
            ),
            "window_counts_match_trials": all(
                int(row["pass"] + row["review"] + row["reject"])
                == int(row["total_windows"])
                for _, row in trials.iterrows()
            ),
        }

        missing_fif = []
        for trial in trials["trial"].astype(int):
            fif = (
                args.processed_root
                / subject
                / "trials"
                / f"{subject}_trial-{trial:02d}_eeg.fif"
            )
            if not fif.exists():
                missing_fif.append(str(fif))
        checks["all_trial_fif_present"] = not missing_fif

        for check, passed in checks.items():
            if not passed:
                failed_checks.append({"subject": subject, "check": check})

        row: dict[str, object] = {
            "subject": subject,
            "raw_snapshot": "ds003751_v1.0.2",
            "matched_trials": int(len(trials)),
            "primary_trials": int(len(primary_trials)),
            "primary_pass_windows": int(len(primary_windows)),
            "review_windows_retained_trial": int(
                (
                    (windows["status"] == "review")
                    & ~windows["exclude_primary"].astype(bool)
                ).sum()
            ),
            "reject_windows_all": int((windows["status"] == "reject").sum()),
            "missing_trial_fif": int(len(missing_fif)),
            "all_consistency_checks_pass": bool(all(checks.values())),
        }

        for task, (label_col, eligible_col, include_col) in TASKS.items():
            task_trials = primary_trials.loc[primary_trials[eligible_col].astype(bool)]
            task_windows = windows.loc[windows[include_col].astype(bool)]
            trial_counts = label_counts(task_trials[label_col])
            window_counts = label_counts(task_windows[label_col])
            row[f"{task}_eligible_trials"] = int(len(task_trials))
            row[f"{task}_trial_classes"] = encode_counts(trial_counts)
            row[f"{task}_eligible_windows"] = int(len(task_windows))
            row[f"{task}_window_classes"] = encode_counts(window_counts)
            row[f"{task}_has_two_or_more_classes"] = len(trial_counts) >= 2
            for label, count in trial_counts.items():
                overall_task_trials[task][label] = (
                    overall_task_trials[task].get(label, 0) + count
                )
            for label, count in window_counts.items():
                overall_task_windows[task][label] = (
                    overall_task_windows[task].get(label, 0) + count
                )

        rows.append(row)

    table = pd.DataFrame(rows)
    totals = {
        "raw_integrity_subjects": len(LOCKED_V102_SUBJECTS),
        "subjects_with_primary_data": int((table["primary_trials"] > 0).sum()),
        "subjects_with_at_least_two_primary_trials": int(
            (table["primary_trials"] >= 2).sum()
        ),
        "matched_trials": int(table["matched_trials"].sum()),
        "primary_trials": int(table["primary_trials"].sum()),
        "primary_pass_windows": int(table["primary_pass_windows"].sum()),
        "review_windows_retained_trial": int(
            table["review_windows_retained_trial"].sum()
        ),
        "reject_windows_all": int(table["reject_windows_all"].sum()),
    }
    task_summary = {}
    for task in TASKS:
        task_summary[task] = {
            "eligible_trials": int(table[f"{task}_eligible_trials"].sum()),
            "eligible_windows": int(table[f"{task}_eligible_windows"].sum()),
            "trial_labels": overall_task_trials[task],
            "window_labels": overall_task_windows[task],
            "subjects_with_data": int((table[f"{task}_eligible_trials"] > 0).sum()),
            "subjects_with_two_or_more_classes": int(
                table[f"{task}_has_two_or_more_classes"].sum()
            ),
        }

    summary = {
        "dataset": "DENS / OpenNeuro ds003751",
        "locked_snapshot": "1.0.2",
        "locked_subjects": list(LOCKED_V102_SUBJECTS),
        "v1.0.6_used": False,
        "all_consistency_checks_pass": not failed_checks,
        "failed_checks": failed_checks,
        "totals": totals,
        "tasks": task_summary,
        "modeling_warning": (
            "Eight subjects have valid raw recordings, but only seven contribute "
            "primary data; sub-mit117 contributes one primary trial and "
            "sub-mitb2017007 contributes none."
        ),
    }

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.csv_output, index=False)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
