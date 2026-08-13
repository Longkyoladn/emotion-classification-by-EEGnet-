"""Aggregate integrity, preprocessing, curation, and fixed-5 label outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REPORTS_DIR = Path("reports")
QC_DIR = REPORTS_DIR / "qc"
INTEGRITY_REPORT = REPORTS_DIR / "dens_eeglab_integrity.csv"
COHORT_REPORT = REPORTS_DIR / "dens_cohort_curation.csv"
LABEL_REPORT = REPORTS_DIR / "dens_cohort_label_distribution.csv"
SUMMARY_REPORT = REPORTS_DIR / "dens_cohort_summary.json"

LABEL_TASKS = {
    "valence_binary": "valence_label_fixed5",
    "arousal_binary": "arousal_label_fixed5",
    "quadrant": "quadrant_label_fixed5",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def selected_ica_components(subject: str, qc_dir: Path) -> list[int]:
    """Return approved ICA components, preserving the subject-003 fallback."""
    decision_path = qc_dir / "p6b_ica_decision.json"
    if decision_path.exists():
        return read_json(decision_path)["selected_components"]

    return [11] if subject == "sub-mit003" else []


def build_subject_row(
    subject: str,
    trials: pd.DataFrame,
    windows: pd.DataFrame,
    bad_channel_summary: dict[str, Any],
    ica_summary: dict[str, Any],
    selected_components: list[int],
) -> dict[str, Any]:
    """Build one subject-level cohort summary row."""
    excluded_trials = trials.loc[
        trials["exclude_primary"],
        "trial",
    ].astype(int)

    primary_windows = int(windows["include_primary"].sum())
    pass_review_windows = int(windows["include_pass_review_ablation"].sum())

    return {
        "subject": subject,
        "matched_trials": len(trials),
        "primary_trials": int((~trials["exclude_primary"]).sum()),
        "excluded_trials": ";".join(map(str, excluded_trials)),
        "bad_channels": ";".join(bad_channel_summary["bad"]),
        "ica_components": ica_summary["components"],
        "ica_removed": ";".join(map(str, selected_components)),
        "all_windows": len(windows),
        "primary_pass_windows": primary_windows,
        "review_ablation_windows": pass_review_windows - primary_windows,
        "reject_windows": int((windows["status"] == "reject").sum()),
        "retained_for_analysis": int(windows["retain_for_analysis"].sum()),
    }


def build_label_rows(
    subject: str,
    primary_windows: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build per-subject fixed-5 label distribution rows."""
    rows = []

    for task, column in LABEL_TASKS.items():
        label_counts = primary_windows[column].value_counts()
        for label, count in label_counts.items():
            rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "label": label,
                    "primary_windows": int(count),
                }
            )

    return rows


def aggregate_primary_labels(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Aggregate primary-window label totals across the curated cohort."""
    totals = {}

    for task, task_rows in labels.groupby("task"):
        counts = task_rows.groupby("label")["primary_windows"].sum().to_dict()
        totals[str(task)] = {
            str(label): int(count)
            for label, count in counts.items()
        }

    return totals


def build_summary(
    integrity: pd.DataFrame,
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    """Build the machine-readable cohort summary."""
    integrity_pass = integrity["integrity_status"].eq("pass")

    return {
        "recordings_in_integrity_audit": int(len(integrity)),
        "integrity_pass_subjects": integrity.loc[
            integrity_pass,
            "subject",
        ].tolist(),
        "integrity_fail_subjects": integrity.loc[
            ~integrity_pass,
            "subject",
        ].tolist(),
        "fully_curated_subjects": cohort["subject"].tolist(),
        "fully_curated_count": int(len(cohort)),
        "cohort_totals": {
            "matched_trials": int(cohort["matched_trials"].sum()),
            "primary_trials": int(cohort["primary_trials"].sum()),
            "all_windows": int(cohort["all_windows"].sum()),
            "primary_pass_windows": int(cohort["primary_pass_windows"].sum()),
            "review_ablation_windows": int(
                cohort["review_ablation_windows"].sum()
            ),
            "reject_windows": int(cohort["reject_windows"].sum()),
        },
        "primary_label_totals": aggregate_primary_labels(labels),
        "raw_or_cleaned_eeg_deleted": False,
        "integrity_policy": (
            "SET/FDT size mismatch subjects are excluded; no truncation, "
            "zero-padding, or fabricated samples"
        ),
    }


def main() -> None:
    integrity = pd.read_csv(INTEGRITY_REPORT)
    subject_rows = []
    label_rows = []

    summary_paths = sorted(QC_DIR.glob("sub-*/p10_curation_summary.json"))
    for summary_path in summary_paths:
        qc_dir = summary_path.parent
        subject = qc_dir.name

        trials = pd.read_csv(qc_dir / "p10_curated_trials.csv")
        windows = pd.read_csv(qc_dir / "p10_curated_windows.csv")
        bad_channel_summary = read_json(qc_dir / "p4_bad_channel_summary.json")
        ica_summary = read_json(qc_dir / "p6_ica_summary.json")
        selected_components = selected_ica_components(subject, qc_dir)

        subject_rows.append(
            build_subject_row(
                subject=subject,
                trials=trials,
                windows=windows,
                bad_channel_summary=bad_channel_summary,
                ica_summary=ica_summary,
                selected_components=selected_components,
            )
        )

        primary_windows = windows.loc[windows["include_primary"]]
        label_rows.extend(build_label_rows(subject, primary_windows))

    cohort = pd.DataFrame(subject_rows)
    labels = pd.DataFrame(label_rows)

    cohort.to_csv(COHORT_REPORT, index=False)
    labels.to_csv(LABEL_REPORT, index=False)

    summary = build_summary(integrity, cohort, labels)
    summary_json = json.dumps(summary, indent=2) + "\n"
    SUMMARY_REPORT.write_text(summary_json, encoding="utf-8")
    print(summary_json, end="")


if __name__ == "__main__":
    main()
