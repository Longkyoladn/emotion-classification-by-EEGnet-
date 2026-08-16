"""Build an auditable window-level modeling index from P10 curation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TASK_COLUMNS = {
    "valence_binary_fixed5": (
        "valence_label_fixed5",
        "include_primary_valence_binary",
    ),
    "arousal_binary_fixed5": (
        "arousal_label_fixed5",
        "include_primary_arousal_binary",
    ),
    "quadrant_fixed5": (
        "quadrant_label_fixed5",
        "include_primary_quadrant",
    ),
}

REQUIRED_COLUMNS = {
    "subject",
    "trial",
    "window_index",
    "start_seconds",
    "end_seconds",
    "samples",
    "status",
    "source_path",
    "include_primary",
    "split_group",
    "valence",
    "arousal",
    "valence_label_fixed5",
    "arousal_label_fixed5",
    "quadrant_label_fixed5",
    "include_primary_valence_binary",
    "include_primary_arousal_binary",
    "include_primary_quadrant",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    )


def build_modeling_index(
    project_root: Path,
    cohort_config: dict[str, Any],
) -> pd.DataFrame:
    """Combine primary pass windows and attach immutable cohort roles."""
    qc_root = project_root / cohort_config["paths"]["qc_root"]
    locked = cohort_config["subjects"]["integrity_pass_locked"]
    role_by_subject: dict[str, str] = {}
    for role in ("modeling_primary", "sensitivity_only", "qc_only"):
        for subject in cohort_config["subjects"][role]:
            role_by_subject[subject] = role

    frames: list[pd.DataFrame] = []
    for subject in locked:
        path = qc_root / subject / "p10_curated_windows.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing P10 window curation: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if not frame.empty and set(frame["subject"].unique()) != {subject}:
            raise ValueError(f"Subject mismatch in {path}")
        frames.append(frame)

    windows = pd.concat(frames, ignore_index=True)
    primary = _as_bool(windows["include_primary"]) & windows["status"].eq("pass")
    index = windows.loc[primary].copy()
    index.insert(1, "cohort_role", index["subject"].map(role_by_subject))
    index.insert(
        0,
        "sample_id",
        index.apply(
            lambda row: (
                f"{row['subject']}_trial-{int(row['trial']):02d}"
                f"_window-{int(row['window_index']):03d}"
            ),
            axis=1,
        ),
    )
    duration_seconds = index["end_seconds"] - index["start_seconds"]
    if (duration_seconds <= 0).any():
        raise ValueError("Window duration must be positive")
    sampling_rates = index["samples"] / duration_seconds
    rounded_sampling_rates = sampling_rates.round().astype(int)
    if not (sampling_rates - rounded_sampling_rates).abs().lt(1e-9).all():
        raise ValueError("Window metadata does not imply an integer sampling rate")
    index["sampling_rate_hz"] = rounded_sampling_rates
    index["start_sample"] = (
        index["start_seconds"] * index["sampling_rate_hz"]
    ).round().astype(int)
    index["stop_sample"] = index["start_sample"] + index["samples"].astype(int)
    index["fif_path"] = index["source_path"].map(lambda value: Path(value).as_posix())
    index["qc_status"] = index["status"]
    index["valence_label"] = index["valence_label_fixed5"]
    index["arousal_label"] = index["arousal_label_fixed5"]
    index["quadrant_label"] = index["quadrant_label_fixed5"]
    windows_per_trial = index.groupby("split_group")["sample_id"].transform("size")
    index["sample_weight"] = 1.0 / windows_per_trial

    for task, (label_column, include_column) in TASK_COLUMNS.items():
        index[f"eligible_{task}"] = _as_bool(index[include_column])
        index[f"label_{task}"] = index[label_column]

    if index["sample_id"].duplicated().any():
        duplicates = index.loc[index["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs: {duplicates[:5]}")
    if index["cohort_role"].isna().any():
        raise ValueError("At least one indexed subject has no cohort role")
    if not (index.groupby(["subject", "trial"])["split_group"].nunique().eq(1).all()):
        raise ValueError("A trial maps to more than one split_group")

    source_paths = index["fif_path"].map(lambda value: project_root / str(value))
    missing_sources = [str(path) for path in source_paths.unique() if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Missing processed trial files: {missing_sources[:5]}")

    columns = [
        "sample_id",
        "subject",
        "cohort_role",
        "trial",
        "split_group",
        "window_index",
        "start_sample",
        "stop_sample",
        "fif_path",
        "qc_status",
        "valence_label",
        "arousal_label",
        "quadrant_label",
        "sample_weight",
        "start_seconds",
        "end_seconds",
        "samples",
        "sampling_rate_hz",
        "valence",
        "arousal",
        "eligible_valence_binary_fixed5",
        "label_valence_binary_fixed5",
        "eligible_arousal_binary_fixed5",
        "label_arousal_binary_fixed5",
        "eligible_quadrant_fixed5",
        "label_quadrant_fixed5",
    ]
    return index[columns].sort_values(
        ["subject", "trial", "window_index"], ignore_index=True
    )
