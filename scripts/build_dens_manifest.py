"""Build validated subject and trial manifests from DENS BIDS metadata.

This step deliberately preserves continuous self-report values and does not
derive fixed-threshold or subject-relative class labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


RATING_COLUMNS = [
    "valence",
    "arousal",
    "dominance",
    "liking",
    "familiarity",
    "relevance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifest"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/manifest_summary.json")
    )
    return parser.parse_args()


def normalize_behavior_stimulus(value: object) -> str:
    return re.sub(r"\.(mp4|m4v)$", "", str(value), flags=re.IGNORECASE).lower()


def normalize_event_stimulus(value: object) -> str:
    return re.sub(r"_\d+$", "", str(value)).lower()


def find_one(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if len(matches) == 1 else None


def relative(path: Path | None, root: Path) -> str | None:
    return path.relative_to(root).as_posix() if path else None


def build_manifest(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    participant_rows = pd.read_csv(dataset_root / "participants.tsv", sep="\t")
    duplicate_participant_rows = int(
        participant_rows.duplicated(subset=["participant_id"]).sum()
    )
    participants = participant_rows.drop_duplicates(
        subset=["participant_id"], keep="first"
    )
    participant_lookup = participants.set_index("participant_id").to_dict("index")
    subject_dirs = {path.name: path for path in dataset_root.glob("sub-*")}
    all_subject_ids = sorted(set(participants.participant_id) | set(subject_dirs))

    trial_rows: list[dict] = []
    subject_rows: list[dict] = []

    for subject_id in all_subject_ids:
        subject_dir = subject_dirs.get(subject_id)
        participant = participant_lookup.get(subject_id, {})

        if subject_dir is None:
            subject_rows.append(
                {
                    "subject": subject_id,
                    **participant,
                    "has_subject_directory": False,
                    "has_eeg_metadata": False,
                    "has_events": False,
                    "has_behavior": False,
                    "event_trials": 0,
                    "behavior_trials": 0,
                    "matched_trials": 0,
                    "event_only_trials": 0,
                    "behavior_only_trials": 0,
                    "raw_payload_materialized": False,
                }
            )
            continue

        eeg_dir = subject_dir / "eeg"
        behavior_dir = subject_dir / "beh"
        eeg_json_path = find_one(eeg_dir, "*eeg.json")
        events_path = find_one(eeg_dir, "*events.tsv")
        behavior_path = find_one(behavior_dir, "*beh.tsv")
        set_path = find_one(eeg_dir, "*.set")
        fdt_path = find_one(eeg_dir, "*.fdt")

        eeg_metadata = {}
        if eeg_json_path:
            eeg_metadata = json.loads(eeg_json_path.read_text(encoding="utf-8"))
        sampling_frequency = eeg_metadata.get("SamplingFrequency")
        recording_duration = eeg_metadata.get("RecordingDuration")

        events = (
            pd.read_csv(events_path, sep="\t")
            if events_path
            else pd.DataFrame(columns=["onset", "trial_type", "label"])
        )
        stimuli = events.loc[events.trial_type.eq("stm")].copy().reset_index(drop=True)
        behavior = (
            pd.read_csv(behavior_path, sep="\t")
            if behavior_path
            else pd.DataFrame(columns=["stimuliName", *RATING_COLUMNS])
        )

        behavior_queues: dict[str, deque[int]] = defaultdict(deque)
        for behavior_index, row in behavior.iterrows():
            behavior_queues[normalize_behavior_stimulus(row.stimuliName)].append(
                behavior_index
            )

        matched_behavior_indices: set[int] = set()
        subject_trial_rows: list[dict] = []
        for event_index, stimulus in stimuli.iterrows():
            stimulus_key = normalize_event_stimulus(stimulus.label)
            behavior_index = (
                behavior_queues[stimulus_key].popleft()
                if behavior_queues[stimulus_key]
                else None
            )
            behavior_row = (
                behavior.loc[behavior_index] if behavior_index is not None else None
            )
            if behavior_index is not None:
                matched_behavior_indices.add(behavior_index)

            onset_samples = float(stimulus.onset)
            rating_events = events.loc[
                events.trial_type.eq("vlnc") & events.onset.gt(onset_samples)
            ]
            end_samples = (
                float(rating_events.iloc[0].onset) if not rating_events.empty else None
            )
            onset_seconds = (
                onset_samples / sampling_frequency if sampling_frequency else None
            )
            end_seconds = (
                end_samples / sampling_frequency
                if sampling_frequency and end_samples is not None
                else None
            )

            issues: list[str] = []
            if sampling_frequency is None:
                issues.append("missing_sampling_frequency")
            if set_path is None:
                issues.append("missing_eeg_set")
            if fdt_path is None:
                issues.append("missing_eeg_fdt")
            if behavior_row is None:
                issues.append("missing_behavior_row")
            if end_samples is None:
                issues.append("missing_valence_event")
            if end_seconds is not None and end_seconds <= onset_seconds:
                issues.append("non_positive_trial_duration")
            if (
                recording_duration is not None
                and end_seconds is not None
                and end_seconds > float(recording_duration) + 1.0
            ):
                issues.append("trial_exceeds_recording")

            row = {
                "subject": subject_id,
                "event_trial_index": event_index + 1,
                "behavior_trial_index": (
                    behavior_index + 1 if behavior_index is not None else None
                ),
                "stimulus": (
                    behavior_row.stimuliName
                    if behavior_row is not None
                    else stimulus_key
                ),
                "stimulus_key": stimulus_key,
                "event_label": stimulus.label,
                "onset_samples": onset_samples,
                "end_samples": end_samples,
                "onset_seconds": onset_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": (
                    end_seconds - onset_seconds
                    if end_seconds is not None and onset_seconds is not None
                    else None
                ),
                "sampling_frequency_hz": sampling_frequency,
                "recording_duration_seconds": recording_duration,
                "match_status": "matched" if behavior_row is not None else "event_only",
                "validation_issues": ";".join(issues),
                "eeg_set_path": relative(set_path, dataset_root),
                "eeg_fdt_path": relative(fdt_path, dataset_root),
                "events_path": relative(events_path, dataset_root),
                "behavior_path": relative(behavior_path, dataset_root),
            }
            for column in RATING_COLUMNS:
                row[column] = (
                    behavior_row.get(column) if behavior_row is not None else None
                )
            for column in ["emotionCateg", "Quadrant", "FivePointScale", "MouseClick"]:
                row[column] = (
                    behavior_row.get(column) if behavior_row is not None else None
                )
            subject_trial_rows.append(row)

        for behavior_index, behavior_row in behavior.iterrows():
            if behavior_index in matched_behavior_indices:
                continue
            row = {
                "subject": subject_id,
                "event_trial_index": None,
                "behavior_trial_index": behavior_index + 1,
                "stimulus": behavior_row.stimuliName,
                "stimulus_key": normalize_behavior_stimulus(behavior_row.stimuliName),
                "event_label": None,
                "onset_samples": None,
                "end_samples": None,
                "onset_seconds": None,
                "end_seconds": None,
                "duration_seconds": None,
                "sampling_frequency_hz": sampling_frequency,
                "recording_duration_seconds": recording_duration,
                "match_status": "behavior_only",
                "validation_issues": "missing_stimulus_event",
                "eeg_set_path": relative(set_path, dataset_root),
                "eeg_fdt_path": relative(fdt_path, dataset_root),
                "events_path": relative(events_path, dataset_root),
                "behavior_path": relative(behavior_path, dataset_root),
            }
            for column in RATING_COLUMNS:
                row[column] = behavior_row.get(column)
            for column in ["emotionCateg", "Quadrant", "FivePointScale", "MouseClick"]:
                row[column] = behavior_row.get(column)
            subject_trial_rows.append(row)

        trial_rows.extend(subject_trial_rows)
        match_counts = pd.Series(
            [row["match_status"] for row in subject_trial_rows]
        ).value_counts()
        subject_rows.append(
            {
                "subject": subject_id,
                **participant,
                "has_subject_directory": True,
                "has_eeg_metadata": eeg_json_path is not None,
                "has_events": events_path is not None,
                "has_behavior": behavior_path is not None,
                "event_trials": len(stimuli),
                "behavior_trials": len(behavior),
                "matched_trials": int(match_counts.get("matched", 0)),
                "event_only_trials": int(match_counts.get("event_only", 0)),
                "behavior_only_trials": int(match_counts.get("behavior_only", 0)),
                "raw_payload_materialized": bool(
                    set_path
                    and fdt_path
                    and set_path.stat().st_size > 1_000
                    and fdt_path.stat().st_size > 1_000
                ),
            }
        )

    trials = pd.DataFrame(trial_rows).sort_values(
        ["subject", "event_trial_index", "behavior_trial_index"], na_position="last"
    )
    subjects = pd.DataFrame(subject_rows).sort_values("subject")
    issue_counts = (
        trials.validation_issues.replace("", pd.NA)
        .dropna()
        .str.split(";")
        .explode()
        .value_counts()
        .to_dict()
    )
    summary = {
        "participant_rows_in_tsv": int(len(participant_rows)),
        "unique_subjects_in_participants_tsv": int(len(participants)),
        "duplicate_participant_rows": duplicate_participant_rows,
        "subject_directories": int(len(subject_dirs)),
        "subjects_with_behavior": int(subjects.has_behavior.sum()),
        "subjects_with_events": int(subjects.has_events.sum()),
        "subjects_with_materialized_raw_payload": int(
            subjects.raw_payload_materialized.sum()
        ),
        "manifest_rows": int(len(trials)),
        "matched_trials": int(trials.match_status.eq("matched").sum()),
        "event_only_trials": int(trials.match_status.eq("event_only").sum()),
        "behavior_only_trials": int(trials.match_status.eq("behavior_only").sum()),
        "fully_valid_labeled_trials": int(
            (
                trials.match_status.eq("matched")
                & trials.validation_issues.eq("")
                & trials.duration_seconds.notna()
                & trials.eeg_set_path.notna()
                & trials.eeg_fdt_path.notna()
                & trials[RATING_COLUMNS].notna().all(axis=1)
            ).sum()
        ),
        "validation_issue_counts": issue_counts,
        "label_policy": "Continuous ratings preserved; no binary labels derived.",
    }
    return trials, subjects, summary


def main() -> None:
    args = parse_args()
    trials, subjects, summary = build_manifest(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "dens_trials.csv", index=False)
    subjects.to_csv(args.output_dir / "dens_subjects.csv", index=False)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
