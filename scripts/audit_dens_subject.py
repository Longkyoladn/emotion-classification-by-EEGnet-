"""Audit one locally cached DENS subject without loading the full dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subject", default="sub-mit003")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def find_one(folder: Path, pattern: str) -> Path:
    matches = list(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} in {folder}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    subject_dir = args.dataset_root / args.subject
    set_path = find_one(subject_dir / "eeg", "*.set")
    events_path = find_one(subject_dir / "eeg", "*events.tsv")
    behavior_path = find_one(subject_dir / "beh", "*beh.tsv")

    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    events = pd.read_csv(events_path, sep="\t")
    behavior = pd.read_csv(behavior_path, sep="\t")

    # DENS stores event onsets/durations as sample indices, although BIDS expects
    # seconds.
    events["onset_seconds"] = events["onset"] / raw.info["sfreq"]
    events["duration_seconds"] = events["duration"] / raw.info["sfreq"]

    report = {
        "subject": args.subject,
        "sampling_frequency_hz": raw.info["sfreq"],
        "recording_duration_seconds": raw.times[-1],
        "samples": raw.n_times,
        "channels": raw.info["nchan"],
        "channel_names": raw.ch_names,
        "mne_channel_types_before_correction": dict(Counter(raw.get_channel_types())),
        "behavior_trials": len(behavior),
        "stimulus_events": int((events["trial_type"] == "stm").sum()),
        "event_types": events["trial_type"].value_counts().to_dict(),
        "missing_ratings": behavior[["valence", "arousal", "dominance", "liking"]]
        .isna()
        .sum()
        .to_dict(),
        "rating_summary": behavior[["valence", "arousal", "dominance", "liking"]]
        .describe()
        .to_dict(),
        "notes": [
            (
                "Treat E1-E128 as EEG, E129 as misc/reference, ECG as ECG, and "
                "EMG/EMG_2 as EMG."
            ),
            (
                "Do not use the Quadrant column as ground truth; derive labels "
                "from continuous ratings."
            ),
            (
                "Convert events onset and duration from samples to seconds using "
                "the recording sampling frequency."
            ),
        ],
    }

    rendered = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        default=lambda value: (
            value.item() if isinstance(value, np.generic) else str(value)
        ),
    )
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
