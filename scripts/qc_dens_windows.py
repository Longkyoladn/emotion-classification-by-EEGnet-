"""Checkpoint P9: segment cleaned trials and audit quality per window.

This checkpoint is deliberately non-destructive: it records QC decisions but
does not delete windows, derive emotion labels, or create train/test splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


WINDOW_SECONDS = 4.0
OVERLAP = 0.5


def robust_variance_outliers(x_uv: np.ndarray) -> int:
    """Count channels with an unusually high or low standard deviation."""
    std = np.std(x_uv, axis=1)
    log_std = np.log10(std + 1e-12)
    median = np.median(log_std)
    mad = np.median(np.abs(log_std - median))
    robust_z = (log_std - median) / max(1.4826 * mad, 1e-12)
    return int(np.sum(np.abs(robust_z) > 3.5))


def classify_window(x_uv: np.ndarray) -> dict[str, object]:
    finite = bool(np.isfinite(x_uv).all())
    max_abs = float(np.nanmax(np.abs(x_uv))) if finite else float("nan")
    peak_to_peak = float(np.nanmax(np.ptp(x_uv, axis=1))) if finite else float("nan")
    fraction_over_200 = float(np.mean(np.abs(x_uv) > 200)) if finite else float("nan")
    flat_channels = int(np.sum(np.std(x_uv, axis=1) < 0.5)) if finite else 0
    variance_outliers = robust_variance_outliers(x_uv) if finite else 0

    reject_reasons: list[str] = []
    review_reasons: list[str] = []
    if not finite:
        reject_reasons.append("nonfinite")
    if finite and (max_abs > 1000 or fraction_over_200 > 0.01):
        reject_reasons.append("extreme_amplitude")
    if flat_channels:
        review_reasons.append("flat_channels")
    if variance_outliers > 5:
        review_reasons.append("variance_outliers")
    if finite and (max_abs > 500 or fraction_over_200 > 0.001):
        review_reasons.append("amplitude_warning")

    status = "reject" if reject_reasons else ("review" if review_reasons else "pass")
    return {
        "status": status,
        "issues": ";".join(reject_reasons + review_reasons),
        "max_abs_uv": max_abs,
        "max_peak_to_peak_uv": peak_to_peak,
        "over_200uv_fraction": fraction_over_200,
        "flat_channels": flat_channels,
        "variance_outlier_channels": variance_outliers,
    }


def plot_timeline(qc: pd.DataFrame, output: Path) -> None:
    colors = {"pass": "#2ca02c", "review": "#ffbf00", "reject": "#d62728"}
    trials = sorted(qc["trial"].unique())
    fig, axes = plt.subplots(len(trials), 1, figsize=(13, 1.25 * len(trials)), sharex=True)
    if len(trials) == 1:
        axes = [axes]
    for ax, trial in zip(axes, trials):
        part = qc[qc["trial"] == trial]
        ax.barh(
            np.zeros(len(part)),
            WINDOW_SECONDS,
            left=part["start_seconds"],
            height=0.7,
            color=[colors[s] for s in part["status"]],
            edgecolor="none",
        )
        ax.set_yticks([0], [f"Trial {trial}"])
        ax.set_ylim(-0.6, 0.6)
    axes[-1].set_xlabel("Thời gian trong trial (giây)")
    fig.suptitle("P9 — trạng thái QC của từng window 4 giây (overlap 50%)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-mit003")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--qc-dir", type=Path, default=Path("reports/qc"))
    args = parser.parse_args()

    trial_dir = args.processed_dir / args.subject / "trials"
    files = sorted(trial_dir.glob("*_eeg.fif"))
    if not files:
        raise FileNotFoundError(f"No cleaned trial FIF files found in {trial_dir}")

    rows: list[dict[str, object]] = []
    for trial_path in files:
        trial = int(trial_path.stem.split("trial-")[1].split("_")[0])
        raw = mne.io.read_raw_fif(trial_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        window_samples = int(round(WINDOW_SECONDS * sfreq))
        step_samples = int(round(window_samples * (1 - OVERLAP)))
        data_uv = raw.get_data() * 1e6

        for window_index, start in enumerate(
            range(0, data_uv.shape[1] - window_samples + 1, step_samples), start=1
        ):
            stop = start + window_samples
            metrics = classify_window(data_uv[:, start:stop])
            rows.append(
                {
                    "subject": args.subject,
                    "trial": trial,
                    "window_index": window_index,
                    "start_seconds": start / sfreq,
                    "end_seconds": stop / sfreq,
                    "samples": window_samples,
                    **metrics,
                    "source_path": trial_path.as_posix(),
                }
            )

    qc = pd.DataFrame(rows).sort_values(["trial", "window_index"])
    output_dir = args.qc_dir / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = output_dir / "p9_window_qc.csv"
    qc.to_csv(qc_path, index=False)

    trial_summary = (
        qc.groupby(["trial", "status"]).size().unstack(fill_value=0)
        .reindex(columns=["pass", "review", "reject"], fill_value=0)
    )
    trial_summary["total"] = trial_summary.sum(axis=1)
    for status in ["pass", "review", "reject"]:
        trial_summary[f"{status}_fraction"] = trial_summary[status] / trial_summary["total"]
    trial_summary.reset_index().to_csv(output_dir / "p9_trial_summary.csv", index=False)
    plot_timeline(qc, output_dir / "p9_window_timeline.png")

    counts = {k: int(v) for k, v in qc["status"].value_counts().to_dict().items()}
    summary = {
        "checkpoint": "P9",
        "subject": args.subject,
        "window_seconds": WINDOW_SECONDS,
        "overlap_fraction": OVERLAP,
        "step_seconds": WINDOW_SECONDS * (1 - OVERLAP),
        "trials": int(qc["trial"].nunique()),
        "windows": int(len(qc)),
        "status_counts": counts,
        "status_fractions": {k: v / len(qc) for k, v in counts.items()},
        "thresholds_inherited_from_p8": {
            "reject_peak_uv": 1000,
            "reject_over_200uv_fraction": 0.01,
            "review_peak_uv": 500,
            "review_over_200uv_fraction": 0.001,
            "review_variance_outlier_channels": 5,
        },
        "windows_deleted": False,
        "labels_derived": False,
        "data_split_created": False,
    }
    (output_dir / "p9_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
