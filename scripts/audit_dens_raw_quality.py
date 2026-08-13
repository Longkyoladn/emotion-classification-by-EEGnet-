"""Run the P1/P2 structural and raw-signal QC checkpoint for one DENS subject."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

EEG_CHANNELS = [f"E{i}" for i in range(1, 129)]
CHANNEL_TYPES = {
    **{name: "eeg" for name in EEG_CHANNELS},
    "E129": "misc",
    "ECG": "ecg",
    "EMG": "emg",
    "EMG_2": "emg",
}
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subject", default="sub-mit003")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--psd-chunks", type=int, default=20)
    return parser.parse_args()


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad)


def band_integral(
    freqs: np.ndarray, psd: np.ndarray, low: float, high: float
) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def main() -> None:
    args = parse_args()
    subject_manifest = pd.read_csv(args.manifest)
    subject_manifest = subject_manifest.loc[
        subject_manifest.subject.eq(args.subject)
    ].copy()
    if subject_manifest.empty:
        raise RuntimeError(f"Subject {args.subject!r} is absent from the manifest")

    set_paths = subject_manifest.eeg_set_path.dropna().unique()
    if len(set_paths) != 1:
        raise RuntimeError(f"Expected one EEG .set path, found {set_paths.tolist()}")
    set_path = args.dataset_root / set_paths[0]
    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")

    original_types = Counter(raw.get_channel_types())
    missing_expected_channels = sorted(set(CHANNEL_TYPES) - set(raw.ch_names))
    unexpected_channels = sorted(set(raw.ch_names) - set(CHANNEL_TYPES))
    available_mapping = {
        name: kind for name, kind in CHANNEL_TYPES.items() if name in raw.ch_names
    }
    raw.set_channel_types(available_mapping, verbose="ERROR")
    raw.set_montage(
        mne.channels.make_standard_montage("GSN-HydroCel-128"),
        on_missing="ignore",
        verbose="ERROR",
    )
    corrected_types = Counter(raw.get_channel_types())
    eeg_picks = mne.pick_channels(raw.ch_names, EEG_CHANNELS, ordered=True)
    if len(eeg_picks) != 128:
        raise RuntimeError(f"Expected 128 EEG channels, found {len(eeg_picks)}")

    sfreq = float(raw.info["sfreq"])
    recording_duration = float(raw.times[-1])
    valid_trials = subject_manifest.loc[
        subject_manifest.match_status.eq("matched")
        & subject_manifest.validation_issues.fillna("").eq("")
    ].copy()
    trial_bounds_valid = bool(
        valid_trials.onset_seconds.ge(0).all()
        and valid_trials.end_seconds.gt(valid_trials.onset_seconds).all()
        and valid_trials.end_seconds.le(recording_duration + 1 / sfreq).all()
    )

    annotation_starts = raw.annotations.onset[
        np.asarray(raw.annotations.description) == "stm"
    ]
    manifest_starts = valid_trials.onset_seconds.to_numpy(float)
    event_alignment_errors = (
        np.abs(annotation_starts[: len(manifest_starts)] - manifest_starts)
        if len(annotation_starts) >= len(manifest_starts)
        else np.array([np.nan])
    )

    # Streaming full-recording metrics avoid loading the complete .fdt into RAM.
    n_channels = len(eeg_picks)
    sums = np.zeros(n_channels)
    sums_sq = np.zeros(n_channels)
    minima = np.full(n_channels, np.inf)
    maxima = np.full(n_channels, -np.inf)
    flat_differences = np.zeros(n_channels, dtype=np.int64)
    finite_values = np.zeros(n_channels, dtype=np.int64)
    nonfinite_values = np.zeros(n_channels, dtype=np.int64)
    chunk_samples = max(1, int(round(args.chunk_seconds * sfreq)))

    for start in range(0, raw.n_times, chunk_samples):
        stop = min(raw.n_times, start + chunk_samples)
        data_uv = raw.get_data(picks=eeg_picks, start=start, stop=stop) * 1e6
        finite = np.isfinite(data_uv)
        safe = np.where(finite, data_uv, 0.0)
        sums += safe.sum(axis=1)
        sums_sq += np.square(safe).sum(axis=1)
        finite_values += finite.sum(axis=1)
        nonfinite_values += (~finite).sum(axis=1)
        minima = np.minimum(minima, np.nanmin(data_uv, axis=1))
        maxima = np.maximum(maxima, np.nanmax(data_uv, axis=1))
        if data_uv.shape[1] > 1:
            flat_differences += (np.abs(np.diff(data_uv, axis=1)) < 0.01).sum(axis=1)

    means = sums / np.maximum(finite_values, 1)
    variances = sums_sq / np.maximum(finite_values, 1) - np.square(means)
    std_uv = np.sqrt(np.maximum(variances, 0.0))
    peak_to_peak_uv = maxima - minima
    flat_fraction = flat_differences / max(raw.n_times - 1, 1)

    # Sample evenly spaced chunks for spectral and cross-channel diagnostics.
    psd_chunk_samples = int(round(30.0 * sfreq))
    max_start = max(0, raw.n_times - psd_chunk_samples)
    starts = np.linspace(0, max_start, args.psd_chunks, dtype=int)
    psd_chunks: list[np.ndarray] = []
    correlation_samples: list[np.ndarray] = []
    local_std_chunks: list[np.ndarray] = []
    local_peak_to_peak_chunks: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for start in starts:
        data_uv = (
            raw.get_data(
                picks=eeg_picks, start=int(start), stop=int(start + psd_chunk_samples)
            )
            * 1e6
        )
        data_uv = detrend(data_uv, axis=1, type="linear")
        local_std_chunks.append(np.std(data_uv, axis=1))
        local_peak_to_peak_chunks.append(np.ptp(data_uv, axis=1))
        freqs, psd = welch(
            data_uv,
            fs=sfreq,
            nperseg=int(4 * sfreq),
            noverlap=int(2 * sfreq),
            axis=1,
        )
        psd_chunks.append(psd)
        correlation_samples.append(data_uv[:, ::10])

    assert freqs is not None
    median_psd = np.median(np.stack(psd_chunks), axis=0)
    local_std_uv = np.median(np.stack(local_std_chunks), axis=0)
    local_peak_to_peak_uv = np.median(np.stack(local_peak_to_peak_chunks), axis=0)
    correlation_data = np.concatenate(correlation_samples, axis=1)
    correlation_matrix = np.corrcoef(correlation_data)
    np.fill_diagonal(correlation_matrix, np.nan)
    median_channel_correlation = np.nanmedian(correlation_matrix, axis=1)

    line_power = band_integral(freqs, median_psd, 49.0, 51.0)
    line_neighbors = band_integral(freqs, median_psd, 45.0, 49.0) + band_integral(
        freqs, median_psd, 51.0, 55.0
    )
    line_noise_ratio = line_power / np.maximum(line_neighbors / 4.0, 1e-20)
    high_frequency_ratio = band_integral(freqs, median_psd, 35.0, 45.0) / np.maximum(
        band_integral(freqs, median_psd, 8.0, 30.0), 1e-20
    )

    std_z = robust_z(np.log10(local_std_uv + 1e-12))
    p2p_z = robust_z(np.log10(local_peak_to_peak_uv + 1e-12))
    line_z = robust_z(np.log10(line_noise_ratio + 1e-12))
    hf_z = robust_z(np.log10(high_frequency_ratio + 1e-12))
    correlation_z = robust_z(median_channel_correlation)

    channel_rows = []
    for index, channel in enumerate(EEG_CHANNELS):
        reasons = []
        if local_std_uv[index] < 0.5 or flat_fraction[index] > 0.99:
            reasons.append("flat")
        if abs(std_z[index]) > 3.5:
            reasons.append("variance_outlier")
        if p2p_z[index] > 3.5:
            reasons.append("peak_to_peak_outlier")
        if line_z[index] > 3.5:
            reasons.append("line_noise_outlier")
        if hf_z[index] > 3.5:
            reasons.append("high_frequency_outlier")
        if correlation_z[index] < -3.5:
            reasons.append("low_correlation")
        if nonfinite_values[index] > 0:
            reasons.append("nonfinite_values")
        channel_rows.append(
            {
                "channel": channel,
                "mean_uv": means[index],
                "global_std_uv": std_uv[index],
                "global_peak_to_peak_uv": peak_to_peak_uv[index],
                "sampled_local_std_uv": local_std_uv[index],
                "sampled_local_peak_to_peak_uv": local_peak_to_peak_uv[index],
                "flat_fraction": flat_fraction[index],
                "nonfinite_values": nonfinite_values[index],
                "median_channel_correlation": median_channel_correlation[index],
                "line_noise_ratio_50hz": line_noise_ratio[index],
                "high_frequency_ratio_35_45_to_8_30": high_frequency_ratio[index],
                "robust_z_log_std": std_z[index],
                "robust_z_log_peak_to_peak": p2p_z[index],
                "robust_z_log_line_noise": line_z[index],
                "robust_z_log_high_frequency": hf_z[index],
                "robust_z_median_correlation": correlation_z[index],
                "suspect": bool(reasons),
                "suspect_reasons": ";".join(reasons),
            }
        )
    channels = pd.DataFrame(channel_rows)

    subject_output = args.output_dir / args.subject
    subject_output.mkdir(parents=True, exist_ok=True)
    channels.to_csv(subject_output / "channel_raw_qc.csv", index=False)

    aggregate_psd = np.nanmedian(median_psd, axis=0)
    lower_psd = np.nanpercentile(median_psd, 25, axis=0)
    upper_psd = np.nanpercentile(median_psd, 75, axis=0)
    figure, axis = plt.subplots(figsize=(10, 5))
    plot_mask = (freqs >= 0.5) & (freqs <= 60.0)
    axis.semilogy(freqs[plot_mask], aggregate_psd[plot_mask], label="channel median")
    axis.fill_between(
        freqs[plot_mask],
        lower_psd[plot_mask],
        upper_psd[plot_mask],
        alpha=0.25,
        label="channel IQR",
    )
    axis.axvline(50.0, color="red", linestyle="--", label="50 Hz")
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="PSD (µV²/Hz)",
        title=f"Raw PSD — {args.subject}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(subject_output / "raw_psd.png", dpi=160)
    plt.close(figure)

    report = {
        "checkpoint": "P1_P2",
        "subject": args.subject,
        "source_set": set_paths[0],
        "raw_file_unchanged": True,
        "structure": {
            "sampling_frequency_hz": sfreq,
            "samples": int(raw.n_times),
            "recording_duration_seconds": recording_duration,
            "channels_total": int(raw.info["nchan"]),
            "channel_types_before_correction": dict(original_types),
            "channel_types_after_correction": dict(corrected_types),
            "missing_expected_channels": missing_expected_channels,
            "unexpected_channels": unexpected_channels,
            "montage_eeg_channels_with_positions": int(
                sum(
                    np.linalg.norm(raw.info["chs"][pick]["loc"][:3]) > 0
                    for pick in eeg_picks
                )
            ),
            "manifest_trials": int(len(subject_manifest)),
            "structurally_valid_trials": int(len(valid_trials)),
            "trial_bounds_valid": trial_bounds_valid,
            "stimulus_annotations": int(len(annotation_starts)),
            "max_manifest_annotation_alignment_error_seconds": (
                float(np.nanmax(event_alignment_errors))
                if event_alignment_errors.size
                else None
            ),
        },
        "raw_quality": {
            "nonfinite_values_total": int(nonfinite_values.sum()),
            "global_std_uv_quantiles": dict(
                zip(
                    ["min", "q25", "median", "q75", "max"],
                    np.quantile(std_uv, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                )
            ),
            "global_peak_to_peak_uv_quantiles": dict(
                zip(
                    ["min", "q25", "median", "q75", "max"],
                    np.quantile(peak_to_peak_uv, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                )
            ),
            "sampled_local_std_uv_quantiles": dict(
                zip(
                    ["min", "q25", "median", "q75", "max"],
                    np.quantile(local_std_uv, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                )
            ),
            "sampled_local_peak_to_peak_uv_quantiles": dict(
                zip(
                    ["min", "q25", "median", "q75", "max"],
                    np.quantile(
                        local_peak_to_peak_uv, [0, 0.25, 0.5, 0.75, 1]
                    ).tolist(),
                )
            ),
            "boundary_annotations": int(
                np.sum(np.asarray(raw.annotations.description) == "boundary")
            ),
            "line_noise_ratio_50hz_quantiles": dict(
                zip(
                    ["min", "q25", "median", "q75", "max"],
                    np.quantile(line_noise_ratio, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                )
            ),
            "median_channel_correlation_quantiles": dict(
                zip(
                    ["min", "q25", "median", "q75", "max"],
                    np.quantile(
                        median_channel_correlation, [0, 0.25, 0.5, 0.75, 1]
                    ).tolist(),
                )
            ),
            "suspect_channels_count": int(channels.suspect.sum()),
            "suspect_channels": channels.loc[
                channels.suspect, ["channel", "suspect_reasons"]
            ].to_dict("records"),
            "thresholds_are_provisional": True,
        },
        "artifacts": {
            "channel_metrics_csv": "channel_raw_qc.csv",
            "raw_psd_figure": "raw_psd.png",
        },
        "operations_not_performed": [
            "filtering",
            "notch filtering",
            "bad-channel marking",
            "interpolation",
            "re-referencing",
            "ICA",
            "trial rejection",
            "label derivation",
        ],
    }
    (subject_output / "p1_p2_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
