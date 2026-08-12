"""Run checkpoint P3: compare candidate filters without saving filtered EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch


EEG_CHANNELS = [f"E{i}" for i in range(1, 129)]
BANDS = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}
CONFIGURATIONS = {
    "raw": {"notch": False, "highpass": None, "lowpass": None},
    "bandpass_0p5_45": {"notch": False, "highpass": 0.5, "lowpass": 45.0},
    "notch50_bandpass_0p5_45": {"notch": True, "highpass": 0.5, "lowpass": 45.0},
    "bandpass_0p5_40": {"notch": False, "highpass": 0.5, "lowpass": 40.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subject", default="sub-mit003")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    parser.add_argument("--padding-seconds", type=float, default=15.0)
    return parser.parse_args()


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad)


def integrate(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def apply_configuration(data: np.ndarray, sfreq: float, config: dict) -> np.ndarray:
    output = data.copy()
    if config["notch"]:
        output = mne.filter.notch_filter(
            output,
            Fs=sfreq,
            freqs=[50.0],
            method="fir",
            phase="zero",
            verbose="ERROR",
        )
    if config["highpass"] is not None or config["lowpass"] is not None:
        output = mne.filter.filter_data(
            output,
            sfreq=sfreq,
            l_freq=config["highpass"],
            h_freq=config["lowpass"],
            method="fir",
            phase="zero",
            verbose="ERROR",
        )
    return output


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    trials = manifest.loc[
        manifest.subject.eq(args.subject)
        & manifest.match_status.eq("matched")
        & manifest.validation_issues.fillna("").eq("")
    ].sort_values("event_trial_index")
    if trials.empty:
        raise RuntimeError(f"No valid trials for {args.subject}")

    set_paths = trials.eeg_set_path.dropna().unique()
    if len(set_paths) != 1:
        raise RuntimeError(f"Expected one .set path, found {set_paths.tolist()}")
    raw = mne.io.read_raw_eeglab(
        args.dataset_root / set_paths[0], preload=False, verbose="ERROR"
    )
    sfreq = float(raw.info["sfreq"])
    eeg_picks = mne.pick_channels(raw.ch_names, EEG_CHANNELS, ordered=True)
    padding_samples = int(round(args.padding_seconds * sfreq))
    edge_samples = int(round(2.0 * sfreq))

    outputs: dict[str, dict[str, list]] = {
        name: {
            "psd": [],
            "std": [],
            "peak_to_peak": [],
            "correlation_samples": [],
            "edge_ratio": [],
        }
        for name in CONFIGURATIONS
    }
    freqs: np.ndarray | None = None

    for trial in trials.itertuples(index=False):
        onset = int(round(trial.onset_seconds * sfreq))
        end = int(round(trial.end_seconds * sfreq))
        padded_start = max(0, onset - padding_samples)
        padded_end = min(raw.n_times, end + padding_samples)
        left_crop = onset - padded_start
        right_crop = left_crop + (end - onset)
        padded = raw.get_data(
            picks=eeg_picks, start=padded_start, stop=padded_end
        ) * 1e6

        for name, config in CONFIGURATIONS.items():
            if name == "raw":
                central = detrend(padded[:, left_crop:right_crop], axis=1, type="linear")
            else:
                filtered = apply_configuration(padded, sfreq, config)
                central = filtered[:, left_crop:right_crop]

            freqs, psd = welch(
                central,
                fs=sfreq,
                nperseg=int(4 * sfreq),
                noverlap=int(2 * sfreq),
                axis=1,
            )
            outputs[name]["psd"].append(psd)
            outputs[name]["std"].append(np.std(central, axis=1))
            outputs[name]["peak_to_peak"].append(np.ptp(central, axis=1))
            outputs[name]["correlation_samples"].append(central[:, ::10])

            middle = central[:, edge_samples:-edge_samples]
            edges = np.concatenate(
                [central[:, :edge_samples], central[:, -edge_samples:]], axis=1
            )
            middle_rms = np.sqrt(np.mean(np.square(middle), axis=1))
            edge_rms = np.sqrt(np.mean(np.square(edges), axis=1))
            outputs[name]["edge_ratio"].append(
                edge_rms / np.maximum(middle_rms, 1e-12)
            )

    assert freqs is not None
    summary_rows = []
    channel_frames = []
    processed: dict[str, dict] = {}
    raw_band_power: dict[str, float] = {}

    for name, output in outputs.items():
        trial_psd = np.stack(output["psd"])
        median_psd_by_channel = np.median(trial_psd, axis=0)
        aggregate_psd = np.median(median_psd_by_channel, axis=0)
        local_std = np.median(np.stack(output["std"]), axis=0)
        local_p2p = np.median(np.stack(output["peak_to_peak"]), axis=0)
        edge_ratio = np.median(np.stack(output["edge_ratio"]), axis=0)
        correlation_data = np.concatenate(output["correlation_samples"], axis=1)
        correlations = np.corrcoef(correlation_data)
        np.fill_diagonal(correlations, np.nan)
        median_correlation = np.nanmedian(correlations, axis=1)

        line_power = integrate(freqs, median_psd_by_channel, 49.0, 51.0)
        neighbor_power = integrate(freqs, median_psd_by_channel, 45.0, 49.0) + integrate(
            freqs, median_psd_by_channel, 51.0, 55.0
        )
        line_ratio = line_power / np.maximum(neighbor_power / 4.0, 1e-20)
        band_power = {
            band: float(np.median(integrate(freqs, median_psd_by_channel, low, high)))
            for band, (low, high) in BANDS.items()
        }
        if name == "raw":
            raw_band_power = band_power

        std_z = robust_z(np.log10(local_std + 1e-12))
        p2p_z = robust_z(np.log10(local_p2p + 1e-12))
        correlation_z = robust_z(median_correlation)
        suspect_reasons = []
        for index in range(128):
            reasons = []
            if local_std[index] < 0.5:
                reasons.append("flat")
            if abs(std_z[index]) > 3.5:
                reasons.append("variance_outlier")
            if p2p_z[index] > 3.5:
                reasons.append("peak_to_peak_outlier")
            if correlation_z[index] < -3.5:
                reasons.append("low_correlation")
            suspect_reasons.append(";".join(reasons))

        frame = pd.DataFrame(
            {
                "configuration": name,
                "channel": EEG_CHANNELS,
                "median_trial_std_uv": local_std,
                "median_trial_peak_to_peak_uv": local_p2p,
                "median_channel_correlation": median_correlation,
                "line_noise_ratio_50hz": line_ratio,
                "median_edge_to_middle_rms_ratio": edge_ratio,
                "suspect": [bool(value) for value in suspect_reasons],
                "suspect_reasons": suspect_reasons,
            }
        )
        channel_frames.append(frame)
        suspect = frame.loc[frame.suspect, ["channel", "suspect_reasons"]]
        processed[name] = {
            "aggregate_psd": aggregate_psd,
            "band_power": band_power,
            "line_ratio": line_ratio,
            "std": local_std,
            "p2p": local_p2p,
            "correlation": median_correlation,
            "edge_ratio": edge_ratio,
            "suspect": suspect.to_dict("records"),
        }

    for name, result in processed.items():
        row = {
            "configuration": name,
            "median_std_uv": float(np.median(result["std"])),
            "median_peak_to_peak_uv": float(np.median(result["p2p"])),
            "median_channel_correlation": float(np.median(result["correlation"])),
            "median_line_noise_ratio_50hz": float(np.median(result["line_ratio"])),
            "median_edge_to_middle_rms_ratio": float(np.median(result["edge_ratio"])),
            "suspect_channels_count": len(result["suspect"]),
        }
        for band, power in result["band_power"].items():
            row[f"{band}_power_uv2"] = power
            row[f"{band}_retained_vs_raw"] = power / max(raw_band_power[band], 1e-20)
        summary_rows.append(row)

    summary_table = pd.DataFrame(summary_rows)
    channels_table = pd.concat(channel_frames, ignore_index=True)
    subject_output = args.output_dir / args.subject
    subject_output.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(subject_output / "filter_comparison.csv", index=False)
    channels_table.to_csv(subject_output / "filter_channel_metrics.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    full_mask = (freqs >= 0.5) & (freqs <= 60.0)
    zoom_mask = (freqs >= 44.0) & (freqs <= 56.0)
    for name, result in processed.items():
        axes[0].semilogy(freqs[full_mask], result["aggregate_psd"][full_mask], label=name)
        axes[1].semilogy(freqs[zoom_mask], result["aggregate_psd"][zoom_mask], label=name)
    for axis in axes:
        axis.axvline(50.0, color="red", linestyle="--", alpha=0.7)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Median PSD (µV²/Hz)")
    axes[0].set_title("Filter comparison: 0.5–60 Hz")
    axes[1].set_title("Residual line noise: 44–56 Hz")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(subject_output / "filter_comparison_psd.png", dpi=160)
    plt.close(figure)

    report = {
        "checkpoint": "P3",
        "subject": args.subject,
        "trials": int(len(trials)),
        "padding_seconds": args.padding_seconds,
        "raw_file_unchanged": True,
        "configurations": CONFIGURATIONS,
        "results": summary_table.to_dict("records"),
        "suspect_channels_by_configuration": {
            name: result["suspect"] for name, result in processed.items()
        },
        "operations_not_performed": [
            "bad-channel marking",
            "interpolation",
            "re-referencing",
            "ICA",
            "trial rejection",
            "saving filtered EEG",
            "label derivation",
        ],
    }
    (subject_output / "p3_filter_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
