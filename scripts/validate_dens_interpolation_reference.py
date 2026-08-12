"""Checkpoint P5: validate interpolation and average reference on pilot trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


CHANNELS = [f"E{i}" for i in range(1, 129)]
BAD_CHANNELS = ["E50", "E103"]
# Spatially distributed channels that passed P4; never use review/bad channels here.
VALIDATION_CHANNELS = ["E5", "E18", "E28", "E41", "E65", "E83", "E96", "E116"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def band_power(x: np.ndarray, sfreq: float) -> dict[str, float]:
    f, p = welch(x, fs=sfreq, nperseg=int(4 * sfreq), noverlap=int(2 * sfreq), axis=-1)
    return {name: float(np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])) for name, (lo, hi) in BANDS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subject", default="sub-mit003")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    trials = manifest[manifest.subject.eq(args.subject) & manifest.match_status.eq("matched") & manifest.validation_issues.fillna("").eq("")].sort_values("event_trial_index")
    set_path = args.dataset_root / trials.eeg_set_path.dropna().unique()[0]
    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    raw.set_channel_types({**{c: "eeg" for c in CHANNELS}, "E129": "misc", "ECG": "ecg", "EMG": "emg", "EMG_2": "emg"}, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), on_missing="ignore", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    pad = int(15 * sfreq)
    metrics, validation = [], []
    waveforms = {state: [] for state in ["filtered", "interpolated", "average_referenced"]}

    for trial in trials.itertuples(index=False):
        start, stop = int(round(trial.onset_seconds * sfreq)), int(round(trial.end_seconds * sfreq))
        pstart, pstop = max(0, start-pad), min(raw.n_times, stop+pad)
        segment = raw.copy().crop(pstart/sfreq, (pstop-1)/sfreq, include_tmax=True).pick(CHANNELS).load_data(verbose="ERROR")
        segment.notch_filter([50.0], phase="zero", verbose="ERROR").filter(0.5, 45.0, phase="zero", verbose="ERROR")
        segment.crop((start-pstart)/sfreq, (stop-pstart-1)/sfreq, include_tmax=True)
        filtered = segment.copy()
        interpolated = filtered.copy()
        interpolated.info["bads"] = BAD_CHANNELS
        interpolated.interpolate_bads(reset_bads=False, method={"eeg": "spline"}, verbose="ERROR")
        referenced = interpolated.copy().set_eeg_reference("average", projection=False, verbose="ERROR")

        arrays = {"filtered": filtered.get_data()*1e6, "interpolated": interpolated.get_data()*1e6, "average_referenced": referenced.get_data()*1e6}
        for state, x in arrays.items():
            waveforms[state].append(x)
            common_mode = np.mean(x, axis=0)
            for channel in BAD_CHANNELS:
                i = CHANNELS.index(channel)
                others = [j for j in range(128) if j != i]
                metrics.append({"trial": int(trial.event_trial_index), "state": state, "channel": channel, "std_uv": np.std(x[i]), "peak_to_peak_uv": np.ptp(x[i]), "median_correlation_to_others": np.median([np.corrcoef(x[i], x[j])[0,1] for j in others]), "common_mode_rms_uv": np.sqrt(np.mean(common_mode**2))})

        # Verify interpolation by hiding one known-good channel at a time.
        for channel in VALIDATION_CHANNELS:
            truth = filtered.get_data(picks=[channel])[0] * 1e6
            test = filtered.copy()
            test.info["bads"] = [channel]
            test.interpolate_bads(reset_bads=False, method={"eeg": "spline"}, verbose="ERROR")
            estimate = test.get_data(picks=[channel])[0] * 1e6
            corr = float(np.corrcoef(truth, estimate)[0, 1])
            nrmse = float(np.sqrt(np.mean((truth-estimate)**2)) / max(np.std(truth), 1e-12))
            true_bp, estimate_bp = band_power(truth, sfreq), band_power(estimate, sfreq)
            validation.append({"trial": int(trial.event_trial_index), "channel": channel, "correlation": corr, "nrmse": nrmse, **{f"{b}_relative_error": abs(estimate_bp[b]-true_bp[b])/max(true_bp[b],1e-20) for b in BANDS}})

    metric_df, validation_df = pd.DataFrame(metrics), pd.DataFrame(validation)
    good_idx = [i for i,c in enumerate(CHANNELS) if c not in BAD_CHANNELS]
    filtered_all = np.concatenate(waveforms["filtered"], axis=1)
    interpolated_all = np.concatenate(waveforms["interpolated"], axis=1)
    referenced_all = np.concatenate(waveforms["average_referenced"], axis=1)
    good_difference = interpolated_all[good_idx] - filtered_all[good_idx]
    common_before = np.sqrt(np.mean(np.mean(interpolated_all, axis=0)**2))
    common_after = np.sqrt(np.mean(np.mean(referenced_all, axis=0)**2))

    summary = {
        "checkpoint": "P5", "subject": args.subject, "trials": int(len(trials)),
        "bad_channels_interpolated": BAD_CHANNELS,
        "interpolation_method": "MNE spherical spline",
        "good_channel_validation_set": VALIDATION_CHANNELS,
        "good_channel_validation": {
            "tests": int(len(validation_df)),
            "median_correlation": float(validation_df.correlation.median()),
            "q25_correlation": float(validation_df.correlation.quantile(.25)),
            "median_nrmse": float(validation_df.nrmse.median()),
            "q75_nrmse": float(validation_df.nrmse.quantile(.75)),
            "median_band_power_relative_error": {b: float(validation_df[f"{b}_relative_error"].median()) for b in BANDS},
        },
        "interpolation_effect": {
            "max_absolute_change_on_126_good_channels_uv": float(np.max(np.abs(good_difference))),
            "E50_median_std_before_uv": float(metric_df[(metric_df.channel=="E50") & (metric_df.state=="filtered")].std_uv.median()),
            "E50_median_std_after_uv": float(metric_df[(metric_df.channel=="E50") & (metric_df.state=="interpolated")].std_uv.median()),
            "E103_median_std_before_uv": float(metric_df[(metric_df.channel=="E103") & (metric_df.state=="filtered")].std_uv.median()),
            "E103_median_std_after_uv": float(metric_df[(metric_df.channel=="E103") & (metric_df.state=="interpolated")].std_uv.median()),
        },
        "average_reference_effect": {"common_mode_rms_before_uv": float(common_before), "common_mode_rms_after_uv": float(common_after)},
        "raw_file_unchanged": True,
        "operations_not_performed": ["ICA", "trial rejection", "saving cleaned EEG", "label derivation"],
    }
    out = args.output_dir / args.subject
    out.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(out / "p5_interpolation_reference_metrics.csv", index=False)
    validation_df.to_csv(out / "p5_leave_one_good_channel_out.csv", index=False)
    (out / "p5_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13,4))
    for ax, channel in zip(axes, BAD_CHANNELS):
        for state in ["filtered", "interpolated", "average_referenced"]:
            values = metric_df[(metric_df.channel==channel)&(metric_df.state==state)].std_uv
            ax.plot(range(1,len(values)+1), values, marker="o", label=state)
        ax.set(title=f"{channel}: std by trial", xlabel="Trial", ylabel="Std (µV)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "p5_bad_channel_std_comparison.png", dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
