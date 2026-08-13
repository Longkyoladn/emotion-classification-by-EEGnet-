"""Checkpoint P6b: compare conservative ICA removal strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

EEG = [f"E{i}" for i in range(1, 129)]
BADS = ["E50", "E103"]
STRATEGIES = {
    "no_ica_removal": [],
    "remove_ecg_ic11": [11],
    "remove_all_candidates": [11, 13, 21, 23, 57],
}
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def powers(x: np.ndarray, sfreq: float) -> dict[str, float]:
    f, p = welch(x, fs=sfreq, nperseg=int(4 * sfreq), noverlap=int(2 * sfreq), axis=-1)
    return {
        name: float(
            np.median(
                np.trapezoid(
                    p[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=1
                )
            )
        )
        for name, (lo, hi) in BANDS.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ica", type=Path, required=True)
    ap.add_argument("--subject", default="sub-mit003")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    args = ap.parse_args()
    mf = pd.read_csv(args.manifest)
    trials = mf[
        mf.subject.eq(args.subject)
        & mf.match_status.eq("matched")
        & mf.validation_issues.fillna("").eq("")
    ].sort_values("event_trial_index")
    raw = mne.io.read_raw_eeglab(
        args.dataset_root / trials.eeg_set_path.dropna().unique()[0],
        preload=False,
        verbose="ERROR",
    )
    raw.set_channel_types(
        {
            **{c: "eeg" for c in EEG},
            "E129": "misc",
            "ECG": "ecg",
            "EMG": "emg",
            "EMG_2": "emg",
        },
        verbose="ERROR",
    )
    raw.set_montage(
        mne.channels.make_standard_montage("GSN-HydroCel-128"),
        on_missing="ignore",
        verbose="ERROR",
    )
    ica = mne.preprocessing.read_ica(args.ica, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    pad = int(15 * sf)
    rows = []
    for tr in trials.itertuples(index=False):
        start, stop = int(round(tr.onset_seconds * sf)), int(round(tr.end_seconds * sf))
        ps, pe = max(0, start - pad), min(raw.n_times, stop + pad)
        seg = (
            raw.copy()
            .crop(ps / sf, (pe - 1) / sf, include_tmax=True)
            .load_data(verbose="ERROR")
        )
        seg.notch_filter([50.0], phase="zero", verbose="ERROR").filter(
            0.5, 45.0, phase="zero", verbose="ERROR"
        )
        seg.crop((start - ps) / sf, (stop - ps - 1) / sf, include_tmax=True)
        ecg = seg.get_data(picks=["ECG"])[0]
        eeg = seg.copy().pick(EEG)
        eeg.info["bads"] = BADS
        eeg.interpolate_bads(
            reset_bads=False, method={"eeg": "spline"}, verbose="ERROR"
        )
        eeg.set_eeg_reference("average", projection=False, verbose="ERROR")
        baseline = eeg.get_data() * 1e6
        base_power = powers(baseline, sf)
        for name, exclude in STRATEGIES.items():
            candidate = eeg.copy()
            if exclude:
                ica.apply(candidate, exclude=exclude, verbose="ERROR")
            x = candidate.get_data() * 1e6
            # Residual ECG leakage: strongest absolute EEG-channel correlation.
            ecg_corr = np.array([np.corrcoef(xi, ecg)[0, 1] for xi in x])
            bp = powers(x, sf)
            diff = x - baseline
            rows.append(
                {
                    "trial": int(tr.event_trial_index),
                    "strategy": name,
                    "excluded_components": ";".join(map(str, exclude)),
                    "max_abs_ecg_channel_correlation": float(np.max(np.abs(ecg_corr))),
                    "median_abs_ecg_channel_correlation": float(
                        np.median(np.abs(ecg_corr))
                    ),
                    "median_std_uv": float(np.median(np.std(x, axis=1))),
                    "median_peak_to_peak_uv": float(np.median(np.ptp(x, axis=1))),
                    "relative_rms_change": float(
                        np.sqrt(np.mean(diff**2))
                        / max(np.sqrt(np.mean(baseline**2)), 1e-12)
                    ),
                    **{
                        f"{b}_retained": bp[b] / max(base_power[b], 1e-20)
                        for b in BANDS
                    },
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("strategy", sort=False).median(numeric_only=True).reset_index()
    )
    out = args.output_dir / args.subject
    detail.to_csv(out / "p6b_ica_ablation_by_trial.csv", index=False)
    summary.to_csv(out / "p6b_ica_ablation_summary.csv", index=False)
    report = {
        "checkpoint": "P6b",
        "subject": args.subject,
        "trials": int(len(trials)),
        "strategies": STRATEGIES,
        "results": summary.to_dict("records"),
        "ica_applied_to_saved_eeg": False,
        "raw_file_unchanged": True,
        "operations_not_performed": [
            "saving cleaned EEG",
            "trial rejection",
            "label derivation",
        ],
    }
    (out / "p6b_ica_ablation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, g in detail.groupby("strategy", sort=False):
        axes[0].plot(g.trial, g.max_abs_ecg_channel_correlation, marker="o", label=name)
        axes[1].plot(g.trial, g.relative_rms_change, marker="o", label=name)
    axes[0].set(
        title="Residual ECG leakage", xlabel="Trial", ylabel="Max |corr(EEG, ECG)|"
    )
    axes[1].set(
        title="Waveform change vs no removal",
        xlabel="Trial",
        ylabel="Relative RMS change",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "p6b_ica_ablation.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
