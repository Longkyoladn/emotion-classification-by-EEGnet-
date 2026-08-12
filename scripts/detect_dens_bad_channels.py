"""Checkpoint P4: confirm persistent bad channels without modifying EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd


CHANNELS = [f"E{i}" for i in range(1, 129)]


def rz(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.zeros_like(x) if mad == 0 else (x - med) / (1.4826 * mad)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--subject", default="sub-mit003")
    p.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest)
    trials = manifest[
        manifest.subject.eq(args.subject)
        & manifest.match_status.eq("matched")
        & manifest.validation_issues.fillna("").eq("")
    ].sort_values("event_trial_index")
    set_path = args.dataset_root / trials.eeg_set_path.dropna().unique()[0]
    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), on_missing="ignore", verbose="ERROR")
    picks = mne.pick_channels(raw.ch_names, CHANNELS, ordered=True)
    sfreq = float(raw.info["sfreq"])
    pad = int(15 * sfreq)

    positions = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :6]

    trial_metrics = []
    for trial in trials.itertuples(index=False):
        start = int(round(trial.onset_seconds * sfreq))
        stop = int(round(trial.end_seconds * sfreq))
        pstart, pstop = max(0, start - pad), min(raw.n_times, stop + pad)
        x = raw.get_data(picks=picks, start=pstart, stop=pstop) * 1e6
        x = mne.filter.notch_filter(x, Fs=sfreq, freqs=[50.0], phase="zero", verbose="ERROR")
        x = mne.filter.filter_data(x, sfreq, 0.5, 45.0, phase="zero", verbose="ERROR")
        x = x[:, start - pstart : start - pstart + stop - start]
        std = np.std(x, axis=1)
        p2p = np.ptp(x, axis=1)
        corr = np.corrcoef(x[:, ::10])
        np.fill_diagonal(corr, np.nan)
        global_corr = np.nanmedian(corr, axis=1)
        neighbor_corr = np.array([np.nanmedian(corr[i, neighbors[i]]) for i in range(128)])
        z_std = rz(np.log10(std + 1e-12))
        z_p2p = rz(np.log10(p2p + 1e-12))
        z_global = rz(global_corr)
        z_neighbor = rz(neighbor_corr)
        for i, channel in enumerate(CHANNELS):
            flags = []
            if z_std[i] > 3.5: flags.append("variance")
            if z_p2p[i] > 3.5: flags.append("peak_to_peak")
            if z_global[i] < -3.5: flags.append("global_correlation")
            if z_neighbor[i] < -3.5: flags.append("neighbor_correlation")
            trial_metrics.append({
                "trial": int(trial.event_trial_index), "channel": channel,
                "std_uv": std[i], "peak_to_peak_uv": p2p[i],
                "global_correlation": global_corr[i], "neighbor_correlation": neighbor_corr[i],
                "flags": ";".join(flags), "flagged": bool(flags),
            })

    details = pd.DataFrame(trial_metrics)
    rows = []
    for channel, group in details.groupby("channel", sort=False):
        counts = group["flags"].str.split(";").explode().replace("", np.nan).dropna().value_counts()
        flagged_trials = int(group.flagged.sum())
        independent_metrics = int(sum(counts.get(k, 0) > 0 for k in ["variance", "peak_to_peak", "global_correlation", "neighbor_correlation"]))
        if flagged_trials >= 6 and independent_metrics >= 2:
            decision = "bad"
        elif flagged_trials >= 3 or independent_metrics >= 2:
            decision = "review"
        else:
            decision = "pass"
        rows.append({
            "channel": channel, "decision": decision, "flagged_trials": flagged_trials,
            "trial_fraction": flagged_trials / len(trials), "independent_metrics": independent_metrics,
            "variance_trials": int(counts.get("variance", 0)),
            "peak_to_peak_trials": int(counts.get("peak_to_peak", 0)),
            "global_correlation_trials": int(counts.get("global_correlation", 0)),
            "neighbor_correlation_trials": int(counts.get("neighbor_correlation", 0)),
            "median_std_uv": float(group.std_uv.median()),
            "median_peak_to_peak_uv": float(group.peak_to_peak_uv.median()),
            "median_global_correlation": float(group.global_correlation.median()),
            "median_neighbor_correlation": float(group.neighbor_correlation.median()),
        })
    summary = pd.DataFrame(rows)
    out = args.output_dir / args.subject
    out.mkdir(parents=True, exist_ok=True)
    details.to_csv(out / "p4_bad_channel_trial_metrics.csv", index=False)
    summary.to_csv(out / "p4_bad_channel_decisions.csv", index=False)
    report = {
        "checkpoint": "P4", "subject": args.subject, "trials": len(trials),
        "filter": "notch 50 Hz then band-pass 0.5-45 Hz",
        "bad": summary.loc[summary.decision.eq("bad"), "channel"].tolist(),
        "review": summary.loc[summary.decision.eq("review"), "channel"].tolist(),
        "pass_count": int(summary.decision.eq("pass").sum()),
        "decision_rule": {
            "bad": "flagged in >=6/11 trials and by >=2 independent metrics",
            "review": "flagged in >=3/11 trials or by >=2 independent metrics",
        },
        "raw_file_unchanged": True,
        "operations_not_performed": ["interpolation", "re-referencing", "ICA", "trial rejection", "saving filtered EEG"],
    }
    (out / "p4_bad_channel_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
