# DENS preprocessing and curation rules

This document defines the reusable rules established with the `sub-mit003`
pilot. Subject-specific findings are evidence for the pilot only and are never
copied as fixed decisions to another subject.

## Shared signal-processing rules

- Preserve the raw recording and all cleaned trial FIF files.
- Apply a 50 Hz notch filter and a 0.5–45 Hz band-pass filter, using 15 seconds
  of padding around each trial before cropping.
- Use average EEG reference after approved bad-channel interpolation.
- Segment cleaned trials into 4-second windows with 50% overlap (2-second
  stride).
- Preserve provenance for filters, interpolation, reference, ICA, trials and
  window-QC decisions.

## Bad channels are subject-specific

For every subject, detect bad channels again using channel variance/standard
deviation, peak-to-peak amplitude, global correlation, neighbour correlation
and persistence across trials. Distinguish persistent subject-level bad
channels from trial-local or window-local artifacts. Interpolate only channels
supported by that subject's evidence. `E50` and `E103` are the approved result
for `sub-mit003`, not a reusable channel list.

## ICA is subject-specific

Fit ICA independently for every subject. Generate candidates using ECG/EMG
correlation, component spectrum, kurtosis and topography. Compare no removal,
conservative removal and the candidate set with an ablation measuring residual
physiological leakage, relative RMS change and retained band power. Component
indices have no cross-subject identity: `IC11` is approved only for
`sub-mit003`.

## Window QC and inclusion

- `pass`: include in the primary dataset when its trial is not excluded.
- `review`: exclude from primary training; retain for the `pass + review`
  ablation when its trial is not excluded.
- `reject`: exclude from every train, validation and test set.
- Do not delete physical EEG files. Record Boolean inclusion flags and reasons.

A whole trial is excluded from the primary dataset when at least one condition
holds, calculated over all 4-second, 50%-overlap windows:

- `pass_fraction < 0.30`;
- `reject_fraction >= 0.20`;
- unrecoverable NaN/Inf, channel-count or duration integrity error.

An excluded trial remains available for diagnostics and explicitly named
ablations. A trial is not excluded merely for a few local rejected windows.

## Emotion labels

Ground truth comes from self-reported valence and arousal, never the video
name or a pre-existing quadrant column.

Fixed-5 is the primary label policy:

- score greater than 5 is `high`;
- score less than 5 is `low`;
- score exactly 5 is `ambiguous` and is not eligible for the corresponding
  binary task;
- four-class labels are `HVHA`, `HVLA`, `LVHA` and `LVLA`; a sample is
  ineligible when either score equals 5.

Subject-relative labels are an ablation only. Their valence and arousal median
thresholds must be fitted using training trials inside each fold and then
applied unchanged to validation/test trials. A full-subject median must not be
used during model evaluation.

## Splitting and normalization

- Split by trial, never by overlapping window.
- For cross-subject evaluation, split by subject.
- Fit normalization statistics only on the training partition and apply them
  unchanged to validation and test partitions.
- Keep amplitude-based QC in physical units before normalization.

## Pilot decision for sub-mit003

- Trial 11 is excluded from the primary dataset because its pass fraction is
  below 30% and its reject fraction is at least 20%.
- Trials 2, 9 and 10 remain; only their non-qualifying windows are excluded.
- The subject-specific bad channels are `E50` and `E103`.
- The subject-specific approved ICA removal is `IC11`.
