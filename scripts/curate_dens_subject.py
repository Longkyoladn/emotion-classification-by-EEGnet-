"""Curate QC windows and derive leakage-safe fixed-5 emotion labels.

No EEG file is removed or modified. Subject-relative labels are intentionally
not materialized here because their thresholds must be fitted inside each
training fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def binary_fixed5(score: float, threshold: float) -> str:
    if score > threshold:
        return "high"
    if score < threshold:
        return "low"
    return "ambiguous"


def quadrant_fixed5(valence: float, arousal: float, threshold: float) -> str:
    v = binary_fixed5(valence, threshold)
    a = binary_fixed5(arousal, threshold)
    if "ambiguous" in (v, a):
        return "ambiguous"
    return {
        ("high", "high"): "HVHA",
        ("high", "low"): "HVLA",
        ("low", "high"): "LVHA",
        ("low", "low"): "LVLA",
    }[(v, a)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-mit003")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifest/dens_trials.csv")
    )
    parser.add_argument("--window-qc", type=Path)
    parser.add_argument("--trial-qc", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    parser.add_argument("--rules", type=Path, default=Path("configs/dens_rules.json"))
    args = parser.parse_args()

    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    fixed_threshold = float(rules["labels"]["fixed_threshold"])
    pass_fraction_lt = float(rules["trial_exclusion"]["pass_fraction_lt"])
    reject_fraction_gte = float(rules["trial_exclusion"]["reject_fraction_gte"])
    integrity_issues = rules["trial_exclusion"]["unrecoverable_integrity_issues"]

    qc_dir = args.output_dir / args.subject
    window_path = args.window_qc or qc_dir / "p9_window_qc.csv"
    trial_path = args.trial_qc or qc_dir / "p7_p8_trial_qc.csv"
    windows = pd.read_csv(window_path)
    manifest = pd.read_csv(args.manifest)
    trial_qc = pd.read_csv(trial_path)

    windows = windows[windows["subject"].eq(args.subject)].copy()
    trials = manifest[
        manifest["subject"].eq(args.subject)
        & manifest["match_status"].eq("matched")
        & manifest["validation_issues"].fillna("").eq("")
    ].copy()
    trials["trial"] = trials["event_trial_index"].astype(int)
    trial_qc = trial_qc[trial_qc["subject"].eq(args.subject)].copy()
    trial_qc["trial"] = trial_qc["trial"].astype(int)

    label_columns = trials[["trial", "stimulus", "valence", "arousal"]].copy()
    label_columns["valence_label_fixed5"] = label_columns["valence"].map(
        lambda x: binary_fixed5(x, fixed_threshold)
    )
    label_columns["arousal_label_fixed5"] = label_columns["arousal"].map(
        lambda x: binary_fixed5(x, fixed_threshold)
    )
    label_columns["quadrant_label_fixed5"] = [
        quadrant_fixed5(v, a, fixed_threshold)
        for v, a in zip(label_columns["valence"], label_columns["arousal"])
    ]
    label_columns["valence_binary_eligible_fixed5"] = label_columns[
        "valence_label_fixed5"
    ].ne("ambiguous")
    label_columns["arousal_binary_eligible_fixed5"] = label_columns[
        "arousal_label_fixed5"
    ].ne("ambiguous")
    label_columns["quadrant_label_eligible_fixed5"] = label_columns[
        "quadrant_label_fixed5"
    ].ne("ambiguous")
    label_columns["label_source"] = "self_report"
    label_columns["subject_relative_label_status"] = "derive_inside_training_fold"

    counts = windows.groupby(["trial", "status"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["pass", "review", "reject"], fill_value=0)
    counts["total_windows"] = counts.sum(axis=1)
    for status in ("pass", "review", "reject"):
        counts[f"{status}_fraction"] = counts[status] / counts["total_windows"]
    counts = counts.reset_index()

    severe = trial_qc[["trial", "issues"]].copy()
    severe["unrecoverable_integrity_error"] = (
        severe["issues"].fillna("").str.contains("|".join(integrity_issues), regex=True)
    )
    trial_cur = counts.merge(
        severe[["trial", "unrecoverable_integrity_error"]], on="trial", how="left"
    )
    trial_cur["unrecoverable_integrity_error"] = trial_cur[
        "unrecoverable_integrity_error"
    ].fillna(False)
    trial_cur["exclude_primary"] = (
        trial_cur["pass_fraction"].lt(pass_fraction_lt)
        | trial_cur["reject_fraction"].ge(reject_fraction_gte)
        | trial_cur["unrecoverable_integrity_error"]
    )

    def trial_reason(row: pd.Series) -> str:
        reasons = []
        if row["pass_fraction"] < pass_fraction_lt:
            reasons.append(f"pass_fraction_lt_{pass_fraction_lt:.2f}")
        if row["reject_fraction"] >= reject_fraction_gte:
            reasons.append(f"reject_fraction_ge_{reject_fraction_gte:.2f}")
        if row["unrecoverable_integrity_error"]:
            reasons.append("unrecoverable_integrity_error")
        return ";".join(reasons)

    trial_cur["exclusion_reason"] = trial_cur.apply(trial_reason, axis=1)
    trial_cur["include_ablation"] = True
    trial_cur = trial_cur.merge(label_columns, on="trial", how="left")

    windows = windows.merge(
        trial_cur[["trial", "exclude_primary", "exclusion_reason"]],
        on="trial",
        how="left",
    ).merge(label_columns, on="trial", how="left")
    windows["include_primary"] = (
        windows["status"].eq("pass") & ~windows["exclude_primary"]
    )
    windows["include_pass_review_ablation"] = windows["status"].isin(["pass", "review"])
    windows["include_pass_review_ablation"] &= ~windows["exclude_primary"]
    windows["retain_for_analysis"] = True
    windows["include_excluded_trial_ablation"] = windows["exclude_primary"] & windows[
        "status"
    ].isin(["pass", "review"])
    windows["include_primary_valence_binary"] = (
        windows["include_primary"] & windows["valence_binary_eligible_fixed5"]
    )
    windows["include_primary_arousal_binary"] = (
        windows["include_primary"] & windows["arousal_binary_eligible_fixed5"]
    )
    windows["include_primary_quadrant"] = (
        windows["include_primary"] & windows["quadrant_label_eligible_fixed5"]
    )

    def window_reason(row: pd.Series) -> str:
        reasons = []
        if row["exclude_primary"]:
            reasons.append("trial_excluded:" + row["exclusion_reason"])
        if row["status"] == "review":
            reasons.append("window_qc_review")
        elif row["status"] == "reject":
            reasons.append("window_qc_reject")
        return ";".join(reasons)

    windows["primary_exclusion_reason"] = windows.apply(window_reason, axis=1)
    windows["split_group"] = windows.apply(
        lambda r: f"{r['subject']}_trial-{int(r['trial']):02d}", axis=1
    )

    qc_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(qc_dir / "p10_curated_windows.csv", index=False)
    trial_cur.to_csv(qc_dir / "p10_curated_trials.csv", index=False)

    active = windows[~windows["exclude_primary"]]
    excluded = windows[windows["exclude_primary"]]
    summary = {
        "checkpoint": "P10_curation",
        "subject": args.subject,
        "physical_eeg_files_deleted_or_modified": False,
        "primary_dataset": {
            "trials": int(trial_cur["exclude_primary"].eq(False).sum()),
            "all_windows_before_qc_selection": int(len(active)),
            "included_pass_windows": int(active["include_primary"].sum()),
            "review_windows_ablation_only": int((active["status"] == "review").sum()),
            "rejected_windows": int((active["status"] == "reject").sum()),
        },
        "excluded_trials": [
            int(x) for x in trial_cur.loc[trial_cur["exclude_primary"], "trial"]
        ],
        "excluded_trial_windows_retained_for_analysis": int(len(excluded)),
        "excluded_trial_pass_review_windows_available_for_explicit_ablation": int(
            excluded["status"].isin(["pass", "review"]).sum()
        ),
        "trial_exclusion_rules": {
            "pass_fraction_lt": pass_fraction_lt,
            "reject_fraction_gte": reject_fraction_gte,
            "unrecoverable_integrity_errors": integrity_issues,
            "computed_on": "4-second windows with 50% overlap",
        },
        "window_policy": {
            "pass": "primary_dataset",
            "review": "pass_plus_review_ablation_only",
            "reject": "excluded_from_all_train_validation_test_sets",
        },
        "label_policy": {
            "source": "self_report_valence_arousal",
            "fixed5": "primary",
            "fixed_threshold": fixed_threshold,
            "score_gt_threshold": "high",
            "score_lt_threshold": "low",
            "score_eq_threshold": "ambiguous_excluded_for_corresponding_task",
            "subject_relative": (
                "ablation_only; thresholds fitted on training trials inside each fold"
            ),
            "video_name_used_as_ground_truth": False,
        },
        "split_policy": (
            "group by trial; overlapping windows are never split independently"
        ),
        "rules_file": args.rules.as_posix(),
    }
    (qc_dir / "p10_curation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
