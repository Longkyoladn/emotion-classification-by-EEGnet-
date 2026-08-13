"""Run the approved subject-specific DENS pipeline with resumable logging."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from dens_cohort_config import (
    DEFAULT_COHORT_CONFIG,
    load_cohort_config,
    locked_subjects,
    project_path,
)


def build_steps(config: dict[str, Any]) -> list[tuple]:
    """Build pipeline commands from the paths locked in the cohort config."""
    dataset_root = project_path(config, "dataset_root").as_posix()
    manifest = project_path(config, "trial_manifest").as_posix()
    qc_root = project_path(config, "qc_root")
    processed_root = project_path(config, "processed_root").as_posix()

    return [
        (
            "bad_channels",
            "detect_dens_bad_channels.py",
            lambda s: [
                "--dataset-root",
                dataset_root,
                "--manifest",
                manifest,
                "--subject",
                s,
                "--output-dir",
                qc_root.as_posix(),
            ],
            "p4_bad_channel_summary.json",
        ),
        (
            "ica_fit",
            "propose_dens_ica_components.py",
            lambda s: [
                "--dataset-root",
                dataset_root,
                "--manifest",
                manifest,
                "--subject",
                s,
                "--output-dir",
                qc_root.as_posix(),
            ],
            "p6_ica_summary.json",
        ),
        (
            "ica_ablation",
            "ablate_dens_ica_subject.py",
            lambda s: [
                "--dataset-root",
                dataset_root,
                "--manifest",
                manifest,
                "--ica",
                (qc_root / s / "p6_ica_solution-ica.fif").as_posix(),
                "--subject",
                s,
                "--output-dir",
                qc_root.as_posix(),
            ],
            "p6b_ica_decision.json",
        ),
        (
            "trial_processing",
            "process_and_qc_dens_trials.py",
            lambda s: [
                "--dataset-root",
                dataset_root,
                "--manifest",
                manifest,
                "--ica",
                (qc_root / s / "p6_ica_solution-ica.fif").as_posix(),
                "--subject",
                s,
                "--processed-dir",
                processed_root,
                "--qc-dir",
                qc_root.as_posix(),
            ],
            "p7_p8_summary.json",
        ),
        (
            "window_qc",
            "qc_dens_windows.py",
            lambda s: [
                "--subject",
                s,
                "--processed-dir",
                processed_root,
                "--qc-dir",
                qc_root.as_posix(),
            ],
            "p9_summary.json",
        ),
        (
            "curation",
            "curate_dens_subject.py",
            lambda s: [
                "--subject",
                s,
                "--manifest",
                manifest,
                "--output-dir",
                qc_root.as_posix(),
            ],
            "p10_curation_summary.json",
        ),
    ]


def materialized(root: Path, subject: str, expected_fdt_bytes: int) -> bool:
    """Require one EEGLAB pair and the exact locked v1.0.2 payload size."""
    eeg_dir = root / subject / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    fdt_files = list(eeg_dir.glob("*_eeg.fdt"))
    return (
        len(set_files) == 1
        and len(fdt_files) == 1
        and set_files[0].stem == fdt_files[0].stem
        and set_files[0].stat().st_size > 1024
        and fdt_files[0].stat().st_size == expected_fdt_bytes
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_COHORT_CONFIG)
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--available-only", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    config = load_cohort_config(a.config)
    root = project_path(config, "dataset_root")
    qc_root = project_path(config, "qc_root")
    table = pd.read_csv(project_path(config, "subject_manifest"))
    manifest_eligible = set(
        table[
            table.has_subject_directory
            & table.has_eeg_metadata
            & table.has_events
            & table.has_behavior
        ].subject.tolist()
    )
    locked = locked_subjects(config)
    eligible = [subject for subject in locked if subject in manifest_eligible]
    subjects = a.subjects or list(config["subjects"]["default_processing"])
    unknown = sorted(set(subjects) - set(locked))
    if unknown:
        p.error(f"subjects are not in the locked cohort: {unknown}")
    missing_manifest = sorted(set(subjects) - manifest_eligible)
    if missing_manifest:
        p.error(f"subjects are incomplete in the manifest: {missing_manifest}")

    steps = build_steps(config)
    expected_bytes = config["expected_fdt_bytes"]
    env = os.environ.copy()
    cache = Path(".cache").resolve()
    (cache / "numba").mkdir(parents=True, exist_ok=True)
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "NUMBA_CACHE_DIR": str(cache / "numba"),
            "TEMP": str(cache / "tmp"),
            "TMP": str(cache / "tmp"),
        }
    )
    log_path = Path("reports/cohort_processing_status.csv")
    rows = []
    for subject in subjects:
        if not materialized(root, subject, expected_bytes[subject]):
            rows.append(
                {
                    "subject": subject,
                    "step": "raw",
                    "status": "pending_download",
                    "seconds": 0,
                    "error": "",
                }
            )
            if a.available_only:
                continue
            raise RuntimeError(f"Raw EEG is not materialized for {subject}")
        failed = False
        for step, script, args_fn, artifact in steps:
            target = qc_root / subject / artifact
            if target.exists() and not a.force:
                rows.append(
                    {
                        "subject": subject,
                        "step": step,
                        "status": "already_complete",
                        "seconds": 0,
                        "error": "",
                    }
                )
                continue
            start = time.perf_counter()
            command = [sys.executable, str(Path("scripts") / script), *args_fn(subject)]
            print("RUN", subject, step, flush=True)
            try:
                subprocess.run(command, check=True, env=env)
                status = "complete"
                error = ""
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"exit_code={exc.returncode}"
                failed = True
            rows.append(
                {
                    "subject": subject,
                    "step": step,
                    "status": status,
                    "seconds": round(time.perf_counter() - start, 2),
                    "error": error,
                }
            )
            pd.DataFrame(rows).to_csv(log_path, index=False)
            if failed:
                break
        print("DONE" if not failed else "FAILED", subject, flush=True)
    pd.DataFrame(rows).to_csv(log_path, index=False)
    summary = {
        "eligible_subjects": len(eligible),
        "requested": len(subjects),
        "fully_curated": int(
            sum(
                (qc_root / s / "p10_curation_summary.json").exists()
                for s in subjects
            )
        ),
        "failed_subjects": sorted(
            set(r["subject"] for r in rows if r["status"] == "failed")
        ),
        "pending_download": sorted(
            set(r["subject"] for r in rows if r["status"] == "pending_download")
        ),
    }
    Path("reports/cohort_processing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
