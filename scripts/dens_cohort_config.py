"""Load and validate the locked DENS cohort configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_COHORT_CONFIG = Path("configs/cohort_dens_v1.0.2.json")


def load_cohort_config(path: Path = DEFAULT_COHORT_CONFIG) -> dict[str, Any]:
    """Load a cohort config and reject internally inconsistent subject lists."""
    config = json.loads(path.read_text(encoding="utf-8"))

    dataset = config["dataset"]
    subjects = config["subjects"]
    expected_bytes = config["expected_fdt_bytes"]

    if dataset["allow_other_snapshots"]:
        raise ValueError("The locked cohort must not allow other snapshots")

    locked = subjects["integrity_pass_locked"]
    if len(locked) != len(set(locked)):
        raise ValueError("Duplicate subject in integrity_pass_locked")

    if set(expected_bytes) != set(locked):
        raise ValueError(
            "expected_fdt_bytes must contain exactly the locked integrity subjects"
        )

    role_names = (
        "default_processing",
        "pilot",
        "modeling_primary",
        "sensitivity_only",
        "qc_only",
    )
    locked_set = set(locked)
    for role_name in role_names:
        role_subjects = subjects[role_name]
        if len(role_subjects) != len(set(role_subjects)):
            raise ValueError(f"Duplicate subject in role {role_name}")
        unknown = set(role_subjects) - locked_set
        if unknown:
            raise ValueError(
                f"Role {role_name} contains unlocked subjects: {sorted(unknown)}"
            )

    analysis_role_names = ("modeling_primary", "sensitivity_only", "qc_only")
    analysis_role_lists = [subjects[name] for name in analysis_role_names]
    analysis_subjects = [subject for role in analysis_role_lists for subject in role]
    if len(analysis_subjects) != len(set(analysis_subjects)):
        raise ValueError(
            "modeling_primary, sensitivity_only and qc_only must not overlap"
        )
    if set(analysis_subjects) != locked_set:
        raise ValueError(
            "modeling_primary, sensitivity_only and qc_only must partition the "
            "locked cohort"
        )

    invalid_sizes = {
        subject: size
        for subject, size in expected_bytes.items()
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0
    }
    if invalid_sizes:
        raise ValueError(f"Invalid expected_fdt_bytes values: {invalid_sizes}")

    return config


def locked_subjects(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the immutable ordered integrity-pass allowlist."""
    return tuple(config["subjects"]["integrity_pass_locked"])


def project_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a project-root-relative path from the config."""
    return Path(config["paths"][key])
