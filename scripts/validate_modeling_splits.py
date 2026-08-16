"""Validate modeling index, FIF ranges and every split protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_emotion.data.artifacts import validate_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modeling_m2_v1.0.2.json"),
    )
    parser.add_argument("--skip-fif-headers", action="store_true")
    args = parser.parse_args()
    report = validate_artifacts(
        PROJECT_ROOT,
        args.config,
        verify_fif_headers=not args.skip_fif_headers,
    )
    print(
        json.dumps(
            {
                "all_checks_pass": report["all_checks_pass"],
                "index_checks": report["modeling_index"]["checks"],
                "split_checks_pass": report["splits"]["all_checks_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

