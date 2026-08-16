"""Build the auditable window-level modeling index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_emotion.data.artifacts import build_index_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modeling_m2_v1.0.2.json"),
    )
    args = parser.parse_args()
    print(json.dumps(build_index_artifact(PROJECT_ROOT, args.config), indent=2))


if __name__ == "__main__":
    main()

