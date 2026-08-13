"""Compare EEGLAB SET dimensions with materialized or annex-pointer FDT bytes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

SIZE = re.compile(r"MD5E-s(\d+)--")


def declared_size(path: Path) -> int:
    if path.stat().st_size > 1024:
        return path.stat().st_size
    match = SIZE.search(path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else path.stat().st_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/ds003751"))
    p.add_argument(
        "--output", type=Path, default=Path("reports/dens_eeglab_integrity.csv")
    )
    a = p.parse_args()
    rows = []
    for folder in sorted(a.dataset_root.glob("sub-*")):
        sets = list((folder / "eeg").glob("*_eeg.set"))
        fdts = list((folder / "eeg").glob("*_eeg.fdt"))
        if len(sets) != 1 or len(fdts) != 1:
            continue
        try:
            eeg = loadmat(sets[0], squeeze_me=True, struct_as_record=False)["EEG"]
            expected = int(eeg.nbchan) * int(eeg.pnts) * int(eeg.trials) * 4
            actual = declared_size(fdts[0])
            issue = "" if expected == actual else "set_fdt_size_mismatch"
            rows.append(
                {
                    "subject": folder.name,
                    "nbchan": int(eeg.nbchan),
                    "pnts": int(eeg.pnts),
                    "trials": int(eeg.trials),
                    "expected_fdt_bytes": expected,
                    "declared_fdt_bytes": actual,
                    "size_ratio": actual / expected,
                    "integrity_status": "pass" if not issue else "fail",
                    "issue": issue,
                    "fdt_materialized": fdts[0].stat().st_size > 1024,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "subject": folder.name,
                    "integrity_status": "fail",
                    "issue": f"set_parse_error:{type(exc).__name__}",
                    "fdt_materialized": fdts[0].stat().st_size > 1024,
                }
            )
    table = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.output, index=False)
    print(
        table[
            ["subject", "integrity_status", "size_ratio", "fdt_materialized"]
        ].to_string(index=False)
    )
    print("\n", table.integrity_status.value_counts().to_dict())


if __name__ == "__main__":
    main()
