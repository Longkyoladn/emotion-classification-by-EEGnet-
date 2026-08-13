"""Download annexed DENS EEG files from the official OpenNeuro S3 bucket.

Pointer files are replaced only after a temporary download matches both the
expected byte count and MD5 stored in the DataLad annex pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import pandas as pd


POINTER = re.compile(r"MD5E-s(?P<size>\d+)--(?P<md5>[0-9a-f]{32})\.(?:set|fdt)")
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds003751"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointer_metadata(path: Path) -> tuple[int, str] | None:
    if path.stat().st_size > 1024:
        return None
    match = POINTER.search(path.read_text(encoding="utf-8").strip())
    if not match:
        return None
    return int(match["size"]), match["md5"]


def download(path: Path, dataset_root: Path) -> str:
    metadata = pointer_metadata(path)
    if metadata is None:
        return "already_materialized"
    expected_size, expected_md5 = metadata
    relative = path.relative_to(dataset_root).as_posix()
    url = f"{BASE_URL}/{relative}"
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"DOWNLOAD {relative} ({expected_size / 1024**2:.1f} MiB)", flush=True)
    subprocess.run(
        [
            "curl.exe", "--location", "--fail", "--retry", "5",
            "--connect-timeout", "30", "--output", str(temporary), url,
        ],
        check=True,
    )
    actual_size = temporary.stat().st_size
    actual_md5 = md5_file(temporary)
    if actual_size != expected_size or actual_md5 != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Integrity failure for {relative}: size {actual_size}/{expected_size}, "
            f"md5 {actual_md5}/{expected_md5}"
        )
    temporary.replace(path)
    print(f"VERIFIED {relative}", flush=True)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ds003751"))
    parser.add_argument("--subjects", type=Path, default=Path("data/manifest/dens_subjects.csv"))
    parser.add_argument("--exclude", nargs="*", default=["sub-mit003"])
    parser.add_argument("--include", nargs="*")
    parser.add_argument("--extensions", nargs="+", choices=["set", "fdt"], default=["set", "fdt"])
    parser.add_argument("--all-subject-directories", action="store_true")
    args = parser.parse_args()

    subjects = pd.read_csv(args.subjects)
    if args.all_subject_directories:
        eligible = sorted(path.name for path in args.dataset_root.glob("sub-*") if path.is_dir())
        eligible = [subject for subject in eligible if subject not in args.exclude]
    else:
        eligible = subjects[
            subjects["has_subject_directory"]
            & subjects["has_eeg_metadata"]
            & subjects["has_events"]
            & subjects["has_behavior"]
            & ~subjects["subject"].isin(args.exclude)
        ]["subject"].tolist()
    if args.include:
        eligible = [subject for subject in eligible if subject in args.include]
    totals = {"downloaded": 0, "already_materialized": 0}
    for subject in eligible:
        files = []
        for extension in args.extensions:
            files += sorted((args.dataset_root / subject / "eeg").glob(f"*_eeg.{extension}"))
        if len(files) != len(args.extensions):
            raise RuntimeError(f"Expected {len(args.extensions)} selected EEG files for {subject}, found {len(files)}")
        for path in files:
            state = download(path, args.dataset_root)
            totals[state] += 1
    print(f"DONE subjects={len(eligible)} {totals}", flush=True)


if __name__ == "__main__":
    main()
