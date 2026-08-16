"""Lazy PyTorch Dataset/DataLoader support for cleaned trial FIF windows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .normalization import ChannelStandardizer, fit_channel_standardizer

DEFAULT_LABEL_MAPPING = {"low": 0, "high": 1}
DATASET_REQUIRED_COLUMNS = {
    "sample_id",
    "subject",
    "trial",
    "window_index",
    "start_sample",
    "stop_sample",
    "fif_path",
    "split_group",
    "sample_weight",
}


class EEGWindowDataset(Dataset):
    """Read fixed windows lazily from cleaned FIF trials."""

    def __init__(
        self,
        table: pd.DataFrame,
        project_root: Path,
        label_column: str = "label",
        label_mapping: dict[str, int] | None = None,
        normalizer: ChannelStandardizer | None = None,
        raw_cache_size: int = 8,
    ) -> None:
        missing = DATASET_REQUIRED_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(f"Dataset table is missing columns: {sorted(missing)}")
        if label_column not in table:
            raise ValueError(f"Dataset table is missing label column {label_column!r}")
        if table["sample_id"].duplicated().any():
            raise ValueError("Dataset table must contain unique sample_id values")
        if raw_cache_size < 1:
            raise ValueError("raw_cache_size must be positive")

        self.table = table.reset_index(drop=True).copy()
        self.project_root = Path(project_root)
        self.label_column = label_column
        self.label_mapping = label_mapping or DEFAULT_LABEL_MAPPING
        unknown_labels = set(self.table[label_column].dropna()) - set(self.label_mapping)
        if unknown_labels:
            raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")
        self.normalizer = normalizer
        self.raw_cache_size = raw_cache_size
        self._raw_cache: OrderedDict[str, mne.io.BaseRaw] = OrderedDict()

    def __len__(self) -> int:
        return len(self.table)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_raw_cache"] = OrderedDict()
        return state

    def _resolve_fif(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _raw(self, path: Path) -> mne.io.BaseRaw:
        key = str(path.resolve())
        if key in self._raw_cache:
            raw = self._raw_cache.pop(key)
            self._raw_cache[key] = raw
            return raw
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        self._raw_cache[key] = raw
        while len(self._raw_cache) > self.raw_cache_size:
            self._raw_cache.popitem(last=False)
        return raw

    def load_raw_window(self, index: int) -> np.ndarray:
        row = self.table.iloc[index]
        path = self._resolve_fif(str(row["fif_path"]))
        raw = self._raw(path)
        start = int(row["start_sample"])
        stop = int(row["stop_sample"])
        if start < 0 or stop <= start or stop > raw.n_times:
            raise IndexError(
                f"Invalid sample range [{start}, {stop}) for {path} with {raw.n_times} samples"
            )
        data = raw.get_data(picks="eeg", start=start, stop=stop, verbose="ERROR")
        expected = stop - start
        if data.shape[1] != expected:
            raise RuntimeError(f"Expected {expected} samples from {path}, got {data.shape[1]}")
        return np.asarray(data, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        data = self.load_raw_window(index)
        if self.normalizer is not None:
            data = self.normalizer.transform(data)
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(data)),
            "label": torch.tensor(
                self.label_mapping[str(row[self.label_column])], dtype=torch.long
            ),
            "sample_weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "subject": str(row["subject"]),
            "trial": int(row["trial"]),
            "window_index": int(row["window_index"]),
            "split_group": str(row["split_group"]),
        }


@dataclass
class DataLoaderBundle:
    """Partition loaders sharing statistics fitted on one training partition."""

    loaders: dict[str, DataLoader]
    datasets: dict[str, EEGWindowDataset]
    normalizer: ChannelStandardizer
    train_partition: str


def make_partition_dataloaders(
    modeling_index: pd.DataFrame,
    assignments: pd.DataFrame,
    project_root: Path,
    train_partition: str,
    label_column: str = "label",
    label_mapping: dict[str, int] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 20260814,
    raw_cache_size: int = 8,
) -> DataLoaderBundle:
    """Fit train-only normalization and reuse it for every requested partition."""
    required = {"sample_id", "partition", label_column}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignments are missing columns: {sorted(missing)}")
    if assignments["sample_id"].duplicated().any():
        raise ValueError(
            "Filter assignments to one fold/stage/scenario before building DataLoaders"
        )

    assignment_columns = ["sample_id", "partition", label_column]
    merged = assignments[assignment_columns].merge(
        modeling_index,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged["fif_path"].isna().any():
        missing_ids = merged.loc[merged["fif_path"].isna(), "sample_id"].tolist()
        raise ValueError(f"Assignments contain unknown sample IDs: {missing_ids[:5]}")
    if train_partition not in set(merged["partition"]):
        raise ValueError(f"Missing training partition {train_partition!r}")

    raw_train = EEGWindowDataset(
        merged[merged["partition"].eq(train_partition)],
        project_root=project_root,
        label_column=label_column,
        label_mapping=label_mapping,
        raw_cache_size=raw_cache_size,
    )
    normalizer = fit_channel_standardizer(raw_train, fit_partition=train_partition)

    datasets: dict[str, EEGWindowDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for partition, table in merged.groupby("partition", sort=False):
        dataset = EEGWindowDataset(
            table,
            project_root=project_root,
            label_column=label_column,
            label_mapping=label_mapping,
            normalizer=normalizer,
            raw_cache_size=raw_cache_size,
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=partition == train_partition,
            num_workers=num_workers,
            generator=generator,
        )
        datasets[str(partition)] = dataset
        loaders[str(partition)] = loader
    return DataLoaderBundle(
        loaders=loaders,
        datasets=datasets,
        normalizer=normalizer,
        train_partition=train_partition,
    )

